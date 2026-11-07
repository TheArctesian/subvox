import Foundation

enum TeacherFeatureMode: String, Codable, CaseIterable, Sendable { case body, invariant }

struct TeacherLearningRow: Sendable {
    var body: [Double]
    var invariant: [Double]
    var target: [Double]
    var head: [Double]
    var baseline: Bool
    func input(_ mode: TeacherFeatureMode) -> [Double] { mode == .body ? body : invariant }
}
struct TeacherLearningTake: Sendable {
    var repetition: Int
    var rows: [TeacherLearningRow]
    var validationGroup: Int? = nil
}
struct TeacherCandidate: Codable, Sendable {
    let mode: TeacherFeatureMode
    let ridge: Double
    let loss: Double
    let zeroLoss: Double
    var gain: Double { 1 - loss / max(1e-12, zeroLoss) }
}

/// Paired sensor supervision. Inputs have no camera measurements, labels, cue phase or clock.
/// A model predicts three mouth-shape velocities from a causal 240 ms IMU context.
struct MouthTeacherModel: Codable, Sendable {
    let mode: TeacherFeatureMode
    let ridge: Double
    let inputMean: [Double]
    let inputScale: [Double]
    let targetScale: [Double]
    let weights: [[Double]]
    let candidates: [TeacherCandidate]
    let sourceValidationGain: Double
    let shiftedTeacherGain: Double

    enum Failure: LocalizedError {
        case data, dynamics, model
        var errorDescription: String? {
            switch self {
            case .data: "Camera-teacher training needs usable paired motion and lip tracking in all three source repetitions."
            case .dynamics: "The camera teacher contains too little changing mouth shape to fit a student. Raw results are retained."
            case .model: "The saved camera-teacher model has incompatible or invalid dimensions."
            }
        }
    }

    func validate() throws {
        let dimensions = mode == .body ? 18 : 21
        guard inputMean.count == dimensions, inputScale.count == dimensions, targetScale.count == 3,
              weights.count == dimensions + 1, weights.allSatisfy({ $0.count == 3 && $0.allSatisfy(\.isFinite) }),
              inputMean.allSatisfy(\.isFinite), (inputScale + targetScale).allSatisfy({ $0.isFinite && $0 > 0 }),
              [0.1, 1, 10].contains(ridge), sourceValidationGain.isFinite, shiftedTeacherGain.isFinite else { throw Failure.model }
    }

    func predict(_ input: [Double]) -> [Double]? {
        guard input.count == inputMean.count,inputScale.count == input.count,weights.count == input.count+1,
              weights.allSatisfy({$0.count==3}),inputScale.allSatisfy({$0.isFinite && $0>0}),input.allSatisfy(\.isFinite) else { return nil }
        let x = [1.0] + input.indices.map { (input[$0] - inputMean[$0]) / inputScale[$0] }
        var result = [Double](repeating: 0, count: 3)
        for i in x.indices { for k in 0..<3 { result[k] += x[i] * weights[i][k] } }
        return result.allSatisfy(\.isFinite) ? result : nil
    }

    static func learn(_ takes: [TeacherLearningTake]) throws -> Self {
        guard Set(takes.map({ $0.validationGroup ?? $0.repetition })).count >= 2, takes.count >= 6,
              takes.allSatisfy({ $0.rows.count >= 50 && $0.rows.filter({ !$0.baseline }).count >= 35 }) else { throw Failure.data }
        var candidates: [TeacherCandidate] = [], nullCandidates: [TeacherCandidate] = []
        for mode in TeacherFeatureMode.allCases {
            for ridge in [0.1, 1.0, 10.0] {
                try Task.checkCancellation()
                candidates.append(try crossValidate(takes, mode: mode, ridge: ridge, shifted: false))
                nullCandidates.append(try crossValidate(takes, mode: mode, ridge: ridge, shifted: true))
            }
        }
        guard let best = candidates.min(by: { $0.loss < $1.loss }),
              let null = nullCandidates.min(by: { $0.loss < $1.loss }) else { throw Failure.data }
        let fitted = try fit(takes, mode: best.mode, ridge: best.ridge, shifted: false)
        return .init(mode: best.mode, ridge: best.ridge, inputMean: fitted.inputMean,
            inputScale: fitted.inputScale, targetScale: fitted.targetScale, weights: fitted.weights,
            candidates: candidates, sourceValidationGain: best.gain, shiftedTeacherGain: null.gain)
    }

    private static func crossValidate(_ takes: [TeacherLearningTake], mode: TeacherFeatureMode, ridge: Double, shifted: Bool) throws -> TeacherCandidate {
        var error = 0.0, zero = 0.0, count = 0.0
        for group in Set(takes.map({$0.validationGroup ?? $0.repetition})).sorted() {
            let training = takes.filter { ($0.validationGroup ?? $0.repetition) != group }
            let model = try fit(training, mode: mode, ridge: ridge, shifted: shifted)
            for take in takes where (take.validationGroup ?? take.repetition) == group {
                var takeError = 0.0, takeZero = 0.0
                for row in take.rows {
                    guard let predicted = model.predict(row.input(mode)) else { throw Failure.data }
                    for k in 0..<3 {
                        let target = row.target[k] / model.targetScale[k]
                        takeError += pow(predicted[k] - target, 2); takeZero += target * target
                    }
                }
                error += takeError / Double(take.rows.count * 3)
                zero += takeZero / Double(take.rows.count * 3); count += 1
            }
        }
        return .init(mode: mode, ridge: ridge, loss: error / count, zeroLoss: zero / count)
    }

    private static func fit(_ takes: [TeacherLearningTake], mode: TeacherFeatureMode, ridge: Double, shifted: Bool) throws -> Self {
        var rows: [TeacherLearningRow] = []
        for take in takes {
            var copy = take.rows
            if shifted {
                let indices = copy.indices.filter { !copy[$0].baseline }
                for (i, index) in indices.enumerated() { copy[index].target = take.rows[indices[(i + indices.count / 2) % indices.count]].target }
            }
            rows += copy
        }
        let dimensions = mode == .body ? 18 : 21
        guard rows.count >= 100, rows.allSatisfy({ row in
            row.input(mode).count == dimensions && row.target.count == 3 && row.head.count == 4 &&
                (row.input(mode) + row.target + row.head).allSatisfy(\.isFinite)
        }) else { throw Failure.data }
        let n = Double(rows.count)
        let mean = (0..<dimensions).map { k in rows.reduce(0) { $0 + $1.input(mode)[k] } / n }
        let scale = (0..<dimensions).map { k in max(1e-5, sqrt(rows.reduce(0) { $0 + pow($1.input(mode)[k] - mean[k], 2) } / n)) }
        let targetMean = (0..<3).map { k in rows.reduce(0) { $0 + $1.target[k] } / n }
        let rawTargetScale = (0..<3).map { k in sqrt(rows.reduce(0) { $0 + pow($1.target[k] - targetMean[k], 2) } / n) }
        guard rawTargetScale.contains(where: { $0 > 0.001 }) else { throw Failure.dynamics }
        let targetScale = rawTargetScale.map { max(0.01, $0) }
        let baseline = rows.filter(\.baseline), bn = Double(baseline.count)
        guard baseline.count >= 35 else { throw Failure.data }
        let headMean = (0..<4).map { k in baseline.reduce(0) { $0 + $1.head[k] } / bn }
        let headScale = (0..<4).map { k in max(0.01, sqrt(baseline.reduce(0) { $0 + pow($1.head[k] - headMean[k], 2) } / bn)) }
        let p = dimensions + 1
        var gram = Array(repeating: Array(repeating: 0.0, count: p), count: p)
        var rhs = Array(repeating: Array(repeating: 0.0, count: 3), count: p)
        var covariance = Array(repeating: Array(repeating: 0.0, count: 4), count: p)
        for row in rows {
            let input = row.input(mode)
            let z = [1.0] + (0..<dimensions).map { (input[$0] - mean[$0]) / scale[$0] }
            for i in 0..<p {
                for j in 0..<p { gram[i][j] += z[i] * z[j] / n }
                for k in 0..<3 { rhs[i][k] += z[i] * row.target[k] / targetScale[k] / n }
                if row.baseline && i > 0 {
                    for k in 0..<4 { covariance[i][k] += z[i] * (row.head[k] - headMean[k]) / headScale[k] / bn }
                }
            }
        }
        for i in 0..<p {
            if i > 0 { gram[i][i] += ridge }
            for j in 0..<p { gram[i][j] += 0.5 * zip(covariance[i], covariance[j]).reduce(0) { $0 + $1.0 * $1.1 } }
        }
        let weights = try solve(gram, rhs)
        return .init(mode: mode, ridge: ridge, inputMean: mean, inputScale: scale,
            targetScale: targetScale, weights: weights, candidates: [], sourceValidationGain: 0, shiftedTeacherGain: 0)
    }

    fileprivate static func solve(_ matrix: [[Double]], _ rhs: [[Double]]) throws -> [[Double]] {
        let n = matrix.count, m = 3
        var a = matrix.indices.map { matrix[$0] + rhs[$0] }
        for i in 0..<n {
            let pivot = (i..<n).max { abs(a[$0][i]) < abs(a[$1][i]) }!
            a.swapAt(i, pivot)
            guard abs(a[i][i]) > 1e-10 else { throw Failure.model }
            let value = a[i][i]
            for j in i..<(n+m) { a[i][j] /= value }
            for r in 0..<n where r != i {
                let factor = a[r][i]
                for j in i..<(n+m) { a[r][j] -= factor * a[i][j] }
            }
        }
        return a.map { Array($0.suffix(m)) }
    }
}

/// A deliberately small discriminative alternative to nearest-template matching.
/// It uses four relative-time bins of predicted mouth velocity, with fixed ridge=1.
struct TeacherLinearClassifier: Codable, Sendable {
    let mean:[Double]
    let scale:[Double]
    let weights:[[Double]]
    func validate() throws {
        guard [24,48].contains(mean.count),scale.count==mean.count,weights.count==mean.count+1,mean.allSatisfy(\.isFinite),
              scale.allSatisfy({$0.isFinite && $0>0}),weights.allSatisfy({$0.count==3 && $0.allSatisfy(\.isFinite)}) else {throw MouthTeacherModel.Failure.model}
    }
    static func features(_ sequence:TeacherSequence)->[Double]? {
        guard sequence.valid,[3,6].contains(sequence.values[0].count),let first=sequence.times.first,let last=sequence.times.last,last>first else {return nil}
        let dimensions=sequence.values[0].count
        var result:[Double]=[]
        for bin in 0..<4 {
            let rows=sequence.values.indices.filter {min(3,Int((sequence.times[$0]-first)/(last-first)*4))==bin}.map {sequence.values[$0]}
            guard rows.count>=2 else {return nil}
            let average=(0..<dimensions).map {k in rows.reduce(0){$0+$1[k]}/Double(rows.count)}
            let sd=(0..<dimensions).map {k in sqrt(rows.reduce(0){$0+pow($1[k]-average[k],2)}/Double(rows.count))}
            result += average+sd
        }
        return result
    }
    static func fit(_ examples:[(Int,TeacherSequence)]) throws -> Self {
        guard examples.allSatisfy({(0..<3).contains($0.0)}),(0..<3).allSatisfy({ label in examples.filter({$0.0==label}).count>=3 }) else {throw MouthTeacherModel.Failure.data}
        let rows=examples.compactMap {example -> (Int,[Double])? in guard let x=features(example.1) else{return nil};return(example.0,x)}
        guard rows.count==examples.count,let dimensions=rows.first?.1.count,rows.allSatisfy({$0.1.count==dimensions}) else {throw MouthTeacherModel.Failure.data}
        let n=Double(rows.count),mean=(0..<dimensions).map {k in rows.reduce(0){$0+$1.1[k]}/Double(rows.count)}
        let scale=(0..<dimensions).map {k in max(1e-4,sqrt(rows.reduce(0){$0+pow($1.1[k]-mean[k],2)}/n))}
        var gram=Array(repeating:Array(repeating:0.0,count:dimensions+1),count:dimensions+1),rhs=Array(repeating:Array(repeating:0.0,count:3),count:dimensions+1)
        for (label,row) in rows {
            let x=[1.0]+(0..<dimensions).map {(row[$0]-mean[$0])/scale[$0]}
            for i in 0...dimensions {for j in 0...dimensions {gram[i][j] += x[i]*x[j]/n};rhs[i][label] += x[i]/n}
        }
        for i in 1...dimensions {gram[i][i] += 1}
        return .init(mean:mean,scale:scale,weights:try MouthTeacherModel.solve(gram,rhs))
    }
    func predict(_ sequence:TeacherSequence)->Int? {
        guard let values=Self.features(sequence),values.count==mean.count,scale.count==mean.count,weights.count==mean.count+1,
              weights.allSatisfy({$0.count==3}),scale.allSatisfy({$0.isFinite && $0>0}) else {return nil}
        let x=[1.0]+values.indices.map {(values[$0]-mean[$0])/scale[$0]}
        let scores=(0..<3).map {k in x.indices.reduce(0){$0+x[$1]*weights[$1][k]}}
        guard scores.allSatisfy(\.isFinite),let best=scores.indices.max(by:{scores[$0]<scores[$1]}),
              !scores.indices.contains(where:{$0 != best && abs(scores[$0]-scores[best])<1e-9}) else {return nil}
        return best
    }
}

struct TeacherSequence: Codable, Sendable {
    let times: [Double]
    let values: [[Double]]
    var valid: Bool {
        guard times.count >= 12, times.count <= 1200, values.count == times.count,
              let dimensions = values.first?.count, (1...24).contains(dimensions),
              times.allSatisfy(\.isFinite), values.allSatisfy({ $0.count == dimensions && $0.allSatisfy(\.isFinite) }) else { return false }
        return zip(times,times.dropFirst()).allSatisfy { $1 > $0 && $1 - $0 <= 0.12 + 1e-7 }
    }
    func centred() -> [[Double]] {
        guard let dims = values.first?.count, !values.isEmpty else { return [] }
        let means = (0..<dims).map { k in values.reduce(0) { $0 + $1[k] } / Double(values.count) }
        return values.map { row in row.indices.map { row[$0] - means[$0] } }
    }
    static func scales(_ sequences: [Self]) -> [Double] {
        let frames = sequences.flatMap { $0.centred() }
        guard let dims = frames.first?.count, !frames.isEmpty else { return [] }
        return (0..<dims).map { k in max(1e-4, sqrt(frames.reduce(0) { $0 + $1[k] * $1[k] } / Double(frames.count))) }
    }
    func shifted() -> Self {
        .init(times: times, values: values.indices.map { values[($0 + values.count / 2) % values.count] })
    }
    func distance(to other: Self, scales: [Double]) -> Double? {
        guard valid, other.valid, scales.count == values[0].count, other.values[0].count == scales.count,
              scales.allSatisfy({ $0.isFinite && $0 >= 1e-4 }) else { return nil }
        let a = centred(), b = other.centred(), n = a.count, m = b.count
        let duration = times[n-1]-times[0], otherDuration = other.times[m-1]-other.times[0]
        guard duration > 0, otherDuration > 0 else { return nil }
        var previous = [Double](repeating: .infinity, count: m+1), previousLength = [Int](repeating: 0, count: m+1)
        previous[0] = 0
        for i in 1...n {
            var current = [Double](repeating: .infinity, count: m+1), length = [Int](repeating: 0, count: m+1)
            for j in 1...m where abs((times[i-1]-times[0])/duration - (other.times[j-1]-other.times[0])/otherDuration) <= 0.35 {
                var cost = 0.0
                for k in scales.indices { cost += pow((a[i-1][k]-b[j-1][k])/scales[k],2) }
                var best = previous[j-1], count = previousLength[j-1]
                if previous[j]+0.02 < best { best = previous[j]+0.02; count = previousLength[j] }
                if current[j-1]+0.02 < best { best = current[j-1]+0.02; count = length[j-1] }
                if best.isFinite { current[j] = best + sqrt(cost/Double(scales.count)); length[j] = count+1 }
            }
            previous = current; previousLength = length
        }
        return previous[m].isFinite && previousLength[m] > 0 ? previous[m]/Double(previousLength[m]) : nil
    }
}
