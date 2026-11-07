import Foundation

/// Leave-one-block-out ridge classification with a within-block label permutation null.
/// Blocks are the exchangeable unit: each holds every gesture once, recorded close together.
enum ObservabilityAnalysis {
    static let version = 1
    static let baselineSeconds = 1.5
    /// The smallest block count at which every contrast can reach significance (see `smallestAttainableP`).
    static let minimumBlocks = 8

    static func analyse(_ takes: [ObservabilityTake], permutations: Int = 499, seed: UInt64 = 1) -> ObservabilityReport {
        var rows: [(take: ObservabilityTake, features: [Double])] = []
        for take in takes where take.discardedReason == nil {
            if let f = features(take) { rows.append((take, f)) }
        }
        var results: [ObservabilityContrastResult] = []
        var insufficient: [String] = []
        for (offset, contrast) in ObservabilityContrast.all.enumerated() {
            let subset = rows.filter { contrast.gestures.contains($0.take.gesture) }
            let complete = Set(subset.map(\.take.block)).filter { block in
                contrast.gestures.allSatisfy { g in subset.contains { $0.take.block == block && $0.take.gesture == g } }
            }.count
            guard smallestAttainableP(classes: contrast.gestures.count, completeBlocks: complete) <= ObservabilityContrast.alpha / 2 else {
                insufficient.append(contrast.id); continue
            }
            let labels = subset.map { contrast.gestures.firstIndex(of: $0.take.gesture)! }
            let result = test(x: subset.map(\.features), labels: labels, blocks: subset.map(\.take.block),
                              classes: contrast.gestures.count, permutations: permutations, seed: seed &+ UInt64(offset))
            results.append(.init(contrast: contrast, takes: subset.count, blocks: Set(subset.map(\.take.block)).count,
                                 balancedAccuracy: result.accuracy, chance: 1 / Double(contrast.gestures.count),
                                 pValue: result.p, permutations: permutations, confusion: result.confusion))
        }
        return .init(results: results, insufficient: insufficient, verdict: verdict(results, insufficient: insufficient))
    }

    /// Relabelling every block the same way leaves accuracy unchanged, so even perfect separation
    /// ties with k! of the (k!)^B within-block permutations: p cannot fall below 1/(k!)^(B-1).
    static func smallestAttainableP(classes: Int, completeBlocks: Int) -> Double {
        guard completeBlocks >= 2 else { return 1 }
        let factorial = Double((1...max(1, classes)).reduce(1, *))
        return pow(factorial, -Double(completeBlocks - 1))
    }

    static func verdict(_ results: [ObservabilityContrastResult], insufficient: [String]) -> ObservabilityReport.Verdict {
        func significant(_ c: ObservabilityContrast) -> Bool? { results.first { $0.contrast.id == c.id }?.significant }
        if significant(.control) == true { return .artifact }
        guard let movement = significant(.movement), insufficient.isEmpty else { return .insufficient }
        if !movement { return .noSignal }
        return significant(.vowels) == true ? .articulation : .jawOnly
    }

    /// 30 features: per-axis deviation from the pre-cue baseline (std, peak, mean speed) for
    /// acceleration, rotation rate, pitch and roll, plus acceleration/rotation magnitude in each third of the cue.
    static func features(_ take: ObservabilityTake) -> [Double]? {
        let baseline = take.samples.filter { $0.t >= take.cueStart - baselineSeconds && $0.t < take.cueStart }
        let window = take.samples.filter { $0.t >= take.cueStart && $0.t <= take.cueEnd }
        guard baseline.count >= 10, window.count >= 40, take.cueEnd - take.cueStart >= 1.5,
              let first = window.first, let last = window.last, last.t > first.t else { return nil }
        let dt = (last.t - first.t) / Double(window.count - 1)
        let channels: [(MotionSample) -> Double] = [{ $0.ax }, { $0.ay }, { $0.az }, { $0.rx }, { $0.ry }, { $0.rz }, { $0.pitch }, { $0.roll }]
        var out: [Double] = []
        for channel in channels {
            let rest = baseline.map(channel).reduce(0, +) / Double(baseline.count)
            let v = window.map { channel($0) - rest }
            let mean = v.reduce(0, +) / Double(v.count)
            out.append(sqrt(v.reduce(0) { $0 + pow($1 - mean, 2) } / Double(v.count)))
            out.append(v.map(abs).max() ?? 0)
            out.append(zip(v, v.dropFirst()).reduce(0) { $0 + abs($1.1 - $1.0) } / Double(v.count - 1) / dt)
        }
        let thirds = (0..<3).map { k in
            window.filter { $0.t >= first.t + Double(k) * (last.t - first.t) / 3 && $0.t <= first.t + Double(k + 1) * (last.t - first.t) / 3 }
        }
        for magnitude in [{ (s: MotionSample) in sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az) },
                          { (s: MotionSample) in sqrt(s.rx * s.rx + s.ry * s.ry + s.rz * s.rz) }] {
            for part in thirds { out.append(part.isEmpty ? 0 : part.map(magnitude).reduce(0, +) / Double(part.count)) }
        }
        return out.allSatisfy(\.isFinite) ? out : nil
    }

    struct TestResult { var accuracy: Double; var p: Double; var confusion: [[Int]] }

    static func test(x: [[Double]], labels: [Int], blocks: [Int], classes: Int, permutations: Int, seed: UInt64) -> TestResult {
        let (observed, confusion) = crossValidate(x: x, labels: labels, blocks: blocks, classes: classes)
        var rng = SplitMix64(seed: seed)
        let groups = Dictionary(grouping: labels.indices, by: { blocks[$0] })
        var exceed = 0
        for _ in 0..<permutations {
            var shuffled = labels
            for indices in groups.values {
                let permuted = indices.map { labels[$0] }.shuffled(using: &rng)
                for (i, value) in zip(indices, permuted) { shuffled[i] = value }
            }
            if crossValidate(x: x, labels: shuffled, blocks: blocks, classes: classes).accuracy >= observed - 1e-12 { exceed += 1 }
        }
        return .init(accuracy: observed, p: Double(exceed + 1) / Double(permutations + 1), confusion: confusion)
    }

    static func crossValidate(x: [[Double]], labels: [Int], blocks: [Int], classes: Int) -> (accuracy: Double, confusion: [[Int]]) {
        var confusion = Array(repeating: Array(repeating: 0, count: classes), count: classes)
        for block in Set(blocks).sorted() {
            let train = x.indices.filter { blocks[$0] != block }
            let model = Ridge(x: train.map { x[$0] }, labels: train.map { labels[$0] }, classes: classes)
            for i in x.indices where blocks[i] == block { confusion[labels[i]][model?.predict(x[i]) ?? 0] += 1 }
        }
        let recalls = confusion.enumerated().compactMap { truth, row -> Double? in
            let total = row.reduce(0, +)
            return total > 0 ? Double(row[truth]) / Double(total) : nil
        }
        return (recalls.isEmpty ? 0 : recalls.reduce(0, +) / Double(recalls.count), confusion)
    }

    /// One-vs-rest least squares on standardised features with a fixed unit ridge, solved in place.
    struct Ridge {
        var mean: [Double], scale: [Double], weights: [Double], p: Int, classes: Int
        init?(x: [[Double]], labels: [Int], classes: Int) {
            guard let p = x.first?.count, x.count >= 2 else { return nil }
            let n = Double(x.count), d = p + 1
            mean = (0..<p).map { k in x.reduce(0) { $0 + $1[k] } / n }
            let mean = mean
            scale = (0..<p).map { k in max(1e-9, sqrt(x.reduce(0) { $0 + pow($1[k] - mean[k], 2) } / n)) }
            self.p = p; self.classes = classes
            var a = [Double](repeating: 0, count: d * (d + classes))
            let width = d + classes
            var z = [Double](repeating: 1, count: d)
            for (row, label) in zip(x, labels) {
                for k in 0..<p { z[k + 1] = (row[k] - mean[k]) / scale[k] }
                for i in 0..<d {
                    for j in 0..<d { a[i * width + j] += z[i] * z[j] }
                    a[i * width + d + label] += z[i]
                }
            }
            for i in 1..<d { a[i * width + i] += 1 }
            for i in 0..<d {
                var pivot = i
                for r in (i + 1)..<max(i + 1, d) where abs(a[r * width + i]) > abs(a[pivot * width + i]) { pivot = r }
                if pivot != i { for j in 0..<width { a.swapAt(i * width + j, pivot * width + j) } }
                let value = a[i * width + i]
                guard abs(value) > 1e-12 else { return nil }
                for j in 0..<width { a[i * width + j] /= value }
                for r in 0..<d where r != i {
                    let factor = a[r * width + i]
                    if factor != 0 { for j in 0..<width { a[r * width + j] -= factor * a[i * width + j] } }
                }
            }
            weights = (0..<d).flatMap { i in (0..<classes).map { a[i * width + d + $0] } }
        }
        func predict(_ row: [Double]) -> Int {
            var best = 0, bestScore = -Double.infinity
            for c in 0..<classes {
                var score = weights[c]
                for k in 0..<p { score += (row[k] - mean[k]) / scale[k] * weights[(k + 1) * classes + c] }
                if score > bestScore { best = c; bestScore = score }
            }
            return best
        }
    }
}
