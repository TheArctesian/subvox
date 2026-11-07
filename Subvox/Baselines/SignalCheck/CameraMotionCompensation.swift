import Foundation

/// An offline nuisance correction, fitted only on mouth-relaxed baseline intervals in A.
/// Mouth landmarks and phrase labels are deliberately absent from the regression inputs.
/// Original samples are never overwritten. All settings/coefficients stay fixed for B.
struct CameraHeadCorrection: Codable, Sendable {
    let id: UUID
    let sourceRunID: UUID
    let cameraID: String
    let estimator: String?
    let means: [Double]
    let scales: [Double]
    let coefficients: [[Double]] // 3 gyro channels × (intercept + 6 camera velocities)
    let heldOutReduction: Double
    let fittingRows: Int
    let validationRows: Int

    enum Failure: LocalizedError {
        case incomplete, missingTracking, insufficientCalibration, unhelpfulCalibration, incompatible
        var errorDescription: String? {
            switch self {
            case .incomplete: "Finish all 18 takes in block A to freeze the camera-assisted reference. Completed takes remain saved."
            case .missingTracking: "Camera tracking does not cover enough of the recordings. Raw results remain available."
            case .insufficientCalibration: "The relaxed-mouth baselines contain too little tracked head movement to learn a reliable correction. Raw motion is retained."
            case .unhelpfulCalibration: "The head-motion correction did not improve held-out baseline prediction by at least 10%. Corrected scores are unavailable; raw motion is retained."
            case .incompatible: "The camera-assisted reference does not match this recording or camera."
            }
        }
    }

    struct Row { let x: [Double]; let y: [Double]; let fold: Int }

    static func fit(_ run: SignalCheckRun) throws -> Self {
        guard let control = run.control, control.block == .unchanged, control.allowsMovement else { throw Failure.incompatible }
        let timeline = CameraHeadTimeline(control.readings, cameraID: control.target.cameraID, estimator: control.target.estimator)
        let takes = run.takes.filter { !$0.discarded }
        var rows: [Row] = []
        for (index, take) in takes.enumerated() {
            try Task.checkCancellation()
            let start = take.recording.cueStart
            // Exclude the button press and the final 0.5 s before speaking starts.
            for sample in take.recording.samples where sample.t >= start - 2.5 && sample.t <= start - 0.5 {
                guard let x = timeline.features(at: sample.t) else { continue }
                let y = [sample.rx, sample.ry, sample.rz]
                if y.allSatisfy(\.isFinite) { rows.append(.init(x: x, y: y, fold: index % 2)) }
            }
        }
        let train = rows.filter { $0.fold == 0 }, test = rows.filter { $0.fold == 1 }
        guard train.count >= 150, test.count >= 150, takes.count >= 6 else { throw Failure.insufficientCalibration }
        let trial = try regression(train)
        // Entire baseline intervals are held out, not interleaved neighboring frames.
        let meanY = (0..<3).map { k in train.map { $0.y[k] }.reduce(0,+) / Double(train.count) }
        var before = 0.0, after = 0.0
        for row in test {
            let predicted = predict(row.x, means: trial.0, scales: trial.1, coefficients: trial.2)
            for k in 0..<3 { before += pow(row.y[k]-meanY[k],2); after += pow(row.y[k]-predicted[k],2) }
        }
        guard before > Double(test.count) * 1e-6 else { throw Failure.insufficientCalibration }
        let reduction = 1 - after / before
        guard reduction.isFinite, reduction >= 0.1 else { throw Failure.unhelpfulCalibration }
        let full = try regression(rows)
        return .init(id: UUID(), sourceRunID: run.id, cameraID: control.target.cameraID,
            estimator: control.target.estimator, means: full.0, scales: full.1, coefficients: full.2,
            heldOutReduction: reduction, fittingRows: train.count, validationRows: test.count)
    }

    func correct(_ samples: [MotionSample], readings: [CameraPoseReading]) -> [MotionSample] {
        guard means.count == 6, scales.count == 6, scales.allSatisfy({ $0.isFinite && $0 > 0 }),
              coefficients.count == 3, coefficients.allSatisfy({ $0.count == 7 && $0.allSatisfy(\.isFinite) }) else { return [] }
        let timeline = CameraHeadTimeline(readings, cameraID: cameraID, estimator: estimator)
        return samples.compactMap { sample in
            guard let x = timeline.features(at: sample.t) else { return nil }
            let predicted = Self.predict(x, means: means, scales: scales, coefficients: coefficients)
            var result = sample
            result.rx -= predicted[0]; result.ry -= predicted[1]; result.rz -= predicted[2]
            // Acceleration and original sample files remain untouched. This diagnostic uses rotation only.
            return result
        }
    }

    private static func predict(_ x: [Double], means: [Double], scales: [Double], coefficients: [[Double]]) -> [Double] {
        let z = [1.0] + x.indices.map { (x[$0]-means[$0])/scales[$0] }
        return coefficients.map { zip($0,z).reduce(0) { $0 + $1.0 * $1.1 } }
    }

    private static func regression(_ rows: [Row]) throws -> ([Double], [Double], [[Double]]) {
        let n = Double(rows.count)
        let means = (0..<6).map { k in rows.map { $0.x[k] }.reduce(0,+)/n }
        let deviations = (0..<6).map { k in sqrt(rows.reduce(0) { $0 + pow($1.x[k]-means[k],2) }/n) }
        guard deviations.prefix(3).contains(where: { $0 > 0.005 }) || deviations.suffix(3).contains(where: { $0 > 0.005 }) else {
            throw Failure.insufficientCalibration
        }
        let scales = deviations.map { max(0.005,$0) }
        var gram = Array(repeating: Array(repeating: 0.0,count: 7),count: 7)
        var rhs = Array(repeating: Array(repeating: 0.0,count: 3),count: 7)
        for row in rows {
            let z = [1.0] + (0..<6).map { (row.x[$0]-means[$0])/scales[$0] }
            for i in 0..<7 {
                for j in 0..<7 { gram[i][j] += z[i]*z[j]/n }
                for k in 0..<3 { rhs[i][k] += z[i]*row.y[k]/n }
            }
        }
        for i in 1..<7 { gram[i][i] += 0.05 } // Fixed ridge penalty; never tuned on phrases or block B.
        var augmented = (0..<7).map { gram[$0]+rhs[$0] }
        for i in 0..<7 {
            let pivot = (i..<7).max { abs(augmented[$0][i]) < abs(augmented[$1][i]) }!
            augmented.swapAt(i,pivot)
            guard abs(augmented[i][i]) > 1e-10 else { throw Failure.insufficientCalibration }
            let scale = augmented[i][i]
            for j in i..<10 { augmented[i][j] /= scale }
            for row in 0..<7 where row != i {
                let factor = augmented[row][i]
                for j in i..<10 { augmented[row][j] -= factor*augmented[i][j] }
            }
        }
        let coefficients = (0..<3).map { k in (0..<7).map { augmented[$0][7+k] } }
        return (means, scales, coefficients)
    }
}

/// Capture-time alignment. Invalid frames remain in the timeline, so gaps, clock fallbacks,
/// camera switches and missing faces cannot silently be interpolated as valid tracking.
struct CameraHeadTimeline {
    let readings: [CameraPoseReading]
    let cameraID: String
    let estimator: String?
    init(_ readings: [CameraPoseReading], cameraID: String, estimator: String?) {
        self.readings = readings.filter { $0.t.isFinite }.sorted { $0.t < $1.t }
        self.cameraID = cameraID; self.estimator = estimator
    }
    func features(at time: Double) -> [Double]? {
        guard time.isFinite, let a = pose(at: time-0.12), let b = pose(at: time+0.12) else { return nil }
        // Require uninterrupted tracking across the derivative window as well as at its endpoints.
        let lo = lowerBound(time-0.12), hi = lowerBound(time+0.12)
        guard lo < readings.count, hi < readings.count else { return nil }
        for i in lo...hi {
            guard valid(readings[i]), i == lo || (readings[i].t > readings[i-1].t && readings[i].t-readings[i-1].t <= 0.12) else { return nil }
        }
        func angular(_ x: Double, _ y: Double) -> Double { atan2(sin((y-x) * .pi/180), cos((y-x) * .pi/180))/0.24 }
        let result = [angular(a.yaw,b.yaw),angular(a.pitch,b.pitch),angular(a.roll,b.roll),
            (b.x-a.x)/0.24,(b.y-a.y)/0.24,log(b.size/a.size)/0.24]
        return result.allSatisfy(\.isFinite) ? result : nil
    }
    private func valid(_ r: CameraPoseReading) -> Bool {
        r.cameraID == cameraID && r.estimator == estimator && r.pose?.isFinite == true && r.usedReceiptTime != true
    }
    private func lowerBound(_ t: Double) -> Int {
        var low = 0, high = readings.count
        while low < high { let middle = (low+high)/2; if readings[middle].t < t { low = middle+1 } else { high = middle } }
        return low
    }
    private func pose(at t: Double) -> CameraHeadPose? {
        let i = lowerBound(t)
        if i < readings.count, abs(readings[i].t-t)<1e-8 { return valid(readings[i]) ? readings[i].pose : nil }
        guard i > 0, i < readings.count else { return nil }
        let a=readings[i-1], b=readings[i]
        guard valid(a),valid(b),b.t>a.t,b.t-a.t<=0.12,let p=a.pose,let q=b.pose else { return nil }
        let f=(t-a.t)/(b.t-a.t)
        func angle(_ x:Double,_ y:Double)->Double { x + atan2(sin((y-x) * .pi/180),cos((y-x) * .pi/180))*180 / .pi*f }
        return .init(yaw:angle(p.yaw,q.yaw),pitch:angle(p.pitch,q.pitch),roll:angle(p.roll,q.roll),
            x:p.x+(q.x-p.x)*f,y:p.y+(q.y-p.y)*f,size:p.size+(q.size-p.size)*f)
    }
}

struct CameraMotionScore: Codable, Sendable {
    let condition: SpeechCondition
    let collected: Int
    let rawCorrect: Int
    let rawEvaluated: Int
    let correctedCorrect: Int?
    let correctedEvaluated: Int?
}

/// New, rotation-only A→B diagnostic. The older six-channel reference remains unchanged.
/// Both paths use identical takes, DTW settings and source-only scales; correction is frozen in A.
struct CameraMotionReference: Codable, Sendable {
    struct Group: Codable, Sendable { let condition: SpeechCondition; let rawScales: [Double]; let correctedScales: [Double]? }
    let id: UUID
    let sourceRunID: UUID
    let sourceSessionID: UUID
    let cameraID: String
    let algorithmVersion: Int
    let testID: UUID
    let targetID: UUID
    let seed: UInt64
    let frozenAt: Date
    let correction: CameraHeadCorrection?
    let correctionNote: String?
    let rawTemplates: [FrozenSignalReference.Template]
    let correctedTemplates: [FrozenSignalReference.Template]
    let groups: [Group]

    static func freeze(_ run: SignalCheckRun) throws -> Self {
        guard run.endedAt != nil, let control = run.control, control.block == .unchanged,control.allowsMovement,
              let session=run.session else { throw CameraHeadCorrection.Failure.incompatible }
        let takes=run.takes.filter { !$0.discarded }
        guard takes.count==18, takes.allSatisfy({ SignalCheckAnalysis.metrics($0).usable }) else { throw CameraHeadCorrection.Failure.incomplete }
        for c in [SpeechCondition.normal,.silent] {
            for sentence in 0..<3 {
                guard Set(takes.filter { $0.prompt.condition==c && $0.prompt.sentence==sentence }.map { $0.prompt.repetition }) == Set(0..<3) else { throw CameraHeadCorrection.Failure.incomplete }
            }
        }
        var correction:CameraHeadCorrection?, note:String?
        do { correction=try CameraHeadCorrection.fit(run) } catch { note=error.localizedDescription }
        func template(_ take:SignalCheckTake,_ samples:[MotionSample])->FrozenSignalReference.Template {
            .init(takeID:take.id,sentence:take.prompt.sentence,repetition:take.prompt.repetition,condition:take.prompt.condition,samples:samples)
        }
        var raw=takes.map { template($0,$0.recording.cueSamples) }
        var corrected:[FrozenSignalReference.Template]=[]
        if let fitted=correction {
            corrected=takes.compactMap { take in
                let clip=fitted.correct(take.recording.cueSamples,readings:control.readings)
                guard covered(clip,recording:take.recording), cost(clip,clip,scales:[1,1,1]) != nil else { return nil }
                return template(take,clip)
            }
            if corrected.count != 18 { correction=nil;corrected=[];note=CameraHeadCorrection.Failure.missingTracking.localizedDescription }
            else {
                raw=zip(raw,corrected).map { original, adjusted in
                    let times=Set(adjusted.samples.map(\.t))
                    return .init(takeID:original.takeID,sentence:original.sentence,repetition:original.repetition,
                        condition:original.condition,samples:original.samples.filter { times.contains($0.t) })
                }
            }
        }
        func scales(_ samples:[FrozenSignalReference.Template],_ c:SpeechCondition)->[Double] {
            MotionFeatures.channelScales(samples.filter { $0.condition==c }.map { MotionFeatures.frames($0.samples,set:.rotation) }).map { max(1e-4,$0) }
        }
        let groups=[SpeechCondition.normal,.silent].map { Group(condition:$0,rawScales:scales(raw,$0),correctedScales:correction == nil ? nil : scales(corrected,$0)) }
        return .init(id:UUID(),sourceRunID:run.id,sourceSessionID:session.id,cameraID:control.target.cameraID,algorithmVersion:1,
            testID:control.testID,targetID:control.target.id,seed:run.seed,frozenAt:Date(),
            correction:correction,correctionNote:note,rawTemplates:raw,correctedTemplates:corrected,groups:groups)
    }

    func evaluate(_ run:SignalCheckRun) throws -> [CameraMotionScore] {
        guard algorithmVersion==1,let control=run.control,control.block == .reseated,control.allowsMovement,
              control.testID==testID,control.target.id==targetID,run.seed==seed,control.pacing == .controlled,
              run.startedAt>=frozenAt,control.blockConfirmedAt>=frozenAt,
              control.target.cameraID==cameraID,run.id != sourceRunID,run.session?.id != sourceSessionID,
              run.reseatedAt.map({ $0>=frozenAt }) == true else { throw CameraHeadCorrection.Failure.incompatible }
        let keys=run.takes.filter { !$0.discarded }.map { "\($0.prompt.condition.rawValue):\($0.prompt.sentence):\($0.prompt.repetition)" }
        guard Set(keys).count==keys.count,Set(run.takes.map(\.id)).count==run.takes.count,run.takes.filter({ !$0.discarded }).allSatisfy({
            (0..<3).contains($0.prompt.sentence) && (0..<3).contains($0.prompt.repetition) && $0.recording.condition==$0.prompt.condition
        }) else { throw CameraHeadCorrection.Failure.incompatible }
        var result:[CameraMotionScore]=[]
        let quality=Dictionary(uniqueKeysWithValues:ControlledSignalComparison.qualities(run).map { ($0.takeID,$0.passes) })
        for group in groups {
            let takes=run.takes.filter { !$0.discarded && $0.prompt.condition==group.condition }
            var rawCorrect=0,rawCount=0,correctedCorrect=0,correctedCount=0
            for take in takes {
                try Task.checkCancellation()
                // Compare both paths on the same tracking/timing-qualified test takes.
                guard quality[take.id]==true,SignalCheckAnalysis.metrics(take).usable else { continue }
                var raw=take.recording.cueSamples
                var adjusted:[MotionSample]?
                if let correction {
                    let clip=correction.correct(raw,readings:control.readings)
                    guard Self.covered(clip,recording:take.recording) else { continue }
                    adjusted=clip
                    let timestamps=Set(clip.map(\.t));raw=raw.filter { timestamps.contains($0.t) }
                }
                rawCount+=1
                if Self.match(raw,templates:rawTemplates,condition:group.condition,scales:group.rawScales)==take.prompt.sentence { rawCorrect+=1 }
                if let adjusted,let scales=group.correctedScales {
                    correctedCount+=1
                    if Self.match(adjusted,templates:correctedTemplates,condition:group.condition,scales:scales)==take.prompt.sentence { correctedCorrect+=1 }
                }
            }
            result.append(.init(condition:group.condition,collected:takes.count,rawCorrect:rawCorrect,rawEvaluated:rawCount,
                correctedCorrect:correction == nil ? nil : correctedCorrect,correctedEvaluated:correction == nil ? nil : correctedCount))
        }
        return result
    }

    static func covered(_ clip:[MotionSample],recording:SentenceRecording)->Bool {
        let start=recording.cueStart,end=recording.cueEnd
        guard clip.count>=12, let first=clip.first,let last=clip.last,
              first.t-start<=0.12,end-last.t<=0.12 else { return false }
        return zip(clip,clip.dropFirst()).allSatisfy { $1.t>$0.t && $1.t-$0.t<=0.12 } &&
            last.t-first.t >= (end-start)*0.9
    }
    private static func match(_ samples:[MotionSample],templates:[FrozenSignalReference.Template],condition:SpeechCondition,scales:[Double])->Int? {
        let distances=templates.filter { $0.condition==condition }.compactMap { template -> (Int,Double)? in
            guard let cost=cost(samples,template.samples,scales:scales) else { return nil };return (template.sentence,cost)
        }
        guard Set(distances.map { $0.0 })==Set(0..<3),let best=distances.min(by: { $0.1<$1.1 }) else { return nil }
        // -1 represents a scored but ambiguous take, never a silently excluded tie.
        return distances.contains { $0.0 != best.0 && abs($0.1-best.1)<1e-9 } ? -1 : best.0
    }
    private static func cost(_ a:[MotionSample],_ b:[MotionSample],scales:[Double])->Double? {
        guard let start=a.first?.t,let end=a.last?.t else { return nil }
        return try? SentenceMotionAlignment.align(voiced:a,silent:b,words:[.init(word:"whole phrase",start:start,end:end)],features:.rotation,referenceScales:scales).normalizedCost
    }
}
