import Foundation

struct MotionWordAnchor: Codable, Hashable, Sendable {
    var word: String
    var start: Double
    var end: Double
}

struct TransferredMotionWord: Codable, Hashable, Sendable {
    var word: String
    var voicedStart: Double
    var voicedEnd: Double
    var silentStart: Double
    var silentEnd: Double
}

struct MotionAlignmentPoint: Codable, Hashable, Sendable {
    /// Indices into the original, supplied arrays (no trimming or resampling).
    var voicedIndex: Int
    var silentIndex: Int
}

struct MotionTransferResult: Codable, Hashable, Sendable {
    /// Mean standardized RMS frame cost plus 0.02 per non-diagonal path step.
    /// A pair-local diagnostic, not a calibrated probability or model distance.
    var normalizedCost: Double
    /// Silent elapsed time divided by voiced elapsed time, independent of sample count.
    var durationRatio: Double
    var path: [MotionAlignmentPoint]
    var words: [TransferredMotionWord]
    var warnings: [String]
}

/// Annotation transfer within ONE paired recording. No model fitting, transcript matching,
/// phone inference, trimming or global normalization. Anchors never influence the DTW path.
enum SentenceMotionAlignment {
    // Bump when changing frame preprocessing, path constraints or cost semantics.
    static let algorithmVersion = 1
    static let timeBand = 0.35
    static let nonDiagonalPenalty = 0.02
    enum Failure: Error, LocalizedError, Equatable {
        case invalidFeatures, invalidScales, invalidSamples, excessiveLength, timestampGap
        case noMotion, badAnchors, excessiveWarp, noPath, collapsedWord

        var errorDescription: String? {
            switch self {
            case .invalidScales: "Frozen channel scales must be finite, positive and match the selected features."
            case .invalidFeatures: "Choose supported motion features for alignment."
            case .invalidSamples: "Motion must contain finite values and strictly increasing timestamps."
            case .excessiveLength: "Align clips with 6–2,400 samples and durations from 0.1 to 48 seconds."
            case .timestampGap: "Motion has a sampling gap or a sample rate below 10 Hz."
            case .noMotion: "Both recordings need informative motion in the selected channels."
            case .badAnchors: "Word anchors must be ordered, nonoverlapping and inside the voiced recording (40 ms tolerance)."
            case .excessiveWarp: "The silent/voiced duration ratio must be between 0.25 and 4."
            case .noPath: "No finite complete motion-alignment path was found."
            case .collapsedWord: "A word collapsed to zero duration; review this recording pair."
            }
        }
    }

    static func align(voiced: [MotionSample], silent: [MotionSample], words: [MotionWordAnchor],
                      features: FeatureSet = [.acceleration, .rotation], referenceScales: [Double]? = nil) throws -> MotionTransferResult {
        guard !features.isEmpty, features.subtracting(.all).isEmpty else { throw Failure.invalidFeatures }
        try validate(voiced)
        try validate(silent)
        let v0 = voiced[0].t, v1 = voiced[voiced.count - 1].t
        let s0 = silent[0].t, s1 = silent[silent.count - 1].t
        let ratio = (s1 - s0) / (v1 - v0)
        guard (0.25...4).contains(ratio) else { throw Failure.excessiveWarp }
        guard !words.isEmpty else { throw Failure.badAnchors }
        var previousEnd = v0 - 0.04
        for word in words {
            guard !word.word.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  word.start.isFinite, word.end.isFinite, word.end > word.start,
                  word.start >= v0 - 0.04, word.end <= v1 + 0.04,
                  word.start >= previousEnd - 1e-9,
                  min(v1, word.end) > max(v0, word.start) else { throw Failure.badAnchors }
            previousEnd = word.end
        }

        let a = MotionFeatures.frames(voiced, set: features)
        let b = MotionFeatures.frames(silent, set: features)
        var floors: [Double] = []
        if features.contains(.acceleration) { floors += [Double](repeating: 1e-5, count: 3) }
        if features.contains(.rotation) { floors += [Double](repeating: 1e-4, count: 3) }
        if features.contains(.attitude) { floors += [Double](repeating: 1e-4, count: 3) }
        func variances(_ frames: [[Double]]) -> [Double] {
            (0..<floors.count).map { d in frames.reduce(0) { $0 + $1[d] * $1[d] } / Double(frames.count) }
        }
        let va = variances(a), vb = variances(b)
        guard (va + vb).allSatisfy(\.isFinite) else { throw Failure.invalidSamples }
        guard zip(va, floors).contains(where: { $0.0 > $0.1 * $0.1 }),
              zip(vb, floors).contains(where: { $0.0 > $0.1 * $0.1 }) else { throw Failure.noMotion }
        // Pair-only scales give each recording equal weight even at different sampling rates.
        if let referenceScales {
            guard referenceScales.count == floors.count,
                  zip(referenceScales, floors).allSatisfy({ $0.isFinite && $0 >= $1 }) else { throw Failure.invalidScales }
        }
        // Fixed-model callers supply scales fitted only on their reference recordings.
        let scales = referenceScales ?? floors.indices.map { max(sqrt((va[$0] + vb[$0]) / 2), floors[$0]) }
        let x = MotionFeatures.scaled(a, by: scales), y = MotionFeatures.scaled(b, by: scales)
        let n = x.count, m = y.count, dims = scales.count
        let width = m + 1
        // Bounded memory: at most 2,401² bytes of predecessors plus two cost rows.
        var predecessors = [UInt8](repeating: 0, count: (n + 1) * width)
        var previous = [Double](repeating: .infinity, count: width)
        previous[0] = 0
        for i in 1...n {
            var current = [Double](repeating: .infinity, count: width)
            let vPhase = (voiced[i - 1].t - v0) / (v1 - v0)
            for j in 1...m {
                let sPhase = (silent[j - 1].t - s0) / (s1 - s0)
                // A relative-TIME band handles different native sample rates without losing
                // original indices. Its width is an annotation heuristic, not a learned prior.
                guard abs(vPhase - sPhase) <= timeBand else { continue }
                var squared = 0.0
                for d in 0..<dims { let delta = x[i - 1][d] - y[j - 1][d]; squared += delta * delta }
                let cost = sqrt(squared / Double(dims))
                // Stable tie preference: diagonal, then advance voiced, then advance silent.
                var best = previous[j - 1], direction: UInt8 = 1
                if previous[j] + nonDiagonalPenalty < best { best = previous[j] + nonDiagonalPenalty; direction = 2 }
                if current[j - 1] + nonDiagonalPenalty < best { best = current[j - 1] + nonDiagonalPenalty; direction = 3 }
                guard best.isFinite, cost.isFinite else { continue }
                current[j] = best + cost
                predecessors[i * width + j] = direction
            }
            previous = current
        }
        guard previous[m].isFinite else { throw Failure.noPath }
        var path: [MotionAlignmentPoint] = []
        var i = n, j = m
        while i > 0 && j > 0 {
            path.append(MotionAlignmentPoint(voicedIndex: i - 1, silentIndex: j - 1))
            switch predecessors[i * width + j] {
            case 1: i -= 1; j -= 1
            case 2: i -= 1
            case 3: j -= 1
            default: throw Failure.noPath
            }
        }
        guard i == 0, j == 0 else { throw Failure.noPath }
        path.reverse()
        let cost = previous[m] / Double(path.count)
        var totals = [Double](repeating: 0, count: n)
        var counts = [Int](repeating: 0, count: n)
        for point in path {
            // Relative arithmetic avoids loss of precision when the uptime clock is large.
            totals[point.voicedIndex] += silent[point.silentIndex].t - s0
            counts[point.voicedIndex] += 1
        }
        guard counts.allSatisfy({ $0 > 0 }) else { throw Failure.noPath }
        var mapped = totals.indices.map { totals[$0] / Double(counts[$0]) }
        mapped[0] = 0
        mapped[n - 1] = s1 - s0
        func transfer(_ time: Double) -> Double {
            if time <= v0 { return s0 }
            if time >= v1 { return s1 }
            var lo = 0, hi = n - 1
            while hi - lo > 1 {
                let mid = (lo + hi) / 2
                if voiced[mid].t <= time { lo = mid } else { hi = mid }
            }
            let fraction = (time - voiced[lo].t) / (voiced[hi].t - voiced[lo].t)
            return s0 + mapped[lo] + fraction * (mapped[hi] - mapped[lo])
        }
        var transferred: [TransferredMotionWord] = []
        for word in words {
            let start = transfer(word.start), end = transfer(word.end)
            guard start.isFinite, end.isFinite, end > start else { throw Failure.collapsedWord }
            transferred.append(TransferredMotionWord(word: word.word, voicedStart: word.start,
                voicedEnd: word.end, silentStart: start, silentEnd: end))
        }
        var warnings = ["Transferred word times are DTW estimates, not ground truth. No phone boundaries were inferred."]
        if words.contains(where: { $0.start < v0 || $0.end > v1 }) {
            warnings.append("Anchors within the 40 ms boundary tolerance were clipped to the motion recording.")
        }
        var warp = 0.0
        for point in path {
            let voicedPhase = (voiced[point.voicedIndex].t - v0) / (v1 - v0)
            let silentPhase = (silent[point.silentIndex].t - s0) / (s1 - s0)
            warp = max(warp, abs(voicedPhase - silentPhase))
        }
        if ratio < 2.0 / 3 || ratio > 1.5 || warp > 0.2 {
            warnings.append("Large time warp: review the paired motion and transferred words.")
        }
        if cost > 0.8 { warnings.append("High pair-local alignment cost: review before accepting annotations.") }
        return MotionTransferResult(normalizedCost: cost, durationRatio: ratio, path: path,
                                    words: transferred, warnings: warnings)
    }

    private static func validate(_ samples: [MotionSample]) throws {
        guard (6...2400).contains(samples.count) else { throw Failure.excessiveLength }
        guard samples.allSatisfy({ s in
            [s.t, s.ax, s.ay, s.az, s.rx, s.ry, s.rz, s.pitch, s.roll, s.yaw,
             s.gx, s.gy, s.gz, s.qw, s.qx, s.qy, s.qz].allSatisfy(\.isFinite)
        }) else { throw Failure.invalidSamples }
        let gaps = zip(samples, samples.dropFirst()).map { $1.t - $0.t }
        guard gaps.allSatisfy({ $0 > 0 && $0.isFinite }) else { throw Failure.invalidSamples }
        let duration = samples[samples.count - 1].t - samples[0].t
        // Uptime timestamps lose a few sub-nanosecond bits when subtracting large offsets.
        guard duration >= 0.1 - 1e-8, duration <= 48 + 1e-8 else { throw Failure.excessiveLength }
        let median = gaps.sorted()[gaps.count / 2]
        guard median <= 0.100001, !gaps.contains(where: { $0 > max(0.12, 3 * median) }) else { throw Failure.timestampGap }
    }
}
