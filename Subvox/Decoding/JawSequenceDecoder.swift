import Foundation

/// Small-data experimental sequence model. Transcript positions are distinct even when
/// their emission distributions are tied to the same coarse jaw unit. Boundaries inferred
/// by Viterbi are estimates, not independently observed phone timing.
struct JawSequenceDecoder: Sendable {
    let wordCount: Int
    let missingPhones: [String]
    /// All buildable words belonging to a unit-sequence bucket containing multiple words.
    let collisionCount: Int
    let trainingSummary: String

    private struct Sequence: Sendable {
        var frames: [[Double]]
        var states: [Int]
        var weights: [Double]
    }
    private struct Gaussian: Sendable {
        var mean: [Double]
        var variance: [Double]
    }
    private struct Entry: Sendable {
        var word: String
        var states: [Int]
        var rarity: Double
    }
    private let featureSet: FeatureSet
    private let scales: [Double]
    private let models: [Gaussian]
    private let vocabulary: [Entry]
    // All channels are standardized from fitting takes. Floors and shrinkage are fixed
    // hyperparameters; neither is selected using validation/final recordings here.
    private static let varianceFloor = 0.05
    private static let priorFrames = 12.0
    private static let loopCost = -log(0.65)
    private static let advanceCost = -log(0.35)
    private static let maximumFrames = 4096
    private static let maximumStates = 256

    init?(takes: [Template], lexicon: Lexicon, vocabulary: [(word: String, count: Int)],
          featureSet: FeatureSet, iterations: Int = 6) {
        guard !featureSet.isEmpty, !Task.isCancelled else { return nil }
        var raw: [(frames: [[Double]], units: [String], weights: [Double])] = []
        var skipped = 0
        for take in takes {
            if Task.isCancelled { return nil }
            guard let phones = lexicon.phones(TrainingExperiment.key(take.label)), !phones.isEmpty,
                  phones.count <= Self.maximumStates,
                  let frames = Self.features(take.samples, featureSet: featureSet) else {
                skipped += 1; continue
            }
            let units = phones.compactMap { JawUnits.unit(for: $0) }
            guard units.count == phones.count, frames.count >= units.count else {
                skipped += 1; continue
            }
            raw.append((frames, units, phones.map { Lexicon.vowels.contains($0) ? 2 : 1 }))
        }
        guard !raw.isEmpty else { return nil }
        let scales = MotionFeatures.channelScales(raw.map(\.frames))
        guard !scales.isEmpty, scales.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        let units = Set(raw.flatMap(\.units)).sorted()
        let indices = Dictionary(uniqueKeysWithValues: units.enumerated().map { ($0.element, $0.offset) })
        let sequences = raw.map {
            Sequence(frames: MotionFeatures.scaled($0.frames, by: scales),
                     states: $0.units.map { indices[$0]! }, weights: $0.weights)
        }
        guard sequences.allSatisfy({ $0.frames.allSatisfy { $0.allSatisfy(\.isFinite) } }) else { return nil }
        let pooled = Self.pooledVariance(sequences, dims: scales.count)
        var assignments = sequences.map { Self.initialPath($0) }
        var models = Self.fit(sequences, assignments: assignments, count: units.count, pooled: pooled)
        let requestedIterations = min(6, max(0, iterations))
        // Shrinkage is regularized estimation, so a Viterbi refit need not improve the
        // unregularized likelihood. Keep the best finite normalized fitting objective,
        // including initialization, rather than blindly publishing the final iterate.
        var bestModels = models
        var bestAssignments = assignments
        var bestObjective = Self.objective(sequences, models: models)
        guard bestObjective.isFinite else { return nil }
        var bestIteration = 0
        var completed = 0
        var failedPaths = 0
        var changedFrames = 0
        for _ in 0..<requestedIterations {
            if Task.isCancelled { return nil }
            var next: [[Int]] = []
            var changed = 0
            for (sequence, previous) in zip(sequences, assignments) {
                if Task.isCancelled { return nil }
                let emissions = Self.emissions(sequence.frames, models: models)
                guard let path = Self.path(emissions, states: sequence.states) else {
                    // Retain the last valid alignment. Never silently drop frames or a unit.
                    failedPaths += 1; next.append(previous); continue
                }
                changed += zip(previous, path).filter { $0 != $1 }.count
                next.append(path)
            }
            assignments = next
            models = Self.fit(sequences, assignments: assignments, count: units.count, pooled: pooled)
            completed += 1
            changedFrames = changed
            let objective = Self.objective(sequences, models: models)
            if objective.isFinite, objective < bestObjective {
                bestObjective = objective
                bestModels = models
                bestAssignments = assignments
                bestIteration = completed
            }
            if changed == 0 { break }
        }
        guard !Task.isCancelled else { return nil }
        models = bestModels
        assignments = bestAssignments
        let maximumCount = Double(max(1, vocabulary.map(\.count).max() ?? 1))
        var entries: [Entry] = []
        var seenWords = Set<String>()
        var collisions: [String: Int] = [:]
        for entry in vocabulary.sorted(by: { $0.word < $1.word }) {
            let word = TrainingExperiment.key(entry.word)
            guard !seenWords.contains(word), let phones = lexicon.phones(word), !phones.isEmpty,
                  phones.count <= Self.maximumStates else { continue }
            let mapped = phones.compactMap { JawUnits.unit(for: $0) }.compactMap { indices[$0] }
            guard mapped.count == phones.count else { continue }
            seenWords.insert(word)
            entries.append(Entry(word: word, states: mapped,
                                 rarity: max(0, log(maximumCount / Double(max(1, entry.count))))))
            collisions[mapped.map(String.init).joined(separator: ","), default: 0] += 1
        }
        var occupancy = [Int](repeating: 0, count: units.count)
        for (sequence, path) in zip(sequences, assignments) {
            for position in path { occupancy[sequence.states[position]] += 1 }
        }
        self.featureSet = featureSet
        self.scales = scales
        self.models = models
        self.vocabulary = entries
        self.wordCount = entries.count
        self.missingPhones = Phoneme.all.map(\.symbol).filter {
            guard let unit = JawUnits.unit(for: $0) else { return true }
            return indices[unit] == nil
        }
        self.collisionCount = collisions.values.filter { $0 > 1 }.reduce(0, +)
        let counts = units.enumerated().map { "\($0.element)=\(occupancy[$0.offset])" }.joined(separator: ", ")
        self.trainingSummary = "\(JawUnits.version); \(sequences.count) fitting takes; \(skipped) skipped takes; "
            + "\(completed)/\(requestedIterations) realignment iterations; best checkpoint \(bestIteration), fitting score \(String(format: "%.6f", bestObjective)); \(changedFrames) changed frames in last iteration; "
            + "\(failedPaths) failed paths retained prior alignment; occupancy: \(counts). Boundaries are model estimates."
    }

    /// Scores are per-frame Gaussian negative log likelihoods shifted by a fixed floor
    /// constant, plus topology cost and optional frequency bias. Score gaps and confidence
    /// are uncalibrated ranking heuristics; DTW rejection thresholds do not transfer.
    func predict(_ samples: [MotionSample], maxScore: Double, minScoreGap: Double,
                 commonBias: Double, limit: Int = 8) -> Prediction {
        let empty = Prediction(ranked: [], confidence: 0, accepted: false)
        guard limit > 0, commonBias.isFinite, commonBias >= 0, !Task.isCancelled,
              let frames = Self.features(samples, featureSet: featureSet) else { return empty }
        let standardized = MotionFeatures.scaled(frames, by: scales)
        guard standardized.allSatisfy({ $0.allSatisfy(\.isFinite) }) else { return empty }
        let emissions = Self.emissions(standardized, models: models)
        // Keep two even if a caller asks for one: rejection must still use the runner-up.
        let retained = max(2, limit)
        var ranked: [RankedLabel] = []
        for entry in vocabulary {
            if Task.isCancelled { return empty }
            let cost = Self.score(emissions, states: entry.states)
            let score = cost / Double(frames.count) + commonBias * entry.rarity
            guard score.isFinite else { continue }
            let candidate = RankedLabel(label: entry.word, distance: score)
            let index = ranked.firstIndex {
                score < $0.distance || (score == $0.distance && entry.word < $0.label)
            } ?? ranked.count
            if index < retained {
                ranked.insert(candidate, at: index)
                if ranked.count > retained { ranked.removeLast() }
            }
        }
        guard let best = ranked.first else { return empty }
        // A single finite candidate supplies no comparative evidence for confidence.
        let gap = ranked.count > 1 ? max(0, ranked[1].distance - best.distance) : 0
        let confidence = 1 - exp(-gap)
        let accepted = !maxScore.isNaN && minScoreGap.isFinite && minScoreGap >= 0
            && best.distance <= maxScore && gap >= minScoreGap
        return Prediction(ranked: Array(ranked.prefix(limit)), confidence: confidence, accepted: accepted)
    }

    private static func features(_ samples: [MotionSample], featureSet: FeatureSet) -> [[Double]]? {
        guard !samples.isEmpty, samples.count <= maximumFrames else { return nil }
        for (index, sample) in samples.enumerated() {
            guard [sample.t, sample.ax, sample.ay, sample.az, sample.rx, sample.ry, sample.rz,
                   sample.pitch, sample.roll, sample.yaw].allSatisfy(\.isFinite),
                  index == 0 || sample.t > samples[index - 1].t else { return nil }
        }
        let frames = MotionFeatures.frames(MotionFeatures.trim(samples), set: featureSet)
        guard let dims = frames.first?.count, dims > 0,
              frames.allSatisfy({ $0.count == dims && $0.allSatisfy(\.isFinite) }) else { return nil }
        return frames
    }

    private static func initialPath(_ sequence: Sequence) -> [Int] {
        let count = sequence.frames.count
        let total = sequence.weights.reduce(0, +)
        var path = [Int](repeating: 0, count: count)
        var start = 0
        var cumulative = 0.0
        for state in sequence.states.indices {
            cumulative += sequence.weights[state]
            let remaining = sequence.states.count - state - 1
            let end = max(start + 1, min(count - remaining, Int((Double(count) * cumulative / total).rounded())))
            for frame in start..<end { path[frame] = state }
            start = end
        }
        return path
    }

    private static func pooledVariance(_ sequences: [Sequence], dims: Int) -> [Double] {
        var sum = [Double](repeating: 0, count: dims)
        var squares = sum
        var count = 0.0
        for sequence in sequences {
            for frame in sequence.frames {
                count += 1
                for d in 0..<dims { sum[d] += frame[d]; squares[d] += frame[d] * frame[d] }
            }
        }
        return (0..<dims).map { max(varianceFloor, squares[$0] / count - pow(sum[$0] / count, 2)) }
    }

    private static func fit(_ sequences: [Sequence], assignments: [[Int]], count: Int,
                            pooled: [Double]) -> [Gaussian] {
        let dims = pooled.count
        var sums = [[Double]](repeating: [Double](repeating: 0, count: dims), count: count)
        var squares = sums
        var occupancy = [Double](repeating: 0, count: count)
        for (sequence, path) in zip(sequences, assignments) {
            for (frame, position) in zip(sequence.frames, path) {
                let unit = sequence.states[position]
                occupancy[unit] += 1
                for d in 0..<dims { sums[unit][d] += frame[d]; squares[unit][d] += frame[d] * frame[d] }
            }
        }
        return (0..<count).map { unit in
            let n = occupancy[unit]
            guard n > 0 else { return Gaussian(mean: [Double](repeating: 0, count: dims), variance: pooled) }
            let means = sums[unit].map { $0 / n }
            let variance = (0..<dims).map { d -> Double in
                let local = max(0, squares[unit][d] / n - means[d] * means[d])
                return max(varianceFloor, (n * local + priorFrames * pooled[d]) / (n + priorFrames))
            }
            return Gaussian(mean: means, variance: variance)
        }
    }

    /// Frame-weighted, complete-path negative log likelihood on fitting takes only.
    private static func objective(_ sequences: [Sequence], models: [Gaussian]) -> Double {
        var total = 0.0
        var frameCount = 0
        for sequence in sequences {
            if Task.isCancelled { return .infinity }
            total += score(emissions(sequence.frames, models: models), states: sequence.states)
            frameCount += sequence.frames.count
        }
        return frameCount > 0 ? total / Double(frameCount) : .infinity
    }

    /// One emission table per utterance, shared across every candidate pronunciation.
    private static func emissions(_ frames: [[Double]], models: [Gaussian]) -> [[Double]] {
        frames.map { frame in
            models.map { model in
                var cost = 0.0
                for d in frame.indices {
                    let delta = frame[d] - model.mean[d]
                    cost += 0.5 * (log(model.variance[d] / varianceFloor) + delta * delta / model.variance[d])
                }
                return cost.isFinite ? cost : .infinity
            }
        }
    }

    /// Complete left-to-right Viterbi path, beginning at position zero and visiting every
    /// transcript position at least once. Ties prefer a self-loop deterministically.
    private static func path(_ emissions: [[Double]], states: [Int]) -> [Int]? {
        guard !states.isEmpty, emissions.count >= states.count else { return nil }
        let n = states.count
        var previous = [Double](repeating: .infinity, count: n)
        previous[0] = emissions[0][states[0]]
        var advances = [[Bool]](repeating: [Bool](repeating: false, count: n), count: emissions.count)
        if emissions.count > 1 {
            for t in 1..<emissions.count {
                if Task.isCancelled { return nil }
                var current = [Double](repeating: .infinity, count: n)
                for s in 0...min(t, n - 1) {
                    let stay = previous[s] + loopCost
                    let advance = s > 0 ? previous[s - 1] + advanceCost : .infinity
                    advances[t][s] = advance < stay
                    current[s] = min(stay, advance) + emissions[t][states[s]]
                }
                previous = current
            }
        }
        guard previous[n - 1].isFinite else { return nil }
        var state = n - 1
        var result = [Int](repeating: 0, count: emissions.count)
        for t in stride(from: emissions.count - 1, through: 0, by: -1) {
            result[t] = state
            if advances[t][state] { state -= 1 }
        }
        return state == 0 ? result : nil
    }

    /// Score-only Viterbi uses O(pronunciation length) scratch memory.
    private static func score(_ emissions: [[Double]], states: [Int]) -> Double {
        guard !states.isEmpty, emissions.count >= states.count else { return .infinity }
        let n = states.count
        var previous = [Double](repeating: .infinity, count: n)
        previous[0] = emissions[0][states[0]]
        if emissions.count > 1 {
            for t in 1..<emissions.count {
                if t % 32 == 0, Task.isCancelled { return .infinity }
                var current = [Double](repeating: .infinity, count: n)
                for s in 0...min(t, n - 1) {
                    let advance = s > 0 ? previous[s - 1] + advanceCost : .infinity
                    current[s] = min(previous[s] + loopCost, advance) + emissions[t][states[s]]
                }
                previous = current
            }
        }
        return previous[n - 1] + advanceCost // complete exit from the last position
    }
}
