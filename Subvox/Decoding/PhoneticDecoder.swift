import Foundation

/// Open-vocabulary decoder: recognises words it was never trained on by building them from sounds.
///
/// Every training take is a mouthed word; its phones are spread over the take's active motion
/// (as in `PhoneEvaluation`) and the frames under each phone become an example of that sound.
/// Each phoneme is modelled by its medoid example, the one closest by DTW to all the others.
/// Every vocabulary word then gets a synthetic template: its pronunciation's phone models joined
/// end to end. A mouthed segment is matched by DTW against the templates of similar length.
/// The optional coarse control pools examples by `JawUnits` while keeping this guessed timing.
///
/// Jaw motion can't separate many sounds (p/b/m look alike), so many words share a template.
/// A small bias toward frequent words breaks those ties; sentence context is the real fix.
struct PhoneticDecoder: Sendable {
    let featureSet: FeatureSet
    /// Phonemes without a model (a shared jaw-unit model in coarse mode).
    /// Words containing them can't be built.
    let missingPhones: [String]
    let wordCount: Int
    /// Number of buildable vocabulary words in unit-sequence buckets containing >1 word.
    /// Uses exact phone sequences in legacy mode and jaw6-v1 sequences in coarse mode.
    /// Counts words, not buckets or pairs; repeated units remain separate positions.
    let collisionCount: Int

    private let scales: [Double]
    private let dims: Int
    /// Templates flattened frame after frame for `DTW.distance(_:_:dims:cutoff:)`.
    private let vocabulary: [(word: String, frames: [Double], rarity: Double)]

    init?(takes: [Template], lexicon: Lexicon, vocabulary words: [(word: String, count: Int)],
          featureSet: FeatureSet, coarseUnits: Bool = false) {
        guard !featureSet.isEmpty else { return nil }
        func unit(_ phone: String) -> String? {
            coarseUnits ? JawUnits.unit(for: phone) : phone
        }

        var examples: [String: [[[Double]]]] = [:]
        var allFrames: [[[Double]]] = []
        for take in takes {
            guard let phones = lexicon.phones(take.label) else { continue }
            let active = MotionFeatures.trim(take.samples)
            guard let first = active.first?.t, let last = active.last?.t, last > first else { continue }
            let frames = MotionFeatures.frames(active, set: featureSet)
            allFrames.append(frames)
            let spans = Alignment.spread(weights: phones.map { Lexicon.vowels.contains($0) ? 2 : 1 }, over: first...last)
            for (phone, span) in zip(phones, spans) {
                guard let key = unit(phone) else { continue }
                var slice = active.indices.filter { span.contains(active[$0].t) }
                if slice.isEmpty, let nearest = active.indices.min(by: {
                    abs(active[$0].t - (span.lowerBound + span.upperBound) / 2) < abs(active[$1].t - (span.lowerBound + span.upperBound) / 2)
                }) {
                    slice = [nearest]
                }
                examples[key, default: []].append(slice.map { frames[$0] })
            }
        }
        guard !examples.isEmpty else { return nil }

        let scales = MotionFeatures.channelScales(allFrames)
        let models = examples.mapValues { Self.medoid($0.map { MotionFeatures.scaled($0, by: scales) }) }

        let maxCount = Double(words.first?.count ?? 1)
        var vocabulary: [(String, [Double], Double)] = []
        var sequenceCounts: [[String]: Int] = [:]
        for entry in words {
            guard let phones = lexicon.phones(entry.word) else { continue }
            let units = phones.compactMap(unit)
            guard units.count == phones.count else { continue }
            let parts = units.compactMap { models[$0] }
            guard parts.count == phones.count else { continue }
            vocabulary.append((entry.word, Self.centred(parts.flatMap { $0 }).flatMap { $0 }, -log(Double(entry.count) / maxCount)))
            sequenceCounts[units, default: 0] += 1
        }

        self.featureSet = featureSet
        self.scales = scales
        self.dims = scales.count
        self.vocabulary = vocabulary
        self.wordCount = vocabulary.count
        self.collisionCount = sequenceCounts.values.filter { $0 > 1 }.reduce(0, +)
        self.missingPhones = Phoneme.all.map(\.symbol).filter { phone in
            guard let key = unit(phone) else { return true }
            return models[key] == nil
        }
    }

    /// - Parameter commonBias: distance added per e-fold of rarity relative to the most common word.
    func predict(_ samples: [MotionSample], maxDistance: Double, minConfidence: Double,
                 commonBias: Double, limit: Int = 8) -> Prediction {
        guard limit > 0 else { return Prediction(ranked: [], confidence: 0, accepted: false) }
        let query = MotionFeatures.scaled(MotionFeatures.frames(MotionFeatures.trim(samples), set: featureSet), by: scales)
            .flatMap { $0 }
        guard !query.isEmpty else { return Prediction(ranked: [], confidence: 0, accepted: false) }
        // Keep the best `limit` so far; a template that can't beat the worst of them is abandoned mid-DTW.
        var ranked: [RankedLabel] = []
        for entry in vocabulary {
            // Templates far from the segment's length can't be the word; skip the DTW.
            let ratio = Double(entry.frames.count) / Double(query.count)
            guard ratio > 0.4, ratio < 2.5 else { continue }
            let bias = commonBias * entry.rarity
            let worst = ranked.count == limit ? ranked[limit - 1].distance : .infinity
            let score = DTW.distance(query, entry.frames, dims: dims, cutoff: worst - bias) + bias
            guard score < worst else { continue }
            let at = ranked.firstIndex { $0.distance > score } ?? ranked.count
            ranked.insert(RankedLabel(label: entry.word, distance: score), at: at)
            if ranked.count > limit { ranked.removeLast() }
        }
        let confidence = margin(ranked.map(\.distance))
        let accepted = (ranked.first?.distance ?? .infinity) <= maxDistance && confidence >= minConfidence
        return Prediction(ranked: ranked, confidence: confidence, accepted: accepted)
    }

    /// 0…1 relative gap between the best and second-best candidate.
    private func margin(_ sortedDistances: [Double]) -> Double {
        guard sortedDistances.count >= 2, sortedDistances[1] > 0 else { return 0 }
        return max(0, 1 - sortedDistances[0] / sortedDistances[1])
    }

    private static func medoid(_ examples: [[[Double]]]) -> [[Double]] {
        guard examples.count > 2 else { return examples[0] }
        let totals = examples.map { a in examples.reduce(0) { $0 + DTW.distance(a, $1) } }
        return examples[totals.indices.min { totals[$0] < totals[$1] }!]
    }

    /// Phone models come from different takes, each centred on its own take; re-centre the
    /// joined template the same way a live segment is.
    private static func centred(_ frames: [[Double]]) -> [[Double]] {
        guard let dims = frames.first?.count, !frames.isEmpty else { return frames }
        let means = (0..<dims).map { d in frames.reduce(0) { $0 + $1[d] } / Double(frames.count) }
        return frames.map { frame in zip(frame, means).map { $0 - $1 } }
    }
}
