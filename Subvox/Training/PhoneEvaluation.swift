import Foundation

/// Per-phoneme test: can each sound be recognised from the motion around it?
///
/// Every take is a mouthed word. Its phones are spread over the take's active motion (vowels
/// weighted double), and each phone becomes an instance: the motion in a window around its
/// centre. Takes are split into four folds; each fold in turn is the held-out 25% and its
/// instances are classified by nearest neighbour (DTW) against instances from the other 75%.
/// Every instance is tested exactly once, always against takes it wasn't part of.
///
/// A phone lasts only a few samples (~4 at the AirPods' 50 Hz), so the window deliberately includes its
/// neighbours: this asks "is the sound identifiable in context", which is what decoding needs.
struct PhoneEvaluation: Sendable {
    struct Result: Hashable, Sendable {
        var tested = 0
        var correct = 0
        /// The phoneme it was most often mistaken for, if any.
        var confusedWith: String?

        var recall: Double { tested == 0 ? 0 : Double(correct) / Double(tested) }
    }

    var results: [String: Result]
    var accuracy: Double
    /// Accuracy of always guessing the most common phoneme.
    var chance: Double
    /// The same predictions scored only on vowel jaw class (close / mid / open / glides).
    var jawAccuracy: Double
    var jawChance: Double
    var instances: Int
    var takes: Int

    static let folds = 4
    static let halfWindow: TimeInterval = 0.2

    static func run(takes: [Template], lexicon: Lexicon, featureSet: FeatureSet) -> PhoneEvaluation? {
        guard !featureSet.isEmpty else { return nil }
        var instances: [(phone: String, take: Int, frames: [[Double]])] = []
        var usable = 0
        for take in takes {
            guard let phones = lexicon.phones(take.label) else { continue }
            let active = MotionFeatures.trim(take.samples)
            guard let first = active.first?.t, let last = active.last?.t, last > first else { continue }
            usable += 1
            let spans = Alignment.spread(weights: phones.map { Lexicon.vowels.contains($0) ? 2 : 1 }, over: first...last)
            for (phone, span) in zip(phones, spans) {
                let centre = (span.lowerBound + span.upperBound) / 2
                let window = take.samples.filter { abs($0.t - centre) <= halfWindow }
                guard window.count >= 3 else { continue }
                instances.append((phone, usable - 1, MotionFeatures.frames(window, set: featureSet)))
            }
        }
        guard usable >= folds, Set(instances.map(\.phone)).count >= 2 else { return nil }

        // Deterministic fold assignment so re-running on the same data gives the same answer.
        var generator = SplitMix64(seed: 0x5EED)
        let order = Array(0..<usable).shuffled(using: &generator)
        var fold = [Int](repeating: 0, count: usable)
        for (position, take) in order.enumerated() { fold[take] = position % folds }

        var predicted = [String](repeating: "", count: instances.count)
        for f in 0..<folds {
            let train = instances.indices.filter { fold[instances[$0].take] != f }
            let test = instances.indices.filter { fold[instances[$0].take] == f }
            guard !train.isEmpty else { continue }
            let scales = MotionFeatures.channelScales(train.map { instances[$0].frames })
            let reference = train.map { (instances[$0].phone, MotionFeatures.scaled(instances[$0].frames, by: scales)) }
            for i in test {
                let query = MotionFeatures.scaled(instances[i].frames, by: scales)
                predicted[i] = reference.map { ($0.0, DTW.distance(query, $0.1)) }.min { $0.1 < $1.1 }?.0 ?? ""
            }
        }

        var results: [String: Result] = [:]
        var mistakes: [String: [String: Int]] = [:]
        for (instance, guess) in zip(instances, predicted) {
            results[instance.phone, default: Result()].tested += 1
            if guess == instance.phone {
                results[instance.phone]!.correct += 1
            } else {
                mistakes[instance.phone, default: [:]][guess, default: 0] += 1
            }
        }
        for (phone, wrong) in mistakes {
            results[phone]?.confusedWith = wrong.max { $0.value < $1.value }?.key
        }

        let truth = instances.map(\.phone)
        let vowelPairs = zip(truth, predicted).compactMap { t, p in
            Lexicon.jawClass(t).map { ($0, Lexicon.jawClass(p)) }
        }
        return PhoneEvaluation(
            results: results,
            accuracy: share(zip(truth, predicted).filter { $0 == $1 }.count, of: truth.count),
            chance: majorityShare(truth),
            jawAccuracy: share(vowelPairs.filter { $0.0 == $0.1 }.count, of: vowelPairs.count),
            jawChance: majorityShare(vowelPairs.map(\.0)),
            instances: instances.count,
            takes: usable)
    }

    private static func share(_ count: Int, of total: Int) -> Double {
        total == 0 ? 0 : Double(count) / Double(total)
    }

    private static func majorityShare(_ labels: [String]) -> Double {
        let counts = Dictionary(labels.map { ($0, 1) }, uniquingKeysWith: +)
        return share(counts.values.max() ?? 0, of: labels.count)
    }
}

/// Small seedable generator; the standard library's is not reproducible.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
