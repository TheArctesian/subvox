import Foundation

/// Investigative development-data audit. Never fits or scores the reserved final split.
@main struct SignalAudit {
    struct Trial: Codable {
        var actual: String
        var predicted: String
        var nearestTake: UUID
        var distance: Double
    }
    struct Result: Codable {
        var name: String
        var correct: Int
        var total: Int
        var accuracy: Double
        var balancedAccuracy: Double
        var permutationP: Double?
        var familywiseP: Double?
        var trials: [Trial]
    }
    struct Quality: Codable {
        var id: UUID; var word: String; var sessionID: UUID?
        var duration: Double; var activeDuration: Double; var trimFraction: Double
        var sampleRate: Double?; var peakEnergy: Double; var preRollContrast: Double?
        var issues: [String]
    }
    struct Report: Codable {
        var developmentTakes: Int
        var eligibleTakes: Int
        var repeatedWords: [String: Int]
        var repeatedTakeCount: Int
        var knownSessionCount: Int
        var majorityBaseline: Double
        var permutations: Int
        var identityRetrieval: Double?
        var withinClassDistance: Double?
        var betweenClassDistance: Double?
        var results: [Result]
        var crossSessionResults: [Result]
        var compositionalResults: [Result]
        var cueOnlyResults: [Result]
        var quality: [Quality]
        var limitations: [String]
    }
    struct Variant {
        var name: String
        var features: FeatureSet
        var trim: Bool
        var articulationOnly = false
        var preRollOnly = false
    }

    static let permutations = 5000
    static let variants = [
        Variant(name: "acceleration / whole", features: .acceleration, trim: false),
        Variant(name: "rotation / whole", features: .rotation, trim: false),
        Variant(name: "attitude / whole", features: .attitude, trim: false),
        Variant(name: "all / whole", features: .all, trim: false),
        Variant(name: "acceleration / trimmed", features: .acceleration, trim: true),
        Variant(name: "rotation / trimmed", features: .rotation, trim: true),
        Variant(name: "attitude / trimmed", features: .attitude, trim: true),
        Variant(name: "all / trimmed", features: .all, trim: true),
        Variant(name: "all / after 0.3 s pre-roll", features: .all, trim: false, articulationOnly: true),
        Variant(name: "all / pre-roll only (confound control)", features: .all, trim: false, preRollOnly: true)
    ]

    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 3 else { print("Usage: audit_signal.sh DATASET_JSON OUTPUT_JSON"); return }
        let data = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
        guard let experiment = data.experiment, experiment.frozenAt == nil else {
            throw NSError(domain: "SignalAudit", code: 1, userInfo: [NSLocalizedDescriptionKey: "Requires an unfrozen experiment; uses development data only."])
        }
        let development = data.templates.filter {
            guard let role = experiment.role(of: $0) else { return false }
            return role != .finalTest
        }
        let eligible = development.filter { $0.isAuditableSpeech && experiment.matchesCondition($0) }
        let diagnostic = eligible.filter { $0.purpose == .signalMouthed }
        let repeatSource = diagnostic.isEmpty ? eligible : diagnostic
        let counts = Dictionary(grouping: repeatSource, by: \.label).mapValues(\.count).filter { $0.value >= 2 }
        let repeated = repeatSource.filter { counts[$0.label] != nil }
        let truth = repeated.map(\.label)
        let majority = truth.isEmpty ? 0 : Double(counts.values.max() ?? 0) / Double(truth.count)
        var results: [Result] = []
        var nearestByVariant: [[Int]] = []
        var identity: Double?
        var within: Double?
        var between: Double?
        for variant in variants where !repeated.isEmpty {
            let raw = repeated.map { features($0, variant) }
            var nearest: [Int] = []
            var distances: [Double] = []
            var own: [Double] = [], other: [Double] = []
            var identities = 0
            for i in repeated.indices {
                let train = repeated.indices.filter { $0 != i }
                let scales = MotionFeatures.channelScales(train.map { raw[$0] })
                let query = MotionFeatures.scaled(raw[i], by: scales)
                var best = (index: -1, distance: Double.infinity)
                for j in train {
                    let distance = DTW.distance(query, MotionFeatures.scaled(raw[j], by: scales))
                    if distance < best.distance { best = (j, distance) }
                    if truth[i] == truth[j] { own.append(distance) } else { other.append(distance) }
                }
                nearest.append(best.index)
                distances.append(best.distance)
                // This identity check tests numerical plumbing, not generalization.
                if DTW.distance(query, query) <= best.distance { identities += 1 }
            }
            guard nearest.allSatisfy({ $0 >= 0 }), distances.allSatisfy(\.isFinite) else { continue }
            nearestByVariant.append(nearest)
            results.append(result(variant.name, takes: repeated, nearest: nearest, distances: distances))
            if variant.name == "all / trimmed" {
                identity = Double(identities) / Double(repeated.count)
                within = mean(own)
                between = mean(other)
            }
        }
        // Familywise correction uses the maximum raw accuracy across all tested motion variants
        // on each shared label permutation. It accounts for selecting a flattering feature set.
        var generator = SplitMix64(seed: 0x51A1)
        var exceeds = [Int](repeating: 1, count: results.count)
        var corrected = exceeds
        for _ in 0..<permutations where !results.isEmpty {
            let shuffled = truth.shuffled(using: &generator)
            let scores = nearestByVariant.map { nearest in
                Double(nearest.indices.filter { shuffled[$0] == shuffled[nearest[$0]] }.count) / Double(truth.count)
            }
            let maximum = scores.max() ?? 0
            for i in results.indices {
                if scores[i] >= results[i].accuracy { exceeds[i] += 1 }
                if maximum >= results[i].accuracy { corrected[i] += 1 }
            }
        }
        for i in results.indices {
            results[i].permutationP = Double(exceeds[i]) / Double(permutations + 1)
            results[i].familywiseP = Double(corrected[i]) / Double(permutations + 1)
        }
        if !repeated.isEmpty {
            let temporal = nearestBy(repeated) { a, b in abs(a.createdAt.timeIntervalSince(b.createdAt)) }
            results.append(result("recording time only (confound control)", takes: repeated,
                                  nearest: temporal, distances: temporal.enumerated().map { abs(repeated[$0.offset].createdAt.timeIntervalSince(repeated[$0.element].createdAt)) }))
            let duration = nearestBy(repeated) { a, b in abs(a.duration - b.duration) }
            results.append(result("duration only (confound control)", takes: repeated, nearest: duration,
                                  distances: duration.enumerated().map { abs(repeated[$0.offset].duration - repeated[$0.element].duration) }))
        }
        // Session test uses only explicit session IDs. Unknown historical sessions are never invented.
        var cross: [Result] = []
        let known = eligible.filter { $0.sessionID != nil }
        let sessionCounts = Dictionary(grouping: known, by: \.label).mapValues { Set($0.compactMap(\.sessionID)).count }
        let multiSession = known.filter { sessionCounts[$0.label, default: 0] >= 2 }
        if !multiSession.isEmpty {
            for variant in variants {
                let raw = multiSession.map { features($0, variant) }
                var nearest: [Int] = [], distances: [Double] = []
                for i in multiSession.indices {
                    let train = multiSession.indices.filter { multiSession[$0].sessionID != multiSession[i].sessionID }
                    let scales = MotionFeatures.channelScales(train.map { raw[$0] })
                    let query = MotionFeatures.scaled(raw[i], by: scales)
                    let winner = train.map { ($0, DTW.distance(query, MotionFeatures.scaled(raw[$0], by: scales))) }
                        .min { $0.1 < $1.1 }!
                    nearest.append(winner.0); distances.append(winner.1)
                }
                cross.append(result(variant.name, takes: multiSession, nearest: nearest, distances: distances))
            }
        }
        var compositional: [Result] = []
        if !repeated.isEmpty {
            let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let lexicon = Lexicon(tsv: try String(contentsOf: root.appending(path: "Corpus/lexicon.tsv"), encoding: .utf8)
                                  + String(contentsOf: root.appending(path: "Corpus/words.tsv"), encoding: .utf8))
            let words = counts.keys.sorted().map { (word: $0, count: 1) }
            for method in LearningMethod.allCases {
                var trials: [Trial] = []
                for i in repeated.indices {
                    var settings = DecoderSettings()
                    settings.method = method
                    settings.commonBias = 0
                    let training = repeated.indices.filter { $0 != i }.map { repeated[$0] }
                    guard let decoder = MotionWordDecoder(takes: training, lexicon: lexicon, vocabulary: words, settings: settings) else { continue }
                    let p = decoder.predict(repeated[i].samples)
                    trials.append(Trial(actual: repeated[i].label, predicted: p.best?.label ?? "?",
                                        nearestTake: repeated[i].id, distance: p.best?.distance ?? 1e30))
                }
                let groups = Dictionary(grouping: trials, by: \.actual)
                let correct = trials.filter { $0.actual == $0.predicted }.count
                let balanced = groups.values.map { Double($0.filter { $0.actual == $0.predicted }.count) / Double($0.count) }
                compositional.append(Result(name: method.rawValue + " / known repeated-word vocabulary LOO",
                    correct: correct, total: trials.count, accuracy: Double(correct)/Double(max(1, trials.count)),
                    balancedAccuracy: mean(balanced) ?? 0, trials: trials))
            }
        }
        let controls = development.filter { $0.purpose == .cueOnly && $0.reviewDecision == .accepted && !$0.quality.invalid }
        var controlResults: [Result] = []
        let controlCounts = Dictionary(grouping: controls, by: \.label).mapValues(\.count)
        let repeatedControls = controls.filter { controlCounts[$0.label, default: 0] >= 2 }
        if !repeatedControls.isEmpty {
            for variant in variants {
                let raw = repeatedControls.map { features($0, variant) }
                var nearest: [Int] = [], distances: [Double] = []
                for i in repeatedControls.indices {
                    let train = repeatedControls.indices.filter { $0 != i }
                    let scales = MotionFeatures.channelScales(train.map { raw[$0] })
                    let query = MotionFeatures.scaled(raw[i], by: scales)
                    let winner = train.map { ($0, DTW.distance(query, MotionFeatures.scaled(raw[$0], by: scales))) }.min { $0.1 < $1.1 }!
                    nearest.append(winner.0); distances.append(winner.1)
                }
                controlResults.append(result(variant.name, takes: repeatedControls, nearest: nearest, distances: distances))
            }
        }
        let quality = eligible.map { take -> Quality in
            let active = MotionFeatures.trim(take.samples)
            let first = take.samples.first!.t
            let before = take.samples.filter { $0.t <= (take.cueStartUptime ?? first + 0.2) }.map(MotionFeatures.energy)
            let after = take.samples.filter { $0.t >= (take.cueStartUptime ?? first + 0.3) }.map(MotionFeatures.energy)
            let pre = mean(before) ?? 0
            return Quality(id: take.id, word: take.label, sessionID: take.sessionID, duration: take.duration,
                           activeDuration: active.last!.t - active.first!.t,
                           trimFraction: 1 - Double(active.count) / Double(take.samples.count),
                           sampleRate: take.quality.sampleRate, peakEnergy: take.samples.map(MotionFeatures.energy).max() ?? 0,
                           preRollContrast: pre > 0 ? (mean(after) ?? 0) / pre : nil, issues: take.quality.issues)
        }
        let report = Report(developmentTakes: development.count, eligibleTakes: eligible.count,
            repeatedWords: counts, repeatedTakeCount: repeated.count, knownSessionCount: Set(known.compactMap(\.sessionID)).count,
            majorityBaseline: majority, permutations: permutations, identityRetrieval: identity,
            withinClassDistance: within, betweenClassDistance: between, results: results, crossSessionResults: cross,
            compositionalResults: compositional, cueOnlyResults: controlResults,
            quality: quality, limitations: [
                "Development-only exploratory analysis; existing validation takes may be fitted in these diagnostics, never the final split. This is not the production validation score.",
                "Most historical words were recorded in blocks; permutation p-values cannot separate articulation from recording-order/posture confounds.",
                "Pre-roll contrast is a rough activity comparison, not measured jaw SNR or a verified resting baseline.",
                "When dedicated signal-check data exists, the primary repetition audit uses only those silently mouthed probes. Legacy and voiced takes are not mixed into that primary test.",
                "Whole-word separability does not establish phoneme decoding or silent speech information sufficiency.",
                "Cross-session results require repeated labels in at least two explicitly identified sessions. Empty results mean insufficient recordings, not failure."
            ])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath: args[2]), options: .atomic)
        print("\(eligible.count) eligible development takes. Repeated words: \(counts). Majority baseline \(majority).")
        for r in results { print(String(format: "%@: %d/%d (%.1f%%); balanced %.1f%%; permutation p %.4f; corrected p %@", r.name, r.correct, r.total, 100*r.accuracy, 100*r.balancedAccuracy, r.permutationP ?? -1, r.familywiseP.map { String(format: "%.4f", $0) } ?? "n/a")) }
        for r in compositional { print("COMPOSITION \(r.name): \(r.correct)/\(r.total)") }
        for r in controlResults { print("CUE-ONLY \(r.name): \(r.correct)/\(r.total)") }
        print("Known sessions \(report.knownSessionCount); cross-session variants \(cross.count); identity sanity \(identity ?? -1); within/between DTW \(within ?? -1) / \(between ?? -1)")
    }

    static func features(_ take: Template, _ variant: Variant) -> [[Double]] {
        let samples: [MotionSample]
        if variant.preRollOnly, let start = take.samples.first?.t {
            samples = take.samples.filter { $0.t <= (take.cueStartUptime ?? start + 0.2) }
        } else if variant.articulationOnly, let start = take.samples.first?.t {
            samples = take.samples.filter { $0.t >= (take.cueStartUptime ?? start + 0.3) }
        } else { samples = variant.trim ? MotionFeatures.trim(take.samples) : take.samples }
        return MotionFeatures.frames(samples, set: variant.features)
    }
    static func mean(_ values: [Double]) -> Double? { values.isEmpty ? nil : values.reduce(0,+) / Double(values.count) }
    static func nearestBy(_ takes: [Template], distance: (Template, Template) -> Double) -> [Int] {
        takes.indices.map { i in takes.indices.filter { $0 != i }.min { distance(takes[i], takes[$0]) < distance(takes[i], takes[$1]) }! }
    }
    static func result(_ name: String, takes: [Template], nearest: [Int], distances: [Double]) -> Result {
        let hits = takes.indices.filter { takes[$0].label == takes[nearest[$0]].label }.count
        let groups = Dictionary(grouping: takes.indices, by: { takes[$0].label })
        let balanced = groups.values.map { group in
            Double(group.filter { takes[$0].label == takes[nearest[$0]].label }.count) / Double(group.count)
        }.reduce(0,+) / Double(groups.count)
        return Result(name: name, correct: hits, total: takes.count, accuracy: Double(hits)/Double(takes.count),
                      balancedAccuracy: balanced, permutationP: nil, trials: takes.indices.map {
            Trial(actual: takes[$0].label, predicted: takes[nearest[$0]].label, nearestTake: takes[nearest[$0]].id, distance: distances[$0])
        })
    }
}
