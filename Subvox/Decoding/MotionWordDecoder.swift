import Foundation
import CryptoKit

enum LearningMethod: String, Codable, CaseIterable, Identifiable, Sendable {
    case legacyMedoids, coarseMedoids, jawHMMInitial, jawHMM
    var id: String { rawValue }
    var title: String {
        switch self {
        case .legacyMedoids: "Original phone templates"
        case .coarseMedoids: "Coarse jaw templates (control)"
        case .jawHMMInitial: "Jaw sequence, initial alignment (control)"
        case .jawHMM: "Jaw sequence, learned alignment (experimental)"
        }
    }
    var isSequence: Bool { self == .jawHMM || self == .jawHMMInitial }
}

enum VocabularyScope: String, Codable, CaseIterable, Identifiable, Sendable {
    case common100, common1000, full
    var id: String { rawValue }
    var title: String {
        switch self { case .common100: "100 common words"; case .common1000: "1,000 common words"; case .full: "Full vocabulary" }
    }
    /// The candidate list depends only on a fixed frequency ordering, never on evaluation truths.
    func select(_ vocabulary: [(word: String, count: Int)]) -> [(word: String, count: Int)] {
        switch self {
        case .common100: Array(vocabulary.prefix(100))
        case .common1000: Array(vocabulary.prefix(1000))
        case .full: vocabulary
        }
    }
}

struct DecoderManifest: Codable, Equatable, Sendable {
    var implementation = "motion-learning-v1"
    var preprocessing = "trim-and-centred-features-v1"
    var units: String
    var trainingHash: String
    var vocabularyHash: String
    var settingsHash: String
}

/// Single entry point for live decoding, validation, ablations and the frozen final test.
struct MotionWordDecoder: Sendable {
    private var templates: PhoneticDecoder?
    private var sequence: JawSequenceDecoder?
    let settings: DecoderSettings
    let candidateWords: Set<String>
    let manifest: DecoderManifest
    var wordCount: Int { templates?.wordCount ?? sequence?.wordCount ?? 0 }
    var missingPhones: [String] { templates?.missingPhones ?? sequence?.missingPhones ?? [] }
    var collisionCount: Int { templates?.collisionCount ?? sequence?.collisionCount ?? 0 }
    var trainingSummary: String {
        sequence?.trainingSummary ?? (settings.method == .legacyMedoids ? "Original evenly allocated phone medoids." : "Six hypothesised jaw units; timing is still evenly allocated.")
    }

    init?(takes: [Template], lexicon: Lexicon, vocabulary: [(word: String, count: Int)], settings: DecoderSettings) {
        self.settings = settings
        let selected = settings.vocabularyScope.select(vocabulary)
        candidateWords = Set(selected.map(\.word))
        manifest = Self.manifest(takes: takes, lexicon: lexicon, vocabulary: vocabulary, settings: settings)
        if settings.method.isSequence {
            sequence = JawSequenceDecoder(takes: takes, lexicon: lexicon, vocabulary: selected,
                                          featureSet: settings.features, iterations: settings.method == .jawHMM ? 6 : 0)
            templates = nil
            guard sequence != nil else { return nil }
        } else {
            templates = PhoneticDecoder(takes: takes, lexicon: lexicon, vocabulary: selected,
                                        featureSet: settings.features, coarseUnits: settings.method == .coarseMedoids)
            sequence = nil
            guard templates != nil else { return nil }
        }
    }

    func predict(_ samples: [MotionSample], limit: Int = 8) -> Prediction {
        if let templates {
            return templates.predict(samples, maxDistance: settings.maxDistance, minConfidence: settings.minConfidence,
                                     commonBias: settings.commonBias, limit: limit)
        }
        return sequence?.predict(samples, maxScore: settings.hmmMaxScore, minScoreGap: settings.hmmMinGap,
                                 commonBias: settings.commonBias, limit: limit)
            ?? Prediction(ranked: [], confidence: 0, accepted: false)
    }

    static func manifest(takes: [Template], lexicon: Lexicon, vocabulary: [(word: String, count: Int)],
                         settings: DecoderSettings) -> DecoderManifest {
        struct Entry: Encodable { var word: String; var count: Int; var phones: [String]? }
        struct TrainingMaterial: Encodable { var takes: [Template]; var pronunciations: [Entry] }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        func hash<T: Encodable>(_ value: T) -> String {
            guard let data = try? encoder.encode(value) else { return "encoding-failed" }
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        return DecoderManifest(units: settings.method == .legacyMedoids ? "arpabet39-v1" : JawUnits.version,
            trainingHash: hash(TrainingMaterial(takes: takes, pronunciations: takes.map {
                Entry(word: $0.label, count: 1, phones: lexicon.phones($0.label))
            })),
            vocabularyHash: hash(settings.vocabularyScope.select(vocabulary).map {
                Entry(word: $0.word, count: $0.count, phones: lexicon.phones($0.word))
            }), settingsHash: hash(settings))
    }
}
