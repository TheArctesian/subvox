import Foundation
import CryptoKit

enum SentenceReference {
    enum Failure: LocalizedError {
        case ambiguousTiming, invalidClock, invalidTiming
        var errorDescription: String? {
            switch self {
            case .ambiguousTiming: "Whisper assigned one interval to multiple words. Word boundaries are ambiguous; no duplicate timings were invented. Re-record with sentence prompts or review the raw audio."
            case .invalidClock: "The recording clock is invalid. Check the raw recording metadata."
            case .invalidTiming: "A word has no usable duration. Retry audio alignment; the recording clock was not changed."
            }
        }
    }
    /// Whisper times already include the audio crop offset; add only the session audio clock.
    static func anchors(_ transcript: LocalWhisperTranscript, sessionStart: Double,
                        audioOffset: Double) throws -> [MotionWordAnchor] {
        guard sessionStart.isFinite, audioOffset.isFinite else { throw Failure.invalidClock }
        return try transcript.words.compactMap { token in
            let words = Lexicon.words(in: token.text)
            guard words.count <= 1 else { throw Failure.ambiguousTiming }
            guard let word = words.first else { return nil }
            guard token.start.isFinite, token.end.isFinite, token.start >= 0, token.end > token.start else {
                throw Failure.invalidTiming
            }
            return MotionWordAnchor(word: word, start: token.start + sessionStart + audioOffset,
                                    end: token.end + sessionStart + audioOffset)
        }
    }
}

/// `community` and `calibration` prompts come from the public-domain reading corpus. Every
/// existing scheduler filters on `.sentence`, so they never enter the personal split or its final-test quota.
enum SentencePromptKind: String, Codable, Sendable {
    case sentence, wordRepair, community, calibration
    /// Train tabs: public-domain speeches and poems, shadowing drills, minimal pairs, the user's own text.
    case speech, poetry, drill, minimalPair, freeText, sound
    var isLiterature: Bool { self == .speech || self == .poetry }
    var isPractice: Bool { isLiterature || [.drill, .minimalPair, .freeText, .sound].contains(self) }
}

struct SentencePrompt: Codable, Identifiable, Hashable, Sendable {
    var text: String
    var kind: SentencePromptKind = .sentence
    /// BCP-47 base language ("en", "fr", "es", "zh"); nil means English (older prompts).
    var language: String? = nil
    var englishTranslation: String? = nil
    var pinyin: String? = nil
    /// Punctuation/case variants have the same persistent identity.
    var id: String { Self.identity(text) }
    static func identity(_ text: String) -> String {
        let normalized = Lexicon.words(in: text).joined(separator: " ")
        return SHA256.hash(data: Data(normalized.utf8)).map { String(format: "%02x", $0) }.joined()
    }
    static var bundled: [SentencePrompt] {
        guard let url = Bundle.main.url(forResource: "paired-sentences", withExtension: "txt", subdirectory: "Corpus"),
              let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return content.split(separator: "\n").map { SentencePrompt(text: String($0)) }
    }
    static var diphoneBank:[SentencePrompt] {bank("diphones")}
    /// Unique slot-grammar sentences, each recorded once, spanning jaw height and labial onsets.
    static var gridBank:[SentencePrompt] {bank("grid-sentences")}
    /// "Say ___ again." minimal pairs differing in one observable feature (Corpus/frames.tsv).
    static var frameBank:[SentencePrompt] {bank("frames")}
    static func bank(_ name:String)->[SentencePrompt] {
        guard let url=Bundle.main.url(forResource:name,withExtension:"txt",subdirectory:"Corpus"),
              let text=try? String(contentsOf:url,encoding:.utf8) else {return []}
        return text.split(separator:"\n").map {Self(text:String($0))}
    }
}

struct SentenceRecording: Codable, Identifiable, Sendable {
    var id = UUID()
    var condition: SpeechCondition
    var samples: [MotionSample]
    var cueStart: Double
    var cueEnd: Double
    var wordFocusGuide:WordFocusGuide?
    var cueSamples: [MotionSample] { samples.filter { $0.t >= cueStart && $0.t <= cueEnd } }
    var quality: TakeQuality { TakeQuality.inspect(cueSamples, maximumDuration: 48) }
}

enum WordTimingSource: String, Codable, Sendable {
    case forcedAlignment, whisperAttention
}

struct SentencePair: Codable, Identifiable, Sendable {
    var id = UUID()
    var visitID: UUID
    var promptIDs: [String]
    var text: String
    var role: DataRole
    var createdAt = Date()
    var spoken: SentenceRecording?
    var silent: SentenceRecording?
    var directory: String?
    var sessionStartUptime: Double?
    var audioOffset: Double?
    var recordingSessionID: UUID?
    var needsBackgroundCheck: Bool?
    var transcript: LocalWhisperTranscript?
    var transcriptMatchRate: Double?
    var timedWordCoverage: Double?
    var wordTimingSource: WordTimingSource?
    var wordTimingModelRevision: String?
    var alignment: MotionTransferResult?
    var review: ReviewDecision = .pending
    var error: String?
    // Optional sidecars preserve old recordings/word alignment. A word interval is never
    // split into measured phoneme boundaries. Missing aligner/confidence stays explicit.
    var pronunciationProfile: PronunciationProfile?
    var spokenPhones: PhoneAlignment?
    var mouthedPhones: PhoneAlignment?
    var phoneAlignmentNote: String?
    var cameraReadings: [CameraPoseReading]?
    /// Practice tabs: the take's language, its mode, and the audio phone check (expected vs produced IPA).
    var language: String?
    var practiceMode: SentencePromptKind?
    var literaturePieceID: String?
    var phoneCheck: PhoneCheckResult?
    var complete: Bool { spoken != nil && silent != nil }
    var canAccept: Bool {
        guard let spoken, let silent, error == nil, alignment?.words.isEmpty == false else { return false }
        // A cached alignment cannot make a left/right source switch safe to learn from.
        return [spoken, silent].allSatisfy {
            Set($0.cueSamples.map { sample in
                let side = sample.sensorLocation ?? -1
                return side == 0 ? -1 : side
            }).count == 1
        }
    }

    /// Pending is automatic mode. Accepted/Excluded are persistent user overrides;
    /// automatic decisions never overwrite them or approve phone-level annotations.
    var hasReliableTranscript: Bool {
        if review == .accepted { return true }
        let rate = transcriptMatchRate ?? 0
        if wordTimingSource == .forcedAlignment {
            guard error == nil, (0.8...1).contains(timedWordCoverage ?? 0) else { return false }
            return practiceMode != .freeText || (0.85...1).contains(rate)
        }
        if practiceMode == .freeText { return (0.85...1).contains(rate) }
        return ((wordTimingSource == nil ? 0.85 : 0.8)...1).contains(rate)
    }
    var usesAlignedWords: Bool {
        role != .finalTest && review != .excluded && hasReliableTranscript && canAccept
    }
    var canIncludeManually: Bool {
        role != .finalTest && (canAccept || (recordingSessionID != nil && [spoken, silent].contains {
            $0.map { SentenceMotionEligibility.inspect($0).reason == nil } == true
        }))
    }
    var automaticUse: SentencePairUse { SentencePairUse(self) }
}

/// Mirrors the timestamp/source checks of the neural dataset resampler. These are
/// eligibility checks, not recognition accuracy; pronunciation/CTC checks run at preparation.
struct SentenceMotionEligibility: Sendable {
    static let minimumRate = 20.0
    var reason: String?
    static func inspect(_ recording: SentenceRecording) -> Self {
        let samples = recording.samples
        guard samples.count >= 12 else { return .init(reason: "Too few motion samples") }
        guard samples.allSatisfy({ [$0.t, $0.ax, $0.ay, $0.az, $0.rx, $0.ry, $0.rz, $0.gx, $0.gy, $0.gz].allSatisfy(\.isFinite) }) else {
            return .init(reason: "Invalid sensor values")
        }
        let intervals = zip(samples, samples.dropFirst()).map { $1.t - $0.t }
        guard intervals.allSatisfy({ $0 > 0 }) else { return .init(reason: "Sensor timestamps are not increasing") }
        let start = recording.cueStart, end = recording.cueEnd
        guard start.isFinite, end.isFinite, (0.5...48).contains(end - start) else { return .init(reason: "Invalid cue duration") }
        let lo = max(start, samples[0].t), hi = min(end, samples[samples.count - 1].t)
        guard lo - start <= 0.1, end - hi <= 0.1, hi - lo >= 0.9 * (end - start) else {
            return .init(reason: "Motion does not cover the cue")
        }
        let context = samples.filter { $0.t >= lo - 0.1 && $0.t <= hi + 0.1 }
        let sides = Set(context.map { [nil, 0, -1].contains($0.sensorLocation) ? -1 : $0.sensorLocation! })
        guard sides.count == 1 else { return .init(reason: "Sensor source switched during the take") }
        let gaps = zip(context, context.dropFirst()).map { $1.t - $0.t }.sorted()
        guard !gaps.isEmpty else { return .init(reason: "Too few motion samples") }
        if let gap = gaps.last, gap > 0.12 { return .init(reason: String(format: "Sensor gap of %.3f s (maximum 0.120 s)", gap)) }
        let middle = gaps.count / 2
        let median = gaps.count.isMultiple(of: 2) ? (gaps[middle - 1] + gaps[middle]) / 2 : gaps[middle]
        // Matches MIN_MOTION_RATE in ml/subvox_ml/data.py. Slower AirPods are interpolated onto the
        // 50 Hz model grid there, and the native rate is kept with the take.
        guard 1 / median >= Self.minimumRate else { return .init(reason: "Motion rate below \(Int(Self.minimumRate)) Hz") }
        let frames = Int(floor((hi - lo) * 50)) + 1
        guard frames >= 25, context.allSatisfy({ sqrt($0.gx * $0.gx + $0.gy * $0.gy + $0.gz * $0.gz) >= 0.1 }) else {
            return .init(reason: "Short sequence or invalid gravity vector")
        }
        return .init(reason: nil)
    }
}

struct SentencePairUse: Sendable {
    enum State: String, Sendable {
        case included = "Included", warnings = "Included with warnings", attention = "Needs attention"
        case excluded = "Excluded", reserved = "Reserved final test"
    }
    struct Take: Sendable {
        var name: String
        var included: Bool
        var reason: String?
    }
    var state: State
    var wordModel: Bool
    var manualInclusionAllowed: Bool
    var takes: [Take]
    var warnings: [String]
    var needsAttention: Bool { state == .attention }

    init(_ pair: SentencePair) {
        wordModel = pair.usesAlignedWords
        manualInclusionAllowed = pair.role != .finalTest && pair.canAccept
        warnings = []
        takes = []
        for (name, recording) in [("Spoken", pair.spoken), ("Mouthed", pair.silent)] {
            let motion = recording.map { SentenceMotionEligibility.inspect($0) }
            if pair.role != .finalTest, pair.recordingSessionID != nil, let motion, motion.reason == nil {
                manualInclusionAllowed = true
            }
            let reason: String?
            if pair.role == .finalTest { reason = "Reserved for the final test" }
            else if pair.review == .excluded { reason = "Excluded by you" }
            else if recording == nil { reason = "Not recorded" }
            else if !pair.hasReliableTranscript { reason = "Transcript needs confirmation" }
            else if pair.recordingSessionID == nil { reason = "Recording session metadata is missing" }
            else { reason = motion?.reason }
            takes.append(.init(name: name, included: reason == nil, reason: reason))
            if let recording { warnings += recording.quality.issues.map { name + ": " + $0 } }
        }
        if let error = pair.error { warnings.append("Word alignment unavailable: " + error) }
        warnings += pair.alignment?.warnings.filter { !$0.hasPrefix("Transferred word times") } ?? []
        if pair.role == .finalTest { state = .reserved }
        else if pair.review == .excluded { state = .excluded }
        else if wordModel || takes.contains(where: \.included) {
            state = warnings.isEmpty && takes.allSatisfy(\.included) && wordModel ? .included : .warnings
        } else { state = .attention }
    }
}

extension SentencePair {
    /// Only explicit, reviewed phone annotations can supply time-local supervision.
    /// Paragraphs stay atomic and are excluded from this first single-sentence pilot.
    func representationTakes(profile:PronunciationProfile) throws -> [RepresentationTake] {
        guard role == .training,review != .excluded,hasReliableTranscript,promptIDs.count==1,
              promptIDs[0]==SentencePrompt.identity(text),let recordingSessionID,
              let pronunciationProfile,try pronunciationProfile.fingerprint()==profile.fingerprint(),
              let cameraReadings,!cameraReadings.isEmpty else {return []}
        return [(spoken,spokenPhones),(silent,mouthedPhones)].compactMap {recording,alignment in
            guard let recording,let alignment,alignment.reviewed,
                  MotionSourceSafety.source(recording) != nil,
                  (try? profile.fingerprint())==alignment.provenance.pronunciationSHA256,
                  (try? alignment.validate(in:recording.cueStart...recording.cueEnd)) != nil else {return nil}
            return .init(id:recording.id,sentenceID:promptIDs[0],text:text,sessionID:recordingSessionID,repetition:0,
                recording:recording,camera:cameraReadings,phoneAlignment:alignment)
        }
    }
}

struct SentenceVisit: Codable, Identifiable, Sendable {
    var id = UUID()
    var startedAt = Date()
    var endedAt: Date?
    var notes: String
    var directory: String?
    var pairIDs: [UUID] = []
    var audioDevice: String?
    var pace: String?
    var presentation: String?
    var motionRate: Double?
    var recordingSessionID: UUID?
    var automaticContributionID: UUID?
    /// Headphone model, product ID and firmware; absent in older visits.
    var motionDevice: MotionDeviceInfo?
}

/// The active personal training experiment, grouped and split by sentence.
/// A logical round can span capture sessions. Each half points to its physical pair,
/// which owns the recording directory, clock, and device provenance.
struct LiteraturePracticeRound: Codable, Identifiable, Sendable {
    var id = UUID()
    let pieceID: String
    let title: String
    let language: String
    let kind: SentencePromptKind
    let groups: [[SentencePrompt]]
    var updatedAt = Date()
    private var silentPairs: [Int: UUID] = [:]
    private var spokenPairs: [Int: UUID] = [:]

    init(pieceID: String, title: String, language: String, kind: SentencePromptKind, groups: [[SentencePrompt]]) {
        self.pieceID = pieceID
        self.title = title
        self.language = language
        self.kind = kind
        self.groups = groups
    }

    mutating func record(_ condition: SpeechCondition, section: Int, pairID: UUID) {
        if condition == .silent { silentPairs[section] = pairID }
        else { spokenPairs[section] = pairID }
        updatedAt = Date()
    }

    func hasTake(_ condition: SpeechCondition, section: Int, pairs: [UUID: SentencePair]) -> Bool {
        guard let id = (condition == .silent ? silentPairs : spokenPairs)[section],
              let pair = pairs[id], pair.practiceMode == kind,
              pair.literaturePieceID == pieceID, pair.language == language,
              groups.indices.contains(section),
              pair.promptIDs == groups[section].map(\.id),
              pair.text == groups[section].map(\.text).joined(separator: " ") else { return false }
        return condition == .silent ? pair.silent != nil : pair.spoken != nil
    }

    func pendingTakes(pairs: [UUID: SentencePair]) -> [(group: Int, condition: SpeechCondition)] {
        [SpeechCondition.silent, .normal].flatMap { condition in
            groups.indices.compactMap { section in
                hasTake(condition, section: section, pairs: pairs) ? nil : (section, condition)
            }
        }
    }
}

enum LiteraturePracticeSelection {
    case new(pieceID: String, title: String)
    case resume(UUID)
}

struct SentenceTrainingData: Codable, Sendable {
    var schemaVersion = 2
    var id = UUID()
    var createdAt = Date()
    var prompts: [SentencePrompt] = []
    var roles: [String: DataRole] = [:]
    var visits: [SentenceVisit] = []
    var pairs: [SentencePair] = []
    var reservedUnlockedAt: Date?
    var validationReport: SentenceValidationReport?
    var decoderSettings: DecoderSettings?
    var literatureRounds: [LiteraturePracticeRound]?

    mutating func add<G: RandomNumberGenerator>(_ incoming: [SentencePrompt], using generator: inout G) {
        guard reservedUnlockedAt == nil else { return }
        let known = Set(prompts.map(\.id))
        let unique = Dictionary(incoming.filter { !known.contains($0.id) && !Lexicon.words(in: $0.text).isEmpty }
            .map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for prompt in unique.values.sorted(by: { $0.id < $1.id }).shuffled(using: &generator) {
            if prompt.kind == .wordRepair {
                roles[prompt.id] = .training
            } else {
                let n = prompts.filter { $0.kind == .sentence }.count + 1
                let final = prompts.filter { $0.kind == .sentence && roles[$0.id] == .finalTest }.count
                let validation = prompts.filter { $0.kind == .sentence && roles[$0.id] == .validation }.count
                let development = n - 1 - final
                roles[prompt.id] = final < (n + 3) / 4 ? .finalTest
                    : validation < (development + 4) / 4 ? .validation : .training
            }
            prompts.append(prompt)
        }
    }

    /// Community prompts are training material for the shared dataset (held-out text is never
    /// assigned), so they join as training without touching the final-test or validation quotas.
    mutating func addCommunity(_ incoming: [SentencePrompt]) {
        guard reservedUnlockedAt == nil else { return }
        var known = Set(prompts.map(\.id))
        for prompt in incoming where (prompt.kind == .community || prompt.kind == .calibration) && !known.contains(prompt.id) {
            roles[prompt.id] = .training
            prompts.append(prompt)
            known.insert(prompt.id)
        }
    }

    /// Practice prompts are always training data: they are never reserved for the final test, and
    /// they only train when the spoken take agrees with the text (see `SentenceTrainingWorkflow.process`).
    mutating func addPractice(_ incoming: [SentencePrompt]) {
        guard reservedUnlockedAt == nil else { return }
        var known = Set(prompts.map(\.id))
        for prompt in incoming where prompt.kind.isPractice && !known.contains(prompt.id) && !Lexicon.words(in: prompt.text).isEmpty {
            roles[prompt.id] = .training
            prompts.append(prompt)
            known.insert(prompt.id)
        }
    }

    /// Prompt IDs with a saved mouthed take.
    var mouthedPromptIDs: Set<String> { Set(pairs.filter { $0.silent != nil }.flatMap(\.promptIDs)) }

    func derivedWords(role: DataRole, lexicon: Lexicon) -> [Template] {
        guard role != .finalTest else { return [] }
        return pairs.filter { $0.role == role && $0.usesAlignedWords }.flatMap { pair -> [Template] in
            guard pair.canAccept, !pair.promptIDs.isEmpty,
                  pair.promptIDs.allSatisfy({ roles[$0] == role }), let silent = pair.silent,
                  let transfer = pair.alignment else { return [] }
            let cueSamples = silent.cueSamples
            return transfer.words.enumerated().compactMap { index, word in
                guard lexicon.phones(word.word) != nil else { return nil }
                guard word.silentStart >= silent.cueStart - 0.04,
                      word.silentEnd <= silent.cueEnd + 0.04,
                      word.silentEnd > word.silentStart else { return nil }
                let samples = cueSamples.filter { $0.t >= word.silentStart && $0.t <= word.silentEnd }
                guard samples.count >= 3, !TakeQuality.inspect(samples).invalid else { return nil }
                return Template(id: Self.wordID(pair: pair.id, index: index), label: word.word, samples: samples, createdAt: pair.createdAt, sessionID: pair.visitID,
                                sourceSentenceID: pair.promptIDs.joined(separator: ","), sourcePairID: pair.id)
            }
        }
    }

    static func wordID(pair: UUID, index: Int) -> UUID {
        let bytes = Array(SHA256.hash(data: Data("\(pair.uuidString):\(index)".utf8)).prefix(16))
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }

    /// Counts dictionary phone occurrences in eligible, aligned TRAINING words only.
    /// This is annotation coverage, never evidence that these phones were recognised.
    func coverage(lexicon: Lexicon) -> [String: Int] {
        var result: [String: Int] = [:]
        for word in derivedWords(role: .training, lexicon: lexicon) {
            for phone in lexicon.phones(word.label) ?? [] { result[phone, default: 0] += 1 }
        }
        return result
    }

    func plannedSentences(limit: Int, passage: Bool, lexicon: Lexicon) -> [[SentencePrompt]] {
        let acquired = Set(pairs.filter(\.complete).flatMap(\.promptIDs))
        let remaining = prompts.filter { $0.kind == .sentence && roles[$0.id] != .finalTest && !acquired.contains($0.id) }
        let training = remaining.filter { roles[$0.id] == .training }.prefix(limit)
        let validation = remaining.filter { roles[$0.id] == .validation }.prefix(max(1, limit / 3))
        let chosen = Array(training) + Array(validation)
        if !passage { return chosen.map { [$0] } }
        // Never mix train/validation/final sentence groups inside a passage.
        return [DataRole.training, .validation].flatMap { role in
            let group = chosen.filter { roles[$0.id] == role }
            return stride(from: 0, to: group.count, by: 3).map { Array(group[$0..<min($0 + 3, group.count)]) }
        }
    }
    /// One role and fresh sentence groups per visit; timing/alignment failures do
    /// not silently schedule the same completed sentence again in every session.
    func variedSentences(role:DataRole,limit:Int,excluding:Set<String> = [])->[SentencePrompt] {
        guard role != .finalTest else {return []}
        let recorded=Set(pairs.filter { $0.spoken != nil || $0.silent != nil }.flatMap(\.promptIDs))
        return Array(prompts.filter {$0.kind == .sentence && roles[$0.id]==role && !recorded.contains($0.id) && !excluding.contains($0.id)}
            .prefix(max(0,limit)))
    }

    func repairWords(lexicon: Lexicon, target: Int = 4) -> [String] {
        // Sentences get the first opportunity to fill gaps. A partial visit cannot trigger
        // a long isolated-word list before sentence collection/review is complete.
        let development = prompts.filter { $0.kind == .sentence && roles[$0.id] == .training }
        guard !development.isEmpty,
              development.allSatisfy({ prompt in pairs.contains {
                  $0.promptIDs.contains(prompt.id) && $0.role == .training && $0.usesAlignedWords
              } }) else { return [] }
        let counts = coverage(lexicon: lexicon)
        let needs = Dictionary(uniqueKeysWithValues: Phoneme.all.map { ($0.symbol, target - counts[$0.symbol, default: 0]) })
        var generator = SplitMix64(seed: 0x5E17)
        let candidates = Lexicon.vocabulary.prefix(4000).map(\.word).filter {
            let phones = lexicon.phones($0) ?? []
            return !phones.isEmpty && phones.filter(Lexicon.vowels.contains).count <= 3
        }
        return CoveragePlanner.plan(needs: needs, candidates: candidates, lexicon: lexicon, avoiding: [], using: &generator)
    }
}

enum SentenceStore {
    static var directory: URL {
        let directory = URL.applicationSupportDirectory.appending(path: "Subvox", directoryHint: .isDirectory)
        return directory
    }
    static var url: URL { directory.appending(path: "sentence-training.json") }
    static func load(from location: URL? = nil) throws -> SentenceTrainingData {
        let location = location ?? url
        guard FileManager.default.fileExists(atPath: location.path) else { return SentenceTrainingData() }
        return try JSONDecoder().decode(SentenceTrainingData.self, from: Data(contentsOf: location))
    }
    static func save(_ data: SentenceTrainingData, to location: URL? = nil) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let target = location ?? url
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoder.encode(data).write(to: target, options: .atomic)
    }
}

/// Output of `subvox-ml phone-check` for one spoken take.
struct PhoneCheckResult: Codable, Sendable, Hashable {
    struct Step: Codable, Sendable, Hashable { var op: String; var expected: String?; var produced: String? }
    var language: String
    var expectedPhones: [String]
    var producedPhones: [String]
    var source: String
    var alignment: [Step]
    var wordAgreement: Double
    var phoneErrorRate: Double
    var trainable: Bool
    enum CodingKeys: String, CodingKey {
        case language, source, alignment, trainable
        case expectedPhones = "expected_phones", producedPhones = "produced_phones"
        case wordAgreement = "word_agreement", phoneErrorRate = "phone_error_rate"
    }
    var mismatches: [Step] { alignment.filter { $0.op != "match" } }
}
