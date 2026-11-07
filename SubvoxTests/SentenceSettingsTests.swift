import XCTest
@testable import Subvox

final class SentenceSettingsTests: XCTestCase {
    @MainActor
    func testSettingsMigrationCannotOverwriteAnUnreadableSentenceStore() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-settings-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), original = Data("not readable JSON".utf8)
        try original.write(to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertFalse(workflow.storageReady)
        workflow.initializeSettings(from: DecoderSettings())
        workflow.learningMethod = .jawHMM
        XCTAssertEqual(try Data(contentsOf: url), original)
    }

    @MainActor
    func testSentenceMethodAndDecodeThresholdsRemainEditableWithFrozenWordExperiment() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-settings-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), prompt = SentencePrompt(text: "blue bee")
        try SentenceStore.save(SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training]), to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        let suite = "subvox-settings-test-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(true, forKey: "useReviewedSentenceModel")
        var experiment = TrainingExperiment(wordRoles: ["old": .training])
        experiment.settings.method = .coarseMedoids
        experiment.settings.minConfidence = 0.07
        experiment.frozenAt = Date()
        let original = experiment.settings
        let model = AppModel(dataset: Dataset(templates: [], experiment: experiment), sentenceTraining: workflow, defaults: defaults)
        XCTAssertTrue(model.experimentLocked)
        XCTAssertFalse(model.decodeSettingsLocked)
        XCTAssertEqual(workflow.settings, original) // Backward-compatible one-time copy.
        model.sentenceLearningMethod = .jawHMM
        model.minConfidence = 0.3
        model.maxDistance = 4.5
        model.hmmMaxScore = 17
        model.hmmMinGap = 0.4
        model.commonBias = 0.02
        XCTAssertEqual(model.decodeSettings.method, .jawHMM)
        XCTAssertEqual(model.minConfidence, 0.3)
        XCTAssertEqual(model.maxDistance, 4.5)
        XCTAssertEqual(model.hmmMaxScore, 17)
        XCTAssertEqual(model.hmmMinGap, 0.4)
        XCTAssertEqual(model.commonBias, 0.02)
        XCTAssertEqual(model.settings, original)
        model.learningMethod = .legacyMedoids
        XCTAssertEqual(model.settings, original) // The old final experiment stays locked.
        let restored = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertEqual(restored.settings, model.decodeSettings)
        restored.initializeSettings(from: DecoderSettings())
        XCTAssertEqual(restored.settings, model.decodeSettings) // Reopening cannot reset personal tuning.
        model.decoderSource = .words
        XCTAssertTrue(model.decodeSettingsLocked)
        XCTAssertEqual(model.decodeSettings, original)
        model.minConfidence = 0.5
        XCTAssertEqual(model.settings, original)
        XCTAssertEqual(workflow.settings.minConfidence, 0.3)
        XCTAssertFalse(defaults.bool(forKey: "useReviewedSentenceModel"))
    }

    @MainActor
    func testSentenceSettingsInvalidateOnlyTheirReportAndRespectTheirOwnLock() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-settings-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), prompt = SentencePrompt(text: "bee")
        var data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training])
        data.validationReport = SentenceValidationReport(trainingPairIDs: [], validationPairIDs: [], settings: DecoderSettings(),
            manifest: MotionWordDecoder.manifest(takes: [], lexicon: Lexicon(entries: [:]), vocabulary: [], settings: DecoderSettings()),
            expected: [], predicted: [], accepted: [])
        try SentenceStore.save(data, to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        workflow.learningMethod = .jawHMMInitial
        XCTAssertNil(try SentenceStore.load(from: url).validationReport)
        data = try SentenceStore.load(from: url)
        data.reservedUnlockedAt = Date()
        try SentenceStore.save(data, to: url)
        let locked = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertTrue(locked.settingsLocked)
        locked.learningMethod = .legacyMedoids
        XCTAssertEqual(locked.settings.method, .jawHMMInitial)
    }
}
