import XCTest
@testable import Subvox

final class SentenceSettingsTests: XCTestCase {
    @MainActor
    func testReviewedTrainingPairsAutomaticallyBuildAndRemoveTheLiveModel() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-auto-model-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), prompt = SentencePrompt(text: "bee bee")
        let samples = (0..<101).map { MotionSample(t: 100 + Double($0) / 50,
            accel: (0.02 * sin(Double($0) / 4), 0, 0), rotation: (0.1 * cos(Double($0) / 5), 0, 0)) }
        var pair = SentencePair(visitID: UUID(), promptIDs: [prompt.id], text: prompt.text, role: .training)
        pair.spoken = SentenceRecording(condition: .normal, samples: samples, cueStart: 100, cueEnd: 102)
        pair.silent = SentenceRecording(condition: .silent, samples: samples, cueStart: 100, cueEnd: 102)
        pair.alignment = MotionTransferResult(normalizedCost: 0, durationRatio: 1, path: [], words: [
            .init(word: "bee", voicedStart: 100, voicedEnd: 101, silentStart: 100, silentEnd: 101),
            .init(word: "bee", voicedStart: 101, voicedEnd: 102, silentStart: 101, silentEnd: 102)], warnings: [])
        try SentenceStore.save(SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training], pairs: [pair]), to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url), model = AppModel(sentenceTraining: workflow)
        XCTAssertNil(model.decoder)
        workflow.review(pair.id, decision: .accepted)
        for _ in 0..<200 {
            if !model.isBuildingDecoder { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertNotNil(model.decoder)
        XCTAssertGreaterThan(model.decoder?.wordCount ?? 0, 0)
        model.screen = .decode
        model.decodeMode = .auto
        XCTAssertTrue(model.capture.autoEnabled)
        workflow.review(pair.id, decision: .excluded)
        XCTAssertNil(model.decoder)
        XCTAssertFalse(model.capture.autoEnabled)
        XCTAssertTrue(workflow.fittingWords.isEmpty)
    }

    @MainActor
    func testSettingsCannotOverwriteAnUnreadableSentenceStore() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-settings-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), original = Data("not readable JSON".utf8)
        try original.write(to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertFalse(workflow.storageReady)
        workflow.learningMethod = .jawSequence
        XCTAssertEqual(try Data(contentsOf: url), original)
    }

    @MainActor
    func testAppUsesOnlySentenceSettingsAndNeverLoadsAnOldDataset() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-settings-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentence-training.json")
        let oldURL = root.appending(path: "dataset.json"), oldData = Data(#"{"templates":["old examples"],"experiment":{"frozenAt":1}}"#.utf8)
        try oldData.write(to: oldURL)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        let model = AppModel(sentenceTraining: workflow)
        XCTAssertNil(model.decoder)
        XCTAssertTrue(workflow.fittingWords.isEmpty)
        XCTAssertTrue(workflow.data.pairs.isEmpty)
        XCTAssertFalse(model.decodeSettingsLocked)
        workflow.learningMethod = .jawSequence
        model.minConfidence = 0.3
        model.maxDistance = 4.5
        model.hmmMaxScore = 17
        model.hmmMinGap = 0.4
        model.commonBias = 0.02
        XCTAssertEqual(model.decodeSettings.method, .jawSequence)
        XCTAssertEqual(model.minConfidence, 0.3)
        XCTAssertEqual(model.maxDistance, 4.5)
        XCTAssertEqual(model.hmmMaxScore, 17)
        XCTAssertEqual(model.hmmMinGap, 0.4)
        XCTAssertEqual(model.commonBias, 0.02)
        let restored = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertEqual(restored.settings, model.decodeSettings)
        XCTAssertEqual(try Data(contentsOf: oldURL), oldData)
        XCTAssertEqual(AppModel.Screen.allCases.map(\.rawValue), ["Home", "Train", "Try model", "Devices"])
    }

    @MainActor
    func testSentenceSettingsInvalidateTheirReportAndRespectTheirLock() async throws {
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
        workflow.learningMethod = .jawSequence
        XCTAssertNil(try SentenceStore.load(from: url).validationReport)
        data = try SentenceStore.load(from: url)
        data.reservedUnlockedAt = Date()
        try SentenceStore.save(data, to: url)
        let locked = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertTrue(locked.settingsLocked)
        locked.learningMethod = .motionTemplates
        XCTAssertEqual(locked.settings.method, .jawSequence)
    }
}
