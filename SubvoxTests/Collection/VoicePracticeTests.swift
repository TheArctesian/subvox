import AVFoundation
import XCTest
@testable import Subvox

final class VoicePracticeTests: XCTestCase {
    @MainActor
    func testFullRunsForAllFourModesReopenAndPlayIndependently() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let practice = VoicePractice(root: root, requestAccess: { true }, makeRecorder: SilentRecorder.init)
        let workflow = SentenceTrainingWorkflow(storeURL: root.appending(path: "paired.json"), corpus: [])
        let model = AppModel(sentenceTraining: workflow, voicePractice: practice)
        let modes = [TrainMode.language, .speech, .memorization, .free]
        let targets = modes.map {
            VoicePracticeTarget(mode: $0.rawValue, language: model.language, itemID: "example", text: "First line. Second line.")
        }
        XCTAssertNil(model.automaticSharingGrant)
        XCTAssertEqual(model.motion.status, .waiting)
        for (mode, target) in zip(modes, targets) {
            model.openPractice(mode)
            XCTAssertTrue(model.canStartVoicePractice)
            await model.startVoicePractice(target)
            XCTAssertEqual(practice.recordingTarget, target)
            XCTAssertTrue(model.navigationLocked)
            XCTAssertFalse(model.canRecordPractice)
            practice.finish()
            XCTAssertNil(practice.error)
            XCTAssertFalse(model.navigationLocked)
        }
        XCTAssertTrue(workflow.data.pairs.isEmpty, "Microphone runs must stay out of paired recordings")

        let speech = targets[1]
        await practice.start(speech)
        practice.prepareForTermination()
        let reopened = VoicePractice(root: root)
        XCTAssertEqual(reopened.runs.count, 5)
        XCTAssertEqual(reopened.recentRuns(for: speech).count, 2)
        XCTAssertEqual(reopened.recentRuns(for: speech).first, practice.runs.first)
        for target in targets {
            let run = try XCTUnwrap(reopened.recentRuns(for: target).first)
            XCTAssertEqual(run.duration, 1, accuracy: 0.1)
            reopened.play(run)
            XCTAssertEqual(reopened.playingRunID, run.id)
            XCTAssertNil(reopened.error)
            reopened.stopPlayback()
            XCTAssertNil(reopened.playingRunID)
        }
        let revised = VoicePracticeTarget(mode: speech.mode, language: speech.language,
                                          itemID: speech.itemID, text: "A revised line.")
        XCTAssertTrue(reopened.recentRuns(for: revised).isEmpty)
    }

    @MainActor
    func testCancelDuringPermissionCannotStartAnOldRunAfterRetry() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let asked = expectation(description: "Microphone permission requested")
        var permission: CheckedContinuation<Bool, Never>?
        var requests = 0
        let practice = VoicePractice(root: root, requestAccess: {
            requests += 1
            if requests > 1 { return true }
            return await withCheckedContinuation { continuation in
                permission = continuation
                asked.fulfill()
            }
        }, makeRecorder: SilentRecorder.init)
        let first = VoicePracticeTarget(mode: "Speech", language: .en, itemID: "first", text: "First.")
        let second = VoicePracticeTarget(mode: "Speech", language: .en, itemID: "second", text: "Second.")
        let start = Task { await practice.start(first) }
        await fulfillment(of: [asked], timeout: 3)
        defer { permission?.resume(returning: false) }
        XCTAssertTrue(practice.starting)
        practice.cancel()
        XCTAssertFalse(practice.isBusy)

        await practice.start(second)
        XCTAssertEqual(practice.recordingTarget, second)
        let stale = try XCTUnwrap(permission)
        permission = nil
        stale.resume(returning: true)
        await start.value
        XCTAssertEqual(practice.recordingTarget, second)
        practice.cancel()
        XCTAssertTrue(VoicePractice(root: root).runs.isEmpty)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    @MainActor
    func testDeniedPermissionAndFailedRecorderLeaveNoSavedRun() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let target = VoicePracticeTarget(mode: "Language", language: .en, itemID: "example", text: "A phrase.")
        let denied = VoicePractice(root: root, requestAccess: { false }, makeRecorder: { _ in
            XCTFail("A microphone denial must not open a recording file.")
            throw CocoaError(.fileWriteNoPermission)
        })
        await denied.start(target)
        XCTAssertNotNil(denied.error)
        XCTAssertFalse(denied.isBusy)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))

        let failed = VoicePractice(root: root, requestAccess: { true }, makeRecorder: { _ in
            throw CocoaError(.fileWriteUnknown)
        })
        await failed.start(target)
        XCTAssertNotNil(failed.error)
        XCTAssertFalse(failed.isBusy)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "voice-practice-\(UUID())", directoryHint: .isDirectory)
    }
}

private final class SilentRecorder: VoicePracticeRecorder {
    weak var delegate: AVAudioRecorderDelegate?

    init(url: URL) throws {
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1
        ])
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 44_100))
        buffer.frameLength = 44_100
        let samples = try XCTUnwrap(buffer.floatChannelData)
        samples[0].update(repeating: 0, count: Int(buffer.frameLength))
        try file.write(from: buffer)
    }

    func record() -> Bool { true }
    func stop() {}
}
