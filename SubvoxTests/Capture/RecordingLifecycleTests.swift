import AVFoundation
import XCTest
@testable import Subvox

final class RecordingLifecycleTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "subvox-lifecycle-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @MainActor
    func testRecorderCannotStopOrCheckpointAReplacementSession() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = SessionRecorder(root: root)
        defer { recorder.stop() }
        let original = try XCTUnwrap(recorder.start(audioSink: nil))
        recorder.stop(sessionID: original.id)
        let replacement = try XCTUnwrap(recorder.start(audioSink: nil))
        XCTAssertNotEqual(original.id, replacement.id)
        XCTAssertNil(recorder.stop(sessionID: original.id))
        XCTAssertNil(try recorder.checkpoint(sessionID: original.id))
        let staleCheckpoint = try await recorder.checkpointCompletedTake(sessionID: original.id)
        XCTAssertNil(staleCheckpoint)
        XCTAssertEqual(recorder.activeSessionID, replacement.id)
    }

    @MainActor
    func testFinishedSessionKeepsItsOwnClockAfterAnotherSessionFinishes() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = SessionRecorder(root: root)
        let original = try XCTUnwrap(recorder.start(audioSink: nil))
        let finished = try XCTUnwrap(recorder.stop(sessionID: original.id))
        let replacement = try XCTUnwrap(recorder.start(audioSink: nil))
        recorder.stop(sessionID: replacement.id)
        XCTAssertEqual(finished.id, original.id)
        XCTAssertEqual(finished.directory, original.directory)
        XCTAssertEqual(finished.startUptime, original.startUptime)
        XCTAssertEqual(recorder.lastSession?.id, replacement.id)
    }

    @MainActor
    func testSavedSentenceHalvesAlreadyContainAudioClockBeforeStop() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let storeURL = root.appending(path: "sentences.json")
        let prompt = SentencePrompt(text: "a blue bee")
        try SentenceStore.save(SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training]), to: storeURL)
        let workflow = SentenceTrainingWorkflow(storeURL: storeURL)
        let recorder = SessionRecorder(root: root), sink = AudioSink()
        defer { recorder.stop() }
        let session = try XCTUnwrap(recorder.start(audioSink: sink, audioDevice: "Synthetic fixture"))
        let visit = try workflow.beginVisit(session: session, audioDevice: "Synthetic fixture", timing: .comfortable, presentation: .passage)
        let pair = try workflow.beginPair(prompts: [prompt], visit: visit, session: session)
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 256))
        buffer.frameLength = 256
        for i in 0..<256 { buffer.floatChannelData![0][i] = 0.1 * sin(Float(i) / 7) }
        sink.dispatch(buffer, AVAudioTime(hostTime: AVAudioTime.hostTime(forSeconds: session.startUptime + 0.25)))
        let checkpoint = try await recorder.checkpointCompletedTake(sessionID: session.id)
        let clock = try XCTUnwrap(checkpoint)
        XCTAssertEqual(try XCTUnwrap(clock.audioOffset), 0.25, accuracy: 1e-6)
        let samples = (0..<51).map { MotionSample(t: session.startUptime + 1 + Double($0) / 50, rotation: (0.1, 0, 0)) }
        let spoken = SentenceRecording(condition: .normal, samples: samples, cueStart: session.startUptime + 1, cueEnd: session.startUptime + 2)
        try workflow.saveCompletedHalf(spoken, pairID: pair.id, session: clock)
        var disk = try SentenceStore.load(from: storeURL)
        XCTAssertNotNil(disk.pairs[0].spoken)
        XCTAssertNil(disk.pairs[0].silent)
        XCTAssertEqual(disk.pairs[0].directory, session.directory.path)
        XCTAssertEqual(disk.pairs[0].sessionStartUptime, session.startUptime)
        XCTAssertEqual(try XCTUnwrap(disk.pairs[0].audioOffset), 0.25, accuracy: 1e-6)
        XCTAssertEqual(disk.visits[0].pairIDs, [pair.id])
        var silent = spoken; silent.condition = .silent
        try workflow.saveCompletedHalf(silent, pairID: pair.id, session: clock)
        disk = try SentenceStore.load(from: storeURL)
        XCTAssertTrue(disk.pairs[0].complete)
        XCTAssertTrue(recorder.isRecording)
        XCTAssertNil(disk.visits[0].endedAt)
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: session.directory.appending(path: "meta.json"))) as? [String: Any])
        XCTAssertEqual(meta["complete"] as? Bool, false)
        XCTAssertEqual(meta["sessionID"] as? String, session.id.uuidString)
        XCTAssertEqual(meta["audioStartOffsetSeconds"] as? Double ?? -1, 0.25, accuracy: 1e-6)
        // A normal quit must synchronously finalize the file before async task cleanup.
        workflow.prepareForTermination(recorder: recorder)
        XCTAssertFalse(recorder.isRecording)
        disk = try SentenceStore.load(from: storeURL)
        XCTAssertNotNil(disk.visits[0].endedAt)
        XCTAssertTrue(disk.pairs[0].complete)
        XCTAssertEqual(disk.pairs[0].audioOffset, clock.audioOffset)
        let audio = try AVAudioFile(forReading: session.directory.appending(path: "audio.caf"))
        XCTAssertEqual(audio.length, 256)
        let recovered = SentenceTrainingWorkflow(storeURL: storeURL)
        XCTAssertTrue(recovered.storageReady)
        XCTAssertEqual(recovered.data.pairs[0].recordingSessionID, session.id)
    }

    @MainActor
    func testTerminationDoesNotStopReplacementOrOverwriteOriginalClock() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "sentences.json"), prompt = SentencePrompt(text: "blue bee")
        try SentenceStore.save(SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training]), to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url), recorder = SessionRecorder(root: root)
        defer { recorder.stop() }
        let original = try XCTUnwrap(recorder.start(audioSink: nil))
        let visit = try workflow.beginVisit(session: original, audioDevice: nil, timing: .comfortable, presentation: .sentences)
        let pair = try workflow.beginPair(prompts: [prompt], visit: visit, session: original)
        recorder.stop(sessionID: original.id)
        let replacement = try XCTUnwrap(recorder.start(audioSink: nil))
        let invalidHalf = SentenceRecording(condition: .normal, samples: [], cueStart: 1, cueEnd: 2)
        XCTAssertThrowsError(try workflow.saveCompletedHalf(invalidHalf, pairID: pair.id, session: replacement))
        workflow.prepareForTermination(recorder: recorder)
        XCTAssertEqual(recorder.activeSessionID, replacement.id)
        let stored = try SentenceStore.load(from: url)
        XCTAssertEqual(stored.pairs[0].directory, original.directory.path)
        XCTAssertEqual(stored.pairs[0].sessionStartUptime, original.startUptime)
        XCTAssertNil(stored.pairs[0].spoken)
    }
}
