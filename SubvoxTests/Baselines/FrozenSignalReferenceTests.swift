import XCTest
@testable import Subvox

final class FrozenSignalReferenceTests: XCTestCase {
    private func source(directory: URL = URL(fileURLWithPath: "/tmp/source-fixture")) -> SignalCheckRun {
        SignalCheckTestData.source(directory: directory)
    }

    private func second(_ source: SignalCheckRun, reference: FrozenSignalReference) throws -> SignalCheckRun {
        let reseated = Date()
        var run = SignalCheckRun(seed: 53, session: .init(id: UUID(), directory: URL(fileURLWithPath: "/tmp/second-fixture"), startUptime: 1100, audioOffset: 0),
            reference: try reference.link(), reseatedAt: reseated)
        run.endedAt = Date()
        run.takes = source.takes.map { original in
            var take = original; take.id = UUID(); take.recording.id = UUID()
            take.recording.samples = take.recording.samples.map { sample in var s = sample; s.t += 1000; return s }
            take.recording.cueStart += 1000; take.recording.cueEnd += 1000
            return take
        }
        return run
    }

    func testNewSessionUsesOnlyFrozenReferencesAndExpectedLabelsDoNotInfluencePredictions() throws {
        let source = source(), frozen = try FrozenSignalReference.freeze(source)
        var test = try second(source, reference: frozen)
        let initialLink = try frozen.link(), initialGroups = frozen.groups.map(\.scales)
        let report = try frozen.evaluate(test)
        for result in report.conditions {
            XCTAssertEqual(result.correct, 9)
            XCTAssertEqual(result.evaluated, 9)
            XCTAssertEqual(result.plannedTakes, 9)
            XCTAssertEqual(result.rejectedTakes, 0)
        }
        XCTAssertEqual(report.reference, initialLink)
        let sourceIDs = Set(source.takes.map(\.id))
        XCTAssertTrue(try XCTUnwrap(report.predictions).allSatisfy { sourceIDs.contains($0.referenceTakeID!) })
        let predictions = Dictionary(uniqueKeysWithValues: report.predictions!.map { ($0.takeID, $0.predictedSentence) })
        // Deliberately wrong evaluation labels must not become new templates or change predictions.
        for i in test.takes.indices { test.takes[i].prompt.sentence = (test.takes[i].prompt.sentence + 1) % 3 }
        let relabelled = try frozen.evaluate(test)
        XCTAssertTrue(relabelled.conditions.allSatisfy { $0.correct == 0 })
        XCTAssertEqual(Dictionary(uniqueKeysWithValues: relabelled.predictions!.map { ($0.takeID, $0.predictedSentence) }), predictions)
        XCTAssertEqual(try frozen.link(), initialLink)
        XCTAssertEqual(frozen.groups.map(\.scales), initialGroups)
    }

    func testPartialInvalidAndUnmatchableTakesRemainVisibleInDenominators() throws {
        let source = source(), frozen = try FrozenSignalReference.freeze(source)
        var test = try second(source, reference: frozen)
        for i in test.takes.indices {
            test.takes[i].recording.samples = test.takes[i].recording.samples.map { MotionSample(t: $0.t) }
        }
        let flat = try frozen.evaluate(test)
        XCTAssertTrue(flat.conditions.allSatisfy { $0.correct == 0 && $0.evaluated == 9 && $0.rejectedTakes == 9 && $0.plannedTakes == 9 })
        test.takes = Array(test.takes.prefix(2))
        test.takes[0].endReason = .timeout
        let partial = try frozen.evaluate(test)
        for condition in [SpeechCondition.normal, .silent] {
            let result = try XCTUnwrap(partial.conditions.first { $0.condition == condition })
            XCTAssertEqual(result.plannedTakes, 9)
            XCTAssertEqual(result.collectedTakes, 1)
        }
        XCTAssertEqual(partial.conditions.first { $0.condition == .normal }?.evaluated, 0)
        XCTAssertNotNil(partial.predictions?.first { $0.takeID == test.takes[0].id }?.reason)
    }

    func testReferenceSessionAndReusedTakeIdentitiesCannotMasqueradeAsNewTests() throws {
        let source = source(), frozen = try FrozenSignalReference.freeze(source)
        var test = try second(source, reference: frozen)
        test.session?.id = source.session!.id
        XCTAssertThrowsError(try frozen.evaluate(test))
        test = try second(source, reference: frozen); test.takes[0].id = source.takes[0].id
        XCTAssertThrowsError(try frozen.evaluate(test))
        test = try second(source, reference: frozen); test.reseatedAt = nil
        XCTAssertThrowsError(try frozen.evaluate(test))
        test = try second(source, reference: frozen); test.reference = .init(id: frozen.id, sourceRunID: frozen.sourceRunID, sha256: "changed")
        XCTAssertThrowsError(try frozen.evaluate(test))
        test = try second(source, reference: frozen); test.takes.append(test.takes[0])
        XCTAssertThrowsError(try frozen.evaluate(test))
    }

    func testFreezeRequiresCompleteUsableSourceAndRejectsTestRunsAsReferences() throws {
        var run = source(); run.takes.removeLast()
        XCTAssertThrowsError(try FrozenSignalReference.freeze(run))
        run = source(); run.takes[0].endReason = .timeout
        XCTAssertThrowsError(try FrozenSignalReference.freeze(run))
        run = source(); run.takes[0].recording.samples = run.takes[0].recording.samples.map { MotionSample(t: $0.t) }
        XCTAssertThrowsError(try FrozenSignalReference.freeze(run))
        run = source(); let frozen = try FrozenSignalReference.freeze(run)
        XCTAssertThrowsError(try FrozenSignalReference.freeze(second(run, reference: frozen)))
    }

    func testFrozenArtifactSurvivesReloadRefusesOverwriteAndDetectsChangedBytes() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-frozen-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appending(path: "reference.json"), frozen = try FrozenSignalReference.freeze(source())
        try frozen.saveNew(to: file)
        let original = try Data(contentsOf: file)
        let restored = try FrozenSignalReference.load(from: file)
        XCTAssertEqual(try restored.link(), try frozen.link())
        XCTAssertEqual(restored.groups.map(\.scales), frozen.groups.map(\.scales))
        XCTAssertThrowsError(try FrozenSignalReference.freeze(source()).saveNew(to: file))
        XCTAssertEqual(try Data(contentsOf: file), original)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: original) as? [String: Any])
        var reference = try XCTUnwrap(json["reference"] as? [String: Any]); reference["algorithmVersion"] = 999
        json["reference"] = reference
        try JSONSerialization.data(withJSONObject: json).write(to: file)
        XCTAssertThrowsError(try FrozenSignalReference.load(from: file))
    }

    @MainActor func testReseatPreparationPersistsModelBeforeAcquisitionAndKeepsSourceFileUnchanged() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-reseat-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appending(path: "signal.json"), source = source(directory: root)
        let bytes = try JSONEncoder().encode(source); try bytes.write(to: store)
        let workflow = SignalCheckWorkflow(storeURL: store)
        workflow.prepareReseat()
        XCTAssertEqual(workflow.phase, .reseat)
        XCTAssertEqual(workflow.reference?.sourceRunID, source.id)
        XCTAssertEqual(try Data(contentsOf: store), bytes)
        let frozenFile = root.appending(path: "frozen-signal-reference.json"), frozenBytes = try Data(contentsOf: frozenFile)
        workflow.prepareForTermination()
        XCTAssertEqual(try Data(contentsOf: store), bytes)
        let restored = SignalCheckWorkflow(storeURL: store)
        XCTAssertEqual(restored.referenceLink, workflow.referenceLink)
        restored.prepareReseat(); restored.stop()
        let recorder = SessionRecorder(root: root.appending(path: "raw"))
        let session = try XCTUnwrap(recorder.start(audioSink: nil))
        let reseated = Date()
        restored.begin(session: session, audioDevice: "Fixture Mac microphone", seed: 8, recorder: recorder,
            reference: restored.referenceLink, reseatedAt: reseated)
        XCTAssertEqual(restored.run?.reference, workflow.referenceLink)
        XCTAssertEqual(restored.run?.reseatedAt, reseated)
        XCTAssertNotEqual(restored.run?.id, source.id)
        XCTAssertEqual(try Data(contentsOf: session.directory.appending(path: "frozen-signal-reference.json")), frozenBytes)
        restored.prepareForTermination()
        XCTAssertEqual(try Data(contentsOf: frozenFile), frozenBytes)
        XCTAssertFalse(recorder.isRecording)
    }

    func testAlignerAcceptsSourceOnlyScalesAndRejectsInvalidFrozenScales() throws {
        let samples = source().takes[0].recording.cueSamples
        let anchor = MotionWordAnchor(word: "whole", start: samples.first!.t, end: samples.last!.t)
        let scales = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
        XCTAssertEqual(try SentenceMotionAlignment.align(voiced: samples, silent: samples, words: [anchor], referenceScales: scales).normalizedCost, 0)
        let louder = samples.map { original in
            var s = original
            s.ax *= 100; s.ay *= 100; s.az *= 100
            s.rx *= 100; s.ry *= 100; s.rz *= 100
            return s
        }
        let adaptive = try SentenceMotionAlignment.align(voiced: louder, silent: samples, words: [anchor]).normalizedCost
        let fixed = try SentenceMotionAlignment.align(voiced: louder, silent: samples, words: [anchor], referenceScales: scales).normalizedCost
        XCTAssertGreaterThan(fixed, adaptive * 10) // New amplitudes must not refit the frozen channel scales.
        for bad in [[], [0.1], [0, 0, 0, 0, 0, 0], [Double.nan, 1, 1, 1, 1, 1]] {
            XCTAssertThrowsError(try SentenceMotionAlignment.align(voiced: samples, silent: samples, words: [anchor], referenceScales: bad)) {
                XCTAssertEqual($0 as? SentenceMotionAlignment.Failure, .invalidScales)
            }
        }
    }
}

enum SignalCheckTestData {
    static func source(directory: URL = URL(fileURLWithPath: "/tmp/source-fixture")) -> SignalCheckRun {
        var run = SignalCheckRun(seed: 42, session: .init(id: UUID(), directory: directory, startUptime: 100, audioOffset: 0))
        run.endedAt = Date()
        run.takes = SignalCheckRun.plan(seed: 42).map { prompt in
            let samples = (0..<251).map { i -> MotionSample in
                let time = Double(i) / 50
                if time < 3 { return MotionSample(t: 100 + time, rotation: (0.001, 0, 0)) }
                let phase = time - 3, sentence = Double(prompt.sentence)
                let a = 0.1 * sin(phase * (3 + sentence * 4)), b = 0.07 * cos(phase * (7 + sentence * 3))
                return MotionSample(t: 100 + time, accel: (a / 10, b / 10, 0), rotation: (a, b, 0.03 * sin(phase * 5)))
            }
            return .init(prompt: prompt, recording: .init(condition: prompt.condition, samples: samples, cueStart: 103, cueEnd: 105), endReason: .done)
        }
        return run
    }

}
