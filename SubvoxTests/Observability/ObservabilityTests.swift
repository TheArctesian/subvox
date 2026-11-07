import XCTest
@testable import Subvox

final class ObservabilityTests: XCTestCase {
    /// 50 Hz take: 1.5 s baseline then a 3 s cue. `bump` adds a smooth pulse to one axis during the cue.
    private func take(_ gesture: ObservabilityGesture, block: Int, rng: inout SplitMix64, bump: (axis: Int, size: Double)?) -> ObservabilityTake {
        let start = 100.0 + Double(block * 10)
        func noise() -> Double { (Double(rng.next() % 10_000) / 10_000 - 0.5) * 0.004 }
        let samples = (0..<225).map { i -> MotionSample in
            let t = start - 1.5 + Double(i) * 0.02
            var v = (0..<6).map { _ in noise() }
            if let bump, t >= start { v[bump.axis] += bump.size * sin(.pi * min(1, (t - start) / 3)) }
            var s = MotionSample(t: t, accel: (v[0], v[1], v[2]), rotation: (v[3], v[4], v[5]))
            s.sensorLocation = 1
            return s
        }
        return .init(gesture: gesture, block: block, samples: samples, cueStart: start, cueEnd: start + 3)
    }

    private func session(blocks: Int, signal: Bool) -> [ObservabilityTake] {
        var rng = SplitMix64(seed: 7)
        let bumps: [ObservabilityGesture: (Int, Double)] = [.jawOpen: (3, 0.05), .vowelAh: (1, 0.02), .vowelEe: (2, 0.02), .vowelOo: (0, 0.02)]
        return (0..<blocks).flatMap { block in
            ObservabilityGesture.allCases.map { g in take(g, block: block, rng: &rng, bump: signal ? bumps[g].map { ($0.0, $0.1) } : nil) }
        }
    }

    func testPlanShowsEveryGestureOncePerBlockInReproducibleOrder() {
        let plan = ObservabilitySession.plan(blocks: 5, seed: 42)
        XCTAssertEqual(plan, ObservabilitySession.plan(blocks: 5, seed: 42))
        XCTAssertEqual(plan.count, 5 * ObservabilityGesture.allCases.count)
        for block in 0..<5 {
            XCTAssertEqual(Set(plan.filter { $0.block == block }.map(\.gesture)), Set(ObservabilityGesture.allCases))
        }
        XCTAssertNotEqual(plan.prefix(7).map(\.gesture), plan.dropFirst(7).prefix(7).map(\.gesture), "Blocks should not all share one order")
    }

    func testSeparableJawAndVowelsWithIdenticalControlGiveArticulation() {
        let report = ObservabilityAnalysis.analyse(session(blocks: 10, signal: true), permutations: 199, seed: 3)
        XCTAssertEqual(report.verdict, .articulation, report.summary)
        let byID = Dictionary(uniqueKeysWithValues: report.results.map { ($0.contrast.id, $0) })
        XCTAssertTrue(byID["movement"]!.significant)
        XCTAssertTrue(byID["vowels"]!.significant)
        XCTAssertFalse(byID["control"]!.significant, "bat/pat share a distribution and must stay at chance")
        XCTAssertEqual(byID["movement"]!.confusion.flatMap { $0 }.reduce(0, +), 20)
    }

    func testPureNoiseFindsNoSignal() {
        let report = ObservabilityAnalysis.analyse(session(blocks: 10, signal: false), permutations: 199, seed: 3)
        XCTAssertEqual(report.verdict, .noSignal, report.summary)
        XCTAssertTrue(report.results.allSatisfy { !$0.significant })
    }

    func testSmallestAttainablePGatesHowManyBlocksAreNeeded() {
        XCTAssertEqual(ObservabilityAnalysis.smallestAttainableP(classes: 2, completeBlocks: 8), 1.0 / 128)
        XCTAssertEqual(ObservabilityAnalysis.smallestAttainableP(classes: 3, completeBlocks: 3), 1.0 / 36)
        for k in [2, 3] {
            XCTAssertLessThanOrEqual(ObservabilityAnalysis.smallestAttainableP(classes: k, completeBlocks: ObservabilityAnalysis.minimumBlocks), ObservabilityContrast.alpha / 2)
        }
        XCTAssertGreaterThan(ObservabilityAnalysis.smallestAttainableP(classes: 2, completeBlocks: 7), ObservabilityContrast.alpha / 2)
    }

    func testTooFewBlocksIsInsufficientNotNegative() {
        let report = ObservabilityAnalysis.analyse(session(blocks: 3, signal: true), permutations: 19)
        XCTAssertEqual(report.verdict, .insufficient)
        XCTAssertEqual(Set(report.insufficient), Set(ObservabilityContrast.all.map(\.id)))
    }

    func testDiscardedAndGappedTakesAreExcluded() {
        var takes = session(blocks: 4, signal: true)
        takes[0].discardedReason = "Discarded by Redo"
        takes[1].samples = Array(takes[1].samples.prefix(60))
        XCTAssertNil(ObservabilityAnalysis.features(takes[1]))
        XCTAssertNotNil(ObservabilityAnalysis.features(takes[2]))
        XCTAssertEqual(ObservabilityAnalysis.features(takes[2])?.count, 30)
    }

    @MainActor func testWorkflowRecordsTimedTakeRedoesAndContinues() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "observability-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        var now = 50.0
        let workflow = ObservabilityWorkflow(root: root.appending(path: "store"), clock: { now })
        let recorder = SessionRecorder(root: root.appending(path: "sessions"))
        defer { recorder.stop() }
        func feed(until end: Double) {
            while now < end { now += 0.02; var s = MotionSample(t: now); s.sensorLocation = 1; workflow.ingest(s) }
        }
        feed(until: 51)
        workflow.blocks = 4
        workflow.start(audio: AudioService(), recorder: recorder, seed: 9)
        XCTAssertEqual(workflow.phase, .ready)
        XCTAssertEqual(workflow.total, 28)
        XCTAssertEqual(recorder.condition, .observability)

        workflow.next(); XCTAssertEqual(workflow.phase, .relax)
        feed(until: 53); workflow.beginCue(at: now)
        feed(until: 56); workflow.finishTake()
        XCTAssertEqual(workflow.phase, .ended)
        let first = try XCTUnwrap(workflow.lastTake)
        XCTAssertNil(first.discardedReason)
        XCTAssertEqual(first.gesture, workflow.current?.gesture)
        XCTAssertTrue(first.samples.contains { $0.t < first.cueStart - 1 }, "Baseline before the cue is kept")

        workflow.redo()
        XCTAssertEqual(workflow.phase, .ready)
        XCTAssertEqual(workflow.lastTake?.discardedReason, "Discarded by Redo")
        XCTAssertEqual(workflow.index, 0)

        workflow.next(); feed(until: 58); workflow.beginCue(at: now); feed(until: 61); workflow.finishTake()
        workflow.next()
        XCTAssertEqual(workflow.index, 1)
        XCTAssertEqual(workflow.phase, .relax, "Continuing from a saved take starts the next one directly")

        var other = MotionSample(t: now + 0.02); other.sensorLocation = 2; now += 0.02
        workflow.ingest(other)
        XCTAssertEqual(workflow.phase, .ready, "A source switch during relax restarts the take")

        let saved = try JSONDecoder().decode(ObservabilitySession.self,
            from: Data(contentsOf: root.appending(path: "store/\(workflow.session!.id).json")))
        XCTAssertEqual(saved.takes.count, 2)
        XCTAssertEqual(saved.validTakes.count, 1)
        workflow.stop()
        XCTAssertFalse(recorder.isRecording)
    }
}
