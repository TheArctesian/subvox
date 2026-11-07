import XCTest
@testable import Subvox

final class SignalCheckTests: XCTestCase {
    func testSlowerPacingFitsTheLocalAudioLimitAtEverySelectableRoundCount() {
        for pace in SignalCheckPace.allCases {
            XCTAssertGreaterThanOrEqual(pace.cue, 5)
            XCTAssertGreaterThanOrEqual(pace.baseline, 2)
            for rounds in 2...pace.maximumRounds(wordCount: 6) {
                XCTAssertLessThan(Double(rounds * 6 * 3) * pace.trialSeconds, 900)
            }
        }
        let samples = (0..<500).map { MotionSample(t: Double($0) / 50, rotation: (0.04, 0, 0)) }
        let diagnostic = Template(label: "joy", samples: samples, purpose: .signalMouthed)
        let regular = Template(label: "joy", samples: samples)
        XCTAssertFalse(diagnostic.quality.issues.contains("Long recording"))
        XCTAssertTrue(regular.quality.issues.contains("Long recording"))
    }

    func testFreshExperimentCanCollectWithoutReusingReservedWords() {
        var experiment = TrainingExperiment()
        var generator = SplitMix64(seed: 8)
        experiment.assign(SignalCheckPlan.seedWords, using: &generator)
        let words = SignalCheckPlan.words(in: experiment)
        XCTAssertEqual(words.count, 6)
        XCTAssertEqual(experiment.wordRoles.values.filter { $0 == .finalTest }.count, 3)
        XCTAssertTrue(words.allSatisfy { experiment.wordRoles[$0] == .training })
    }

    func testPlanNeverRecruitsReservedWordsAndBalancesEveryConditionWithinRounds() {
        var experiment = TrainingExperiment()
        experiment.wordRoles = ["joy": .training, "back": .training, "go": .training,
                                "good": .training, "huge": .training, "next": .training,
                                "who": .finalTest, "oil": .validation]
        let words = SignalCheckPlan.words(in: experiment)
        XCTAssertEqual(words.count, 6)
        XCTAssertTrue(words.allSatisfy { experiment.wordRoles[$0] == .training })
        var generator = SplitMix64(seed: 42)
        let trials = SignalCheckPlan.trials(words: words, rounds: 3, using: &generator)
        XCTAssertEqual(trials.count, 54)
        for round in 0..<3 {
            let block = trials[(round * 18)..<((round + 1) * 18)]
            for word in words {
                let purposes = block.filter { $0.word == word }.map(\.purpose)
                XCTAssertEqual(Set(purposes), Set([.signalMouthed, .cueOnly, .voicedReference]))
                XCTAssertEqual(purposes.count, 3)
            }
        }
        XCTAssertNotEqual(trials.prefix(18).map(\.word), words.sorted().flatMap { [$0, $0, $0] })
    }

    func testAllDiagnosticPurposesStayOutOfTrainingEvenWhenAccepted() throws {
        let samples = (0..<50).map { MotionSample(t: Double($0) / 50, rotation: (0.04, 0, 0)) }
        var experiment = TrainingExperiment()
        experiment.wordRoles = ["joy": .training]
        var takes = [Template(label: "joy", samples: samples, review: .accepted)]
        for purpose in [RecordingPurpose.signalMouthed, .cueOnly, .voicedReference] {
            let take = Template(label: "joy", samples: samples, review: .accepted, purpose: purpose,
                                cueStartUptime: 0.2, cueEndUptime: 0.9)
            XCTAssertFalse(take.isUsable)
            let restored = try JSONDecoder().decode(Template.self, from: JSONEncoder().encode(take))
            XCTAssertEqual(restored.purpose, purpose)
            XCTAssertEqual(restored.cueStartUptime, 0.2)
            XCTAssertFalse(restored.isUsable)
            takes.append(take)
        }
        XCTAssertEqual(experiment.fittingTakes(from: takes).count, 1)
        experiment.frozenAt = Date()
        XCTAssertEqual(experiment.fittingTakes(from: takes).count, 1)
        XCTAssertFalse(takes[2].isAuditableSpeech) // mouth-still control
        XCTAssertTrue(takes[1].isAuditableSpeech)  // mouthed diagnostic speech
    }

    func testCueClockAccountsForAudioOffsetAndRejectsMissingAnchors() throws {
        var take = Template(label: "joy", samples: [], cueStartUptime: 1001, cueEndUptime: 1003)
        let range = try XCTUnwrap(SignalCueTiming.audioRange(for: take, startUptime: 1000, audioOffset: 0.15))
        XCTAssertEqual(range.lowerBound, 0.85, accuracy: 1e-9)
        XCTAssertEqual(range.upperBound, 2.85, accuracy: 1e-9)
        XCTAssertNil(SignalCueTiming.audioRange(for: take, startUptime: 1000, audioOffset: nil))
        take.cueEndUptime = 1000
        XCTAssertNil(SignalCueTiming.audioRange(for: take, startUptime: 1000, audioOffset: 0))
    }

    func testPerTrialInstructionsFollowUndoAndSkipWithoutChangingTheLabel() {
        var session = TrainingSession(prompts: ["joy", "joy", "joy"],
                                      purposes: [.signalMouthed, .cueOnly, .voicedReference])
        XCTAssertEqual(session.currentPurpose, .signalMouthed)
        session.record(UUID())
        XCTAssertEqual(session.currentPurpose, .cueOnly)
        session.skip()
        XCTAssertEqual(session.currentPurpose, .voicedReference)
        _ = session.undo()
        XCTAssertEqual(session.currentPurpose, .cueOnly)
        XCTAssertEqual(session.current, "joy")
    }
}
