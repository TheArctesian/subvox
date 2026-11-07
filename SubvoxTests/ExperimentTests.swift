import XCTest
@testable import Subvox

final class ExperimentTests: XCTestCase {
    private func take(_ word: String, condition: SpeechCondition? = .silent, review: ReviewDecision? = .accepted) -> Template {
        let samples = (0..<50).map { i in
            MotionSample(t: Double(i) / 50, rotation: (0.04 * sin(Double(i) / 8), 0, 0))
        }
        return Template(label: word, samples: samples, condition: condition, review: review)
    }

    func testPersistedWordAssignmentsNeverMoveOnAppendOrRetry() throws {
        var experiment = TrainingExperiment()
        var generator = SplitMix64(seed: 12)
        experiment.assign((0..<100).map { "word\($0)" }, using: &generator)
        XCTAssertEqual(experiment.wordRoles.values.filter { $0 == .finalTest }.count, 25)
        XCTAssertEqual(experiment.wordRoles.values.filter { $0 == .validation }.count, 19)
        let original = experiment.wordRoles
        var restored = try JSONDecoder().decode(TrainingExperiment.self, from: JSONEncoder().encode(experiment))
        restored.assign([" WORD1 ", "word2", "new1", "new2", "new3", "new4"], using: &generator)
        for (word, role) in original { XCTAssertEqual(restored.wordRoles[word], role) }
        XCTAssertEqual(restored.wordRoles.count, 104)
        XCTAssertEqual(restored.wordRoles.values.filter { $0 == .finalTest }.count, 26)
        XCTAssertEqual(restored.role(of: take("WORD1")), original["word1"])
    }

    func testOnlyAcceptedTrainingTakesFitUntilFrozenAndFinalNeverFits() {
        var experiment = TrainingExperiment()
        experiment.wordRoles = ["train": .training, "pending": .training, "bad": .training,
                                "validation": .validation, "final": .finalTest, "voiced": .training]
        var pending = take("pending", review: .pending)
        // A user's explicit pending decision excludes even a clean take.
        XCTAssertFalse(pending.isUsable)
        pending.review = .excluded
        var bad = take("bad")
        bad.samples[1].t = bad.samples[0].t
        let takes = [take("train"), pending, bad, take("validation"), take("final"), take("voiced", condition: .normal)]
        XCTAssertEqual(experiment.fittingTakes(from: takes).map(\.label), ["train"])
        experiment.frozenAt = Date()
        XCTAssertEqual(experiment.fittingTakes(from: takes).map(\.label), ["train", "validation"])
        XCTAssertFalse(experiment.fittingTakes(from: takes).contains { $0.label == "final" })
    }

    func testFrozenExperimentKeepsItsLedgerAndReportAcrossReload() throws {
        var experiment = TrainingExperiment()
        experiment.wordRoles = ["training": .training, "validation": .validation, "reserved": .finalTest]
        experiment.frozenAt = Date()
        let trainingID = UUID()
        experiment.finalReport = DecoderReport(role: .finalTest, settings: experiment.settings,
            trainingIDs: [trainingID], rows: [.init(id: UUID(), expected: "reserved", candidates: [],
                                                   accepted: false, representable: false)],
            vocabularyCount: 1, missingPhones: ["B"])
        var copy = try JSONDecoder().decode(TrainingExperiment.self, from: JSONEncoder().encode(experiment))
        let ledger = copy.wordRoles
        copy.assign(["new word"])
        XCTAssertEqual(copy.wordRoles, ledger)
        XCTAssertNotNil(copy.frozenAt)
        XCTAssertEqual(copy.finalReport?.trainingIDs, [trainingID])
        XCTAssertEqual(copy.finalReport?.rows.count, 1)
        XCTAssertEqual(copy.finalReport?.top1, 0)
    }

    func testQualityFlagsRequireReviewAndHardInvalidDataCannotBeAccepted() {
        let clean = take("clean")
        XCTAssertTrue(clean.quality.issues.isEmpty)
        XCTAssertEqual(clean.quality.sampleRate ?? 0, 50, accuracy: 1e-6)
        var still = clean
        still.samples = (0..<50).map { MotionSample(t: Double($0) / 50) }
        still.review = nil
        XCTAssertEqual(still.reviewDecision, .pending)
        still.review = .accepted
        XCTAssertTrue(still.isUsable) // Soft warnings can be manually accepted.
        var gap = clean
        for i in 20..<gap.samples.count { gap.samples[i].t += 0.3 }
        XCTAssertTrue(gap.quality.issues.contains("Gap in sensor samples"))
        var invalid = clean
        invalid.samples[1].ax = .nan
        XCTAssertTrue(invalid.quality.invalid)
        XCTAssertFalse(invalid.isUsable) // An accepted flag never overrides corrupt values.
    }

    func testLegacyJSONLoadsWithoutInventingSessionMetadata() throws {
        let legacy = """
        {"templates":[{"id":"00000000-0000-0000-0000-000000000001","label":"old","samples":[],"createdAt":1}],"phoneTargets":{"AA":4}}
        """
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(legacy.utf8))
        XCTAssertEqual(dataset.templates.count, 1)
        XCTAssertNil(dataset.templates[0].sessionID)
        XCTAssertNil(dataset.templates[0].condition)
        XCTAssertNil(dataset.experiment)
        let restored = try JSONDecoder().decode(Dataset.self, from: JSONEncoder().encode(dataset))
        XCTAssertEqual(restored.templates, dataset.templates)
    }

    func testSessionMetadataRoundTripsAndKeepsReviewState() throws {
        let visit = RecordingVisit(condition: .whisper, notes: "Fit adjusted", sampleRate: 50, prompts: ["hello"])
        var template = take("hello", condition: .whisper, review: .pending)
        template.sessionID = visit.id
        template.measuredSampleRate = 50
        var dataset = Dataset(templates: [template])
        dataset.visits = [visit]
        let copy = try JSONDecoder().decode(Dataset.self, from: JSONEncoder().encode(dataset))
        XCTAssertEqual(copy.templates[0].sessionID, copy.visits?[0].id)
        XCTAssertEqual(copy.visits?[0].notes, "Fit adjusted")
        XCTAssertEqual(copy.templates[0].review, .pending)
    }
}
