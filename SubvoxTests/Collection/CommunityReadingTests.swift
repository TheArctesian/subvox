import XCTest
@testable import Subvox

final class CommunityReadingTests: XCTestCase {
    func testBundledCorpusLoadsCalibrationAndPassagesButNeverHeldOutText() throws {
        let corpus = CommunityCorpus.bundled
        XCTAssertEqual(corpus.calibration.count, 40)
        XCTAssertGreaterThan(corpus.passages.count, 1000)
        XCTAssertTrue(corpus.passages.allSatisfy { p in p.prompts.map(\.order) == Array(0..<p.prompts.count) }, "Passages keep source order")
        let url = try XCTUnwrap(Bundle.main.url(forResource: "heldout-v1", withExtension: "jsonl", subdirectory: "Corpus/community"))
        let held = CommunityCorpus.parse(try String(contentsOf: url, encoding: .utf8))
        XCTAssertFalse(held.isEmpty)
        XCTAssertTrue(held.allSatisfy { corpus.prompt($0.promptID) == nil }, "Held-out prompts are never assignable")
        let some = try XCTUnwrap(corpus.passages.first?.prompts.first)
        XCTAssertEqual(some.promptID, SentencePrompt.identity(some.text), "Prompt IDs match the app's sentence identity")
    }

    func testAssignmentMatchesThePythonPipeline() {
        // Expected IDs from `Tools/corpus/community.py assign` on prompts-v1.jsonl.
        let corpus = CommunityCorpus.bundled
        XCTAssertEqual(corpus.assign(counts: [:], recordedPassages: [], count: 4).map(\.id),
                       ["austen-pride-0020", "cbronte-eyre-0037", "stevenson-treasure-0146", "dickinson-poems-0040"])
        XCTAssertEqual(corpus.assign(counts: ["DH AH": 50, "AH N": 30, "T UW": 20], recordedPassages: [], count: 4).map(\.id),
                       ["whitman-leaves-0033", "cbronte-eyre-0037", "baum-oz-0063", "austen-pride-0134"])
        XCTAssertFalse(corpus.assign(counts: [:], recordedPassages: ["austen-pride-0020"], count: 4).map(\.id).contains("austen-pride-0020"))
    }

    func testCommunityPromptsJoinAsTrainingWithoutChangingThePersonalSplit() {
        var data = SentenceTrainingData()
        var rng = SplitMix64(seed: 3)
        data.add((0..<8).map { SentencePrompt(text: "personal sentence number \($0) here") }, using: &rng)
        let before = data.roles
        let community = CommunityCorpus.bundled
        data.addCommunity(community.calibration.prefix(5).map(\.sentencePrompt) + community.passages[0].prompts.map(\.sentencePrompt))
        XCTAssertEqual(data.roles.filter { before[$0.key] != nil }, before)
        XCTAssertTrue(data.prompts.filter { $0.kind != .sentence }.allSatisfy { data.roles[$0.id] == .training })
        XCTAssertTrue(data.variedSentences(role: .training, limit: 100).allSatisfy { $0.kind == .sentence }, "Banks never schedule community text")
        // A later personal addition gets the same role with or without community prompts present.
        var control = SentenceTrainingData(), controlRNG = SplitMix64(seed: 3)
        control.add((0..<8).map { SentencePrompt(text: "personal sentence number \($0) here") }, using: &controlRNG)
        let later = (0..<5).map { SentencePrompt(text: "another personal sentence \($0) to add") }
        data.add(later, using: &rng); control.add(later, using: &controlRNG)
        XCTAssertEqual(later.map { data.roles[$0.id] }, later.map { control.roles[$0.id] }, "Final/validation quotas count personal sentences only")
    }

    @MainActor func testCommunityPlanRecordsCalibrationFirstThenPassagesInOrder() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "community-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let small = CommunityCorpus(prompts: CommunityCorpus.bundled.passages.prefix(5).flatMap(\.prompts),
                                    calibration: Array(CommunityCorpus.bundled.calibration.prefix(2)))
        let fresh = SentenceTrainingWorkflow(storeURL: root.appending(path: "s.json"), community: small)
        fresh.readingSource = .community
        XCTAssertEqual(fresh.plan.flatMap { $0 }.map(\.id), small.calibration.map(\.promptID))
        XCTAssertTrue(fresh.plan.flatMap { $0 }.allSatisfy { $0.kind == .calibration })
        // A store where calibration was already mouthed: passages follow, each prompt in source order.
        var data = SentenceTrainingData()
        data.addCommunity(small.calibration.map(\.sentencePrompt))
        for prompt in small.calibration {
            var pair = SentencePair(visitID: UUID(), promptIDs: [prompt.promptID], text: prompt.text, role: .training)
            pair.silent = SentenceRecording(condition: .silent, samples: [], cueStart: 0, cueEnd: 1)
            data.pairs.append(pair)
        }
        try SentenceStore.save(data, to: root.appending(path: "after.json"))
        let workflow = SentenceTrainingWorkflow(storeURL: root.appending(path: "after.json"), community: small)
        workflow.readingSource = .community
        workflow.communityPassageCount = 2
        let ids = workflow.plan.flatMap { $0 }.map(\.id)
        let chosen = small.assign(counts: small.counts(recordedPromptIDs: workflow.data.mouthedPromptIDs), recordedPassages: [], count: 2)
        XCTAssertEqual(ids, chosen.flatMap { $0.prompts.map(\.promptID) })
        XCTAssertTrue(workflow.plan.allSatisfy { $0.count == 1 && $0[0].kind == .community })
        workflow.readingSource = .banks
        XCTAssertFalse(workflow.plan.flatMap { $0 }.contains { $0.kind == .community })
    }
}
