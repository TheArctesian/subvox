import XCTest
@testable import Subvox

final class SentenceCollectionTests:XCTestCase {
    @MainActor private func wait(_ phase:SentenceTrainingWorkflow.Phase,_ workflow:SentenceTrainingWorkflow) async throws {
        for _ in 0..<400 {if workflow.phase==phase {return};try await Task.sleep(for:.milliseconds(5))}
        XCTFail("Expected \(phase), got \(workflow.phase)")
    }
    @MainActor func testGuidedSentenceWaitsAtReadyAndEndAndRedoResetsFocus() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"sentence-guide-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let workflow=SentenceTrainingWorkflow(storeURL:root.appending(path:"sentences.json"));workflow.guidedRecording=true
        let model=AppModel(sentenceTraining:workflow)
        model.openPractice(.reading)
        var completed:[SentenceRecording]=[],discarded=0
        let task=Task {try await workflow.collectTake(condition:.normal,maximumDuration:5,baselineSeconds:0.04,recoverySeconds:0,text:"Pack my blue bag.",
            onCompleted:{completed.append($0)},onRedo:{_ in discarded+=1;completed=[]})}
        defer {task.cancel()}
        try await wait(.ready,workflow)
        XCTAssertNil(workflow.activeWordIndex)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode:124,isRepeat:true));XCTAssertEqual(workflow.phase,.ready)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode:124,isRepeat:false))
        try await wait(.speak,workflow)
        try await Task.sleep(for:.milliseconds(450))
        XCTAssertEqual(workflow.activeWordIndex,0)
        XCTAssertTrue(workflow.advance());try await wait(.ended,workflow)
        try await Task.sleep(for:.milliseconds(80));XCTAssertEqual(workflow.phase,.ended)
        XCTAssertEqual(completed.count,1);XCTAssertTrue(workflow.canRedoTake)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode:123,isRepeat:false))
        try await wait(.ready,workflow);XCTAssertEqual(discarded,1);XCTAssertTrue(completed.isEmpty);XCTAssertNil(workflow.activeWordIndex)
        XCTAssertTrue(workflow.advance());try await wait(.speak,workflow)
        XCTAssertNil(workflow.activeWordIndex)
        XCTAssertTrue(workflow.advance());try await wait(.ended,workflow)
        XCTAssertTrue(workflow.advance());try await task.value
        XCTAssertEqual(completed.count,1);XCTAssertEqual(completed[0].wordFocusGuide,.comfortable)
    }
    @MainActor func testSourceSwitchDiscardsGuidedAttemptBeforeSavingLabels() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"sentence-source-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let workflow=SentenceTrainingWorkflow(storeURL:root.appending(path:"sentences.json"));workflow.guidedRecording=true
        var saved=0,discarded=0
        let task=Task {try await workflow.collectTake(condition:.silent,maximumDuration:5,baselineSeconds:0.02,recoverySeconds:0,
            onCompleted:{_ in saved+=1},onRedo:{_ in discarded+=1})}
        defer {task.cancel()}
        try await wait(.ready,workflow);workflow.advance();try await wait(.mouth,workflow)
        var sample=MotionSample(t:ProcessInfo.processInfo.systemUptime);sample.sensorLocation=1;workflow.ingest(sample)
        sample.t+=0.02;sample.sensorLocation=2;workflow.ingest(sample)
        try await wait(.ready,workflow)
        XCTAssertEqual(saved,0);XCTAssertEqual(discarded,1);XCTAssertNotNil(workflow.error)
    }
    func testExpandedBankKeepsExistingReservationAndPlansFreshSingleRoleGroups() throws {
        var data=SentenceTrainingData(),rng=SplitMix64(seed:82)
        data.add(SentencePrompt.bundled,using:&rng);let original=data.roles
        XCTAssertEqual(SentencePrompt.diphoneBank.count,235)
        data.add(SentencePrompt.diphoneBank,using:&rng)
        for (id,role) in original {XCTAssertEqual(data.roles[id],role)}
        XCTAssertEqual(data.roles.values.filter {$0 == .finalTest}.count,(data.prompts.count+3)/4)
        let planned=data.variedSentences(role:.training,limit:6);XCTAssertEqual(planned.count,6)
        let first=planned[0]
        var partial=SentencePair(visitID:UUID(),promptIDs:[first.id],text:first.text,role:.training)
        partial.spoken=SentenceRecording(condition:.normal,samples:[],cueStart:100,cueEnd:101);data.pairs=[partial]
        let next=data.variedSentences(role:.training,limit:6,excluding:[planned[1].id])
        XCTAssertFalse(next.contains {$0.id==first.id || $0.id==planned[1].id})
        XCTAssertTrue(next.allSatisfy {data.roles[$0.id] == .training})
        XCTAssertTrue(data.variedSentences(role:.validation,limit:6).allSatisfy {data.roles[$0.id] == .validation})
        XCTAssertTrue(data.variedSentences(role:.finalTest,limit:6).isEmpty)
    }
    @MainActor func testTranscriptionBacklogIncludesSpokenPartialAndExcludesSkippedAndFinalPairs() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"sentence-backlog-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var pair=SentencePair(visitID:UUID(),promptIDs:[],text:"Pack my blue bag.",role:.training)
        pair.spoken=SentenceRecording(condition:.normal,samples:[],cueStart:100,cueEnd:101)
        var excluded=pair;excluded.id=UUID();excluded.review = .excluded
        var final=pair;final.id=UUID();final.role = .finalTest
        var done=pair;done.id=UUID();done.transcript = .init(text:done.text,words:[]);done.transcriptMatchRate=1
        let data=SentenceTrainingData(prompts:[.init(text:pair.text)],pairs:[pair,excluded,final,done])
        try SentenceStore.save(data,to:root.appending(path:"sentences.json"))
        let workflow=SentenceTrainingWorkflow(storeURL:root.appending(path:"sentences.json"))
        XCTAssertEqual(workflow.pendingTranscriptionIDs,[pair.id])
    }
}
