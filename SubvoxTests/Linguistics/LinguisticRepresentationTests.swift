import AVFoundation
import XCTest
@testable import Subvox

final class LinguisticRepresentationTests:XCTestCase {
    let lexicon=Lexicon(entries:["pack":["P","AE1","K"],"bag":["B","AE1","G"],"map":["M","AE1","P"],"cup":["K","AH1","P"],
        "cat":["K","AE1","T"],"bat":["B","AE1","T"],"cap":["K","AE1","P"],"mat":["M","AE1","T"],"about":["AH0","B","AW1","T"]])

    func testInventoryMapsEveryExistingARPAbetAndKeepsStressAndAttributesSeparateFromSensorClasses() throws {
        XCTAssertEqual(ArticulatoryInventory.byARPAbet.count,39)
        for phone in Phoneme.all {XCTAssertNotNil(ArticulatoryInventory.phone(arpabet:phone.symbol))}
        XCTAssertEqual(ArticulatoryInventory.phone(arpabet:"AH0")?.ipa,"ə")
        XCTAssertEqual(ArticulatoryInventory.phone(arpabet:"AH1")?.ipa,"ʌ")
        XCTAssertEqual(ArticulatoryInventory.byIPA["p"]?.gesturePrior,ArticulatoryInventory.byIPA["b"]?.gesturePrior)
        XCTAssertNotEqual(ArticulatoryInventory.byIPA["p"]?.attributes.voiced,ArticulatoryInventory.byIPA["b"]?.attributes.voiced)
        XCTAssertEqual(ArticulatoryInventory.byIPA["aɪ"]?.attributes.trajectory,["a","ɪ"])
    }
    func testEditableMixedAccentAlternativesRemainDistinctAndCoverageIsNotCertain() throws {
        var profile=PronunciationProfile()
        XCTAssertFalse(profile.alternatives("pack",lexicon:lexicon)[0].speakerConfirmed)
        try profile.set(word:"about",ipaLines:"ə b aʊ t\nə b aʊ ʔ")
        let variants=profile.alternatives("ABOUT",lexicon:lexicon)
        XCTAssertEqual(variants.count,2);XCTAssertNotEqual(variants[0].id,variants[1].id)
        let graph=LinguisticGraph(text:"about",profile:profile,lexicon:lexicon)
        XCTAssertEqual(graph.phonemeCoverage["t"],0.5);XCTAssertEqual(graph.phonemeCoverage["ʔ"],0.5)
        XCTAssertThrowsError(try profile.set(word:"about",ipaLines:"madeup"))
        let restored=try JSONDecoder().decode(PronunciationProfile.self,from:JSONEncoder().encode(profile))
        XCTAssertEqual(try profile.fingerprint(),try restored.fingerprint())
    }
    func testDiphonesIncludeCrossWordTransitionsAndUnknownWordsBreakTheChain() throws {
        let profile=PronunciationProfile(),graph=LinguisticGraph(text:"pack bag",profile:profile,lexicon:lexicon)
        XCTAssertEqual(graph.diphoneCoverage[.init(left:"k",right:"b",crossesWord:true)],1)
        XCTAssertNil(LinguisticGraph(text:"pack unknown bag",profile:profile,lexicon:lexicon).gestureSignature)
        XCTAssertFalse(LinguisticGraph(text:"pack unknown bag",profile:profile,lexicon:lexicon).diphoneCoverage.keys.contains {$0.crossesWord})
        XCTAssertEqual(LinguisticGraph(text:"pack",profile:profile,lexicon:lexicon).gestureSignature?.count,72)
        XCTAssertEqual(LinguisticGraph(text:"pack",profile:profile,lexicon:lexicon).gestureSignature,
                       LinguisticGraph(text:"bag",profile:profile,lexicon:lexicon).gestureSignature)
    }
    func provenance()->PhoneAlignmentProvenance {
        .init(method:.localAcousticAlignment,backend:"local fixture",modelSHA256:"model",pronunciationSHA256:"profile",audioSHA256:"audio",rawOutputPath:nil,note:"acoustic estimate; confidence unavailable")
    }
    func testLocalAcousticParserPreservesClockWordIdentityAndUnknownConfidenceWithoutSpreading() throws {
        let data=Data(#"{"tiers":{"words":{"entries":[[0.1,0.8,"pack"]]},"phones":{"entries":[[0,0.1,"sil"],[0.1,0.2,"P"],[0.2,0.6,"AE1"],[0.6,0.8,"K"]]}}}"#.utf8)
        let parsed=try LocalPhonemeAligner.parse(data,phoneMap:["P":"p","AE1":"æ","K":"k"],origin:100,duration:1,provenance:provenance())
        XCTAssertEqual(parsed.phones.map(\.ipa),["p","æ","k"])
        XCTAssertEqual(parsed.phones[1].start,100.2,accuracy:1e-9)
        XCTAssertEqual(parsed.phones[0].word,"pack");XCTAssertEqual(parsed.phones[0].wordIndex,0)
        XCTAssertTrue(parsed.phones.allSatisfy {$0.confidence==nil})
        XCTAssertThrowsError(try LocalPhonemeAligner.parse(data,phoneMap:["P":"p"],origin:100,duration:1,provenance:provenance()))
        XCTAssertThrowsError(try LocalPhonemeAligner.parse(data,phoneMap:["P":"p","AE1":"æ","K":"k"],origin:100,duration:0.5,provenance:provenance()))
    }
    func testMouthedAlignmentUsesSeparateCameraPathAtDifferentPacingAndKeepsProvenance() throws {
        let source=PhoneAlignment(condition:.normal,phones:[.init(ipa:"æ",start:100.2,end:100.8,confidence:0.8,backendScore:-2)],provenance:provenance())
        let spoken=(0...50).map {i in let u=Double(i)/50;return CameraPhoneAlignment.Frame(time:100+u,shape:[sin(u*3),cos(u*3),u])}
        let mouth=(0...90).map {i in let u=Double(i)/90;return CameraPhoneAlignment.Frame(time:200+u*1.8,shape:[sin(u*3),cos(u*3),u])}
        let result=try CameraPhoneAlignment.transfer(source,spoken:spoken,mouthed:mouth,profileHash:"profile")
        XCTAssertEqual(result.condition,.silent);XCTAssertEqual(result.provenance.sourceAlignmentID,source.id)
        XCTAssertEqual(result.provenance.method,.cameraWarpFromAcoustic);XCTAssertNil(result.phones[0].confidence)
        XCTAssertEqual(result.phones[0].start,200.36,accuracy:0.08)
        XCTAssertGreaterThan(result.phones[0].end-result.phones[0].start,0.9)
        XCTAssertFalse(result.reviewed)
        var gap=mouth;gap.removeSubrange(10...30)
        XCTAssertThrowsError(try CameraPhoneAlignment.transfer(source,spoken:spoken,mouthed:gap,profileHash:"profile"))
    }
    func testMFAExportsAllSpeakerVariantsAndRefusesUnknownPhoneMappings() throws {
        var profile=PronunciationProfile();try profile.set(word:"pack",ipaLines:"p æ k\np ɑː k")
        let dict=try LocalPhonemeAligner.dictionary(text:"pack",profile:profile,lexicon:lexicon,map:["P":"p","AE1":"æ","AA1":"ɑː","K":"k"])
        XCTAssertTrue(dict.contains("pack\tP AA1 K"));XCTAssertTrue(dict.contains("pack\tP AE1 K"))
        XCTAssertThrowsError(try LocalPhonemeAligner.dictionary(text:"pack",profile:profile,lexicon:lexicon,map:["P":"p"]))
    }
    func testObservedMotionOverlapNeedsMultipleContextsAndDoesNotForceUniquePhoneClasses() {
        let session=UUID()
        let examples=(0..<4).flatMap {i in ["p","b"].map {phone in
            SensorObservability.Example(ipa:phone,movement:[0.1+Double(i)*0.001,0.8,0.2],sessionID:session,sentenceGroup:"sentence\(i%2)")
        }}
        XCTAssertEqual(SensorObservability.learn(examples).count,1)
        XCTAssertTrue(SensorObservability.learn(examples)[0].interpretation.contains("Overlapping"))
        XCTAssertTrue(SensorObservability.learn(Array(examples.prefix(2))).isEmpty)
    }

    func testLocalAlignerProcessUsesExactCropClockAndRetainsHashesAndPronunciationLinks() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"phone-process-fixture-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let audio=root.appending(path:"reference.wav"),executable=root.appending(path:"mfa-fixture"),model=root.appending(path:"local-model.zip"),map=root.appending(path:"map.json")
        let format=AVAudioFormat(standardFormatWithSampleRate:16000,channels:1)!
        do {
            let file=try AVAudioFile(forWriting:audio,settings:format.settings),buffer=AVAudioPCMBuffer(pcmFormat:format,frameCapacity:32000)!
            buffer.frameLength=32000
            for i in 0..<32000 {buffer.floatChannelData![0][i]=Float(sin(Double(i)*0.03)*0.1)}
            try file.write(from:buffer)
        } // Finish the WAV header before another reader opens it.
        try Data("fixture acoustic model, not a real recogniser".utf8).write(to:model)
        try JSONEncoder().encode(["P":"p","AE1":"æ","K":"k"]).write(to:map)
        let script="""
        #!/bin/sh
        [ "$1" = "align_one" ] || exit 8
        [ -f "$5" ] || exit 9
        /bin/cat > "$6" <<'JSON'
        {"tiers":{"words":{"entries":[[0.1,0.9,"pack"]]},"phones":{"entries":[[0.1,0.2,"P"],[0.2,0.7,"AE1"],[0.7,0.9,"K"]]}}}
        JSON
        """
        try script.write(to:executable,atomically:true,encoding:.utf8)
        try FileManager.default.setAttributes([.posixPermissions:0o755],ofItemAtPath:executable.path)
        let aligned=try await LocalPhonemeAligner.align(configuration:.init(executable:executable.path,model:model.path,phoneMap:map.path),
            audio:audio,range:0.2...1.2,sessionAudioOrigin:100,text:"pack",profile:PronunciationProfile(),lexicon:lexicon,output:root)
        XCTAssertEqual(aligned.phones[0].start,100.3,accuracy:1e-8)
        XCTAssertEqual(aligned.provenance.modelSHA256,try LocalPhonemeAligner.digest(model))
        XCTAssertEqual(aligned.provenance.phoneMapSHA256,try LocalPhonemeAligner.digest(map))
        XCTAssertEqual(aligned.provenance.audioSHA256?.count,64)
        XCTAssertEqual(aligned.phones[0].pronunciationVariantIDs,["dictionary:pack"])
        XCTAssertTrue(aligned.phones.allSatisfy {$0.confidence==nil})
    }
}
