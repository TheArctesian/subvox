import XCTest
@testable import Subvox

final class PracticeTests: XCTestCase {
    func testWordsKeepEnglishIdentitiesAndSplitAccentsAndHan() {
        XCTAssertEqual(Lexicon.words(in: "Don't stop, Mr. Smith!"), ["don't", "stop", "mr", "smith"])
        XCTAssertEqual(Lexicon.words(in: "Vous êtes là"), ["vous", "êtes", "là"])
        XCTAssertEqual(Lexicon.words(in: "我是学生。"), ["我", "是", "学", "生"])
        XCTAssertEqual(SentencePrompt.identity("The cat sat."), SentencePrompt.identity("the CAT sat"))
    }

    func testBundledPracticeCorpusCoversEveryLanguageAndTab() {
        let corpus = PracticeCorpus.bundled
        for language in PracticeLanguage.allCases {
            XCTAssertFalse(corpus.pieces.filter { $0.language == language }.isEmpty, "speech/poetry in \(language)")
            XCTAssertFalse(corpus.drills.filter { $0.language == language }.isEmpty, "drills in \(language)")
            XCTAssertFalse(corpus.minimalPairs.filter { $0.language == language }.isEmpty, "minimal pairs in \(language)")
        }
        XCTAssertFalse(corpus.pieces.contains { $0.author.contains("Churchill") || $0.author.contains("King") })
        let etre = corpus.drills.first { $0.id == "etre" }
        XCTAssertEqual(etre?.phrases.first?.ipa, ["ʒ ə", "s ɥ i"])
    }

    func testPracticePromptsAreTrainingDataAndKeepTheirLanguage() {
        var data = SentenceTrainingData()
        data.addPractice([SentencePrompt(text: "je suis", kind: .drill, language: "fr"),
                          SentencePrompt(text: "Some sentence", kind: .sentence)])
        XCTAssertEqual(data.prompts.map(\.text), ["je suis"], "Only practice kinds are added this way")
        XCTAssertEqual(data.roles[data.prompts[0].id], .training)
        XCTAssertEqual(data.prompts[0].language, "fr")
    }

    func testFreeTextSplitsIntoSentencesAndRoundsUseWholeSets() {
        XCTAssertEqual(FreePracticeView.sentences("Hello there. How are you?\nFine!"), ["Hello there.", "How are you?", "Fine!"])
        let pairs = PracticeCorpus.MinimalPairs(language: .fr, contrast: "y u", sets: [
            [.init(text: "rue", ipa: ["ʁ y"]), .init(text: "roue", ipa: ["ʁ u"])],
            [.init(text: "tu", ipa: ["t y"]), .init(text: "tout", ipa: ["t u"])],
            [.init(text: "vu", ipa: ["v y"]), .init(text: "vous", ipa: ["v u"])]])
        var generator = SystemRandomNumberGenerator()
        let round = PracticeCorpus.round(pairs, count: 2, using: &generator)
        XCTAssertEqual(round.count, 4)
        let texts = Set(round.map(\.text))
        XCTAssertTrue([["rue", "roue"], ["tu", "tout"], ["vu", "vous"]].filter { texts.isSuperset(of: $0) }.count == 2, "Both members of each set")
    }

    func testPhoneCheckDecodesThePythonOutput() throws {
        let json = #"{"language": "fr", "expected_phones": ["t", "y"], "produced_phones": ["t", "u"], "source": "whisper-words", "alignment": [{"op": "match", "expected": "t", "produced": "t"}, {"op": "sub", "expected": "y", "produced": "u"}], "word_agreement": 0.0, "phone_error_rate": 0.5, "trainable": false, "timing": null}"#
        let result = try JSONDecoder().decode(PhoneCheckResult.self, from: Data(json.utf8))
        XCTAssertEqual(result.mismatches.map(\.expected), ["y"])
        XCTAssertFalse(result.trainable)
    }

    func testOldPairsWithoutPracticeFieldsStillDecode() throws {
        let prompt = try JSONDecoder().decode(SentencePrompt.self, from: Data(#"{"text":"Go and ask her.","kind":"calibration"}"#.utf8))
        XCTAssertNil(prompt.language)
    }
}
