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
        for language in AppLanguage.allCases {
            XCTAssertFalse(corpus.pieces.filter { $0.language == language.corpus }.isEmpty, "speech/poetry in \(language)")
            XCTAssertFalse(corpus.drills.filter { $0.language == language.corpus }.isEmpty, "drills in \(language)")
            XCTAssertFalse(corpus.minimalPairs.filter { $0.language == language.corpus }.isEmpty, "minimal pairs in \(language)")
        }
        for language in ["en", "fr", "es", "zh"] {
            let sounds = corpus.sounds[language] ?? []
            XCTAssertGreaterThan(sounds.count, 25, "sound guide for \(language)")
            XCTAssertTrue(sounds.allSatisfy { !$0.examples.isEmpty && !$0.say.isEmpty }, "every sound has examples and a demo in \(language)")
        }
        XCTAssertEqual(corpus.sounds["zh"]?.filter { $0.group == "Tones" }.count, 5)
        let poem = corpus.pieces.first { $0.id == "jing-ye-si" }
        XCTAssertEqual(poem?.lines[2].text(.zhHant), "舉頭望明月，")
        XCTAssertEqual(poem?.lines[2].text(.zhHans), "举头望明月，")
        XCTAssertFalse(corpus.pieces.contains { $0.author.contains("Churchill") || $0.author.contains("King") })
        let etre = corpus.drills.first { $0.id == "fr-etre-present" }
        XCTAssertEqual(etre?.phrases.first?.ipa, ["ʒə", "sɥi", "fʁɑ̃sɛ"])
        XCTAssertTrue(corpus.drills.filter { $0.language == "zh" }.allSatisfy { d in d.phrases.allSatisfy { $0.hant != nil && $0.pinyin != nil } })
        XCTAssertGreaterThan(corpus.drills.count, 100)
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
        let pairs = PracticeCorpus.MinimalPairs(language: "fr", contrast: "y u", sets: [
            [.init(text: "rue", ipa: ["ʁy"]), .init(text: "roue", ipa: ["ʁu"])],
            [.init(text: "tu", ipa: ["t y"]), .init(text: "tout", ipa: ["t u"])],
            [.init(text: "vu", ipa: ["v y"]), .init(text: "vous", ipa: ["v u"])]])
        var generator = SystemRandomNumberGenerator()
        let round = PracticeCorpus.round(pairs, count: 2, using: &generator)
        XCTAssertEqual(round.count, 4)
        let texts = Set(round.map(\.text))
        XCTAssertTrue([["rue", "roue"], ["tu", "tout"], ["vu", "vous"]].filter { texts.isSuperset(of: $0) }.count == 2, "Both members of each set")
    }

    func testLongPiecesAreSplitIntoReadableTakes() {
        let corpus = PracticeCorpus.bundled
        for piece in corpus.pieces {
            let language: AppLanguage = piece.language == "zh" ? .zhHans : AppLanguage(rawValue: piece.language) ?? .en
            let takes = PracticeCorpus.takes(piece.lines, language: language)
            XCTAssertEqual(takes.flatMap { $0 }, piece.lines, "\(piece.id): every line once, in order")
            XCTAssertTrue(takes.allSatisfy { $0.count == 1 || $0.map { $0.text.split(separator: " ").count }.reduce(0, +) <= 30 }, piece.id)
        }
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
