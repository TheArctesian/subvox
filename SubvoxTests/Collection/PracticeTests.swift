import XCTest
@testable import Subvox

final class PracticeTests: XCTestCase {
    @MainActor func testPersonalScriptReopensAndKeepsProgressBoundToItsTextRevision() throws {
        let suite = "personal-script-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let scripts = PersonalScripts(defaults: defaults)
        let script = try XCTUnwrap(scripts.save(id: nil, title: "Mon entretien", text: "Bonjour. Je m’appelle Claire.", language: .fr))
        XCTAssertEqual(script.sections, ["Bonjour.", "Je m’appelle Claire."])
        XCTAssertEqual(script.memoryPiece.parts(language: .fr).map { $0.map(\.text) }, script.sections.map { [$0] })

        let reopened = PersonalScripts(defaults: defaults)
        XCTAssertEqual(reopened.items, [script])
        let renamed = try XCTUnwrap(reopened.save(id: script.id, title: "Mon introduction", text: script.text, language: .fr))
        XCTAssertEqual(renamed.pieceID, script.pieceID)
        let revised = try XCTUnwrap(reopened.save(id: script.id, title: renamed.title, text: "Bonsoir.", language: .fr))
        XCTAssertEqual(revised.id, script.id)
        XCTAssertNotEqual(revised.pieceID, script.pieceID)
        XCTAssertEqual(PersonalScripts(defaults: defaults).items, [revised])
    }

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
        XCTAssertEqual(etre?.phrases.first?.ipa, ["ʒə", "sɥi", "fatiɡe"])
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

    func testLanguageDrillRoundsKeepEveryRegionalPhraseInOrder() throws {
        let corpus = PracticeCorpus.bundled
        for language in AppLanguage.allCases {
            let drills = corpus.drills.filter { $0.language == language.corpus && language.shows(region: $0.region) }
            for drill in drills {
                let rounds = LanguageDrillRounds.phrases(in: drill, language: language)
                XCTAssertTrue(rounds.allSatisfy { (1...3).contains($0.count) }, drill.id)
                XCTAssertEqual(rounds.flatMap { $0 }, drill.phrases.filter { language.shows(region: $0.region) }, drill.id)
            }
        }
        let cafe = try XCTUnwrap(corpus.drills.first { $0.id == "fr-cafe" })
        XCTAssertEqual(LanguageDrillRounds.phrases(in: cafe, language: .fr).map(\.count), [3, 3, 2])
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

    func testSpeechAndPoetryKeepDistinctPracticeKinds() throws {
        let pieces = PracticeCorpus.bundled.pieces
        let speech = try XCTUnwrap(pieces.first { $0.id == "gettysburg" })
        let poem = try XCTUnwrap(pieces.first { $0.id == "invictus" })
        XCTAssertEqual(speech.practiceKind, .speech)
        XCTAssertEqual(poem.practiceKind, .poetry)

        var data = SentenceTrainingData()
        let prompt = SentencePrompt(poem.lines[0], kind: poem.practiceKind, language: .en)
        data.addPractice([prompt])
        XCTAssertEqual(data.roles[prompt.id], .training)
    }

    func testGettysburgOpeningHasTwoSelectableParts() throws {
        let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == "gettysburg" })
        let opening = Array(piece.lines.prefix(2))
        let parts = PracticeCorpus.takes(piece.lines, language: .en)
        XCTAssertEqual(Array(parts.prefix(2)), opening.map { [$0] })
    }

    func testLibraryPilotCatalogMatchesTheExactSectionsRecordedByTheApp() throws {
        let catalog = LibraryPilotCatalog.bundled
        for pieceID in ["gettysburg", "invictus"] {
            let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == pieceID })
            let groups = piece.parts(language: .en).map { part in
                part.map { SentencePrompt($0, kind: piece.practiceKind, language: .en) }
            }
            let sections = catalog.sections(pieceID: pieceID, language: "en")
            XCTAssertEqual(sections.count, pieceID == "invictus" ? 4 : 3)
            XCTAssertEqual(sections.map(\.lines), groups.map { $0.map(\.text) })
            XCTAssertTrue(catalog.allows(pieceID: pieceID, language: "en", groups: groups))
            XCTAssertTrue(catalog.allows(pieceID: pieceID, language: "en", groups: Array(groups.dropFirst())))
            XCTAssertFalse(catalog.allows(pieceID: pieceID, language: "en", groups: [groups[0], groups[2]]))
            XCTAssertFalse(catalog.allows(pieceID: pieceID, language: "fr", groups: groups))

            var changed = groups
            changed[0][0].text += "!"
            XCTAssertFalse(catalog.allows(pieceID: pieceID, language: "en", groups: changed))

            let group = groups[0]
            var pair = SentencePair(visitID: UUID(), promptIDs: group.map(\.id),
                                    text: group.map(\.text).joined(separator: " "), role: .training)
            pair.practiceMode = piece.practiceKind
            pair.literaturePieceID = pieceID
            pair.language = "en"
            XCTAssertEqual(catalog.prompt(for: pair)?.promptID, sections.first?.promptID)
            pair.promptIDs.append("extra")
            XCTAssertNil(catalog.prompt(for: pair))
        }
        XCTAssertNotEqual(
            LibraryPilotCatalog.identity(pieceID: "example", language: "zh-Hans", text: "你好"),
            LibraryPilotCatalog.identity(pieceID: "example", language: "zh-Hans", text: "再见")
        )
    }

    func testBundledGageAccountIncludesTheWholeSpeechInReadableParts() throws {
        let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == "aint-i-a-woman" })
        let parts = piece.parts(language: .en)
        XCTAssertEqual(piece.year, 1863)
        XCTAssertEqual(piece.lines.count, 28)
        XCTAssertEqual(parts.flatMap { $0 }, piece.lines)
        XCTAssertTrue(parts.allSatisfy { $0.reduce(0) { $0 + $1.text.split(whereSeparator: \.isWhitespace).count } <= 24 })
    }

    func testBundledPoemsPreserveExplicitPartBoundaries() throws {
        for (id, lengths) in [("invictus", [4, 4, 4, 4]), ("ozymandias", [4, 4, 3, 3]),
                              ("a-fairy-song", [4, 4, 4, 2]), ("holy-sonnet-10", [4, 4, 4, 2]),
                              ("tyger", [4, 4, 4, 4, 4, 4]), ("sonnet-18", [4, 4, 4, 2])] {
            let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == id })
            let parts = piece.parts(language: .en)
            XCTAssertEqual(parts.map(\.count), lengths)
            XCTAssertEqual(parts.flatMap { $0 }, piece.lines)
        }
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

    func testStreakCountsConsecutiveDaysWithAMouthedTake() {
        let calendar = Calendar(identifier: .gregorian)
        let now = calendar.date(from: DateComponents(year: 2026, month: 11, day: 8, hour: 9))!
        func pair(daysAgo: Int, mouthed: Bool = true) -> SentencePair {
            var p = SentencePair(visitID: UUID(), promptIDs: [], text: "x", role: .training)
            p.createdAt = calendar.date(byAdding: .day, value: -daysAgo, to: now)!
            if mouthed { p.silent = SentenceRecording(condition: .silent, samples: [], cueStart: 0, cueEnd: 1) }
            return p
        }
        let progress = PracticeProgress(pairs: [pair(daysAgo: 0), pair(daysAgo: 0), pair(daysAgo: 1), pair(daysAgo: 2), pair(daysAgo: 4),
                                                pair(daysAgo: 3, mouthed: false)], calendar: calendar, now: now)
        XCTAssertEqual(progress.today, 2)
        XCTAssertEqual(progress.streak, 3, "A day with only a spoken half doesn't keep the streak")
        XCTAssertEqual(PracticeProgress(pairs: [pair(daysAgo: 1)], calendar: calendar, now: now).streak, 1, "Yesterday still counts until today ends")

        let target = VoicePracticeTarget(mode: TrainMode.speech.rawValue, language: .en, itemID: "speech", text: "Hello.")
        let voiceRuns = [0, 1].map { daysAgo in
            VoicePracticeRun(id: UUID(), target: target, createdAt: calendar.date(byAdding: .day, value: -daysAgo, to: now)!,
                             duration: 10)
        }
        let combined = PracticeProgress(pairs: [pair(daysAgo: 0), pair(daysAgo: 2)], voiceRuns: voiceRuns,
                                        calendar: calendar, now: now)
        XCTAssertEqual(combined.today, 2)
        XCTAssertEqual(combined.streak, 3)
    }

    func testTraditionalAndSimplifiedMatchForSpeechRecognition() {
        XCTAssertEqual(Lexicon.words(in: "我們去銀行").map(Lexicon.matchKey), Lexicon.words(in: "我们去银行").map(Lexicon.matchKey))
        XCTAssertEqual(Lexicon.matchKey("hello"), "hello")
    }

    func testVerdictFollowsTheTrainingAdmissionRule() {
        var pair = SentencePair(visitID: UUID(), promptIDs: [], text: "x", role: .training)
        pair.transcript = LocalWhisperTranscript(text: "x", words: [])
        pair.transcriptMatchRate = 0.82
        XCTAssertFalse(PracticeFeedback.verdict(pair).kept, "82% passes processing but not admission (85%)")
        pair.review = .excluded
        XCTAssertFalse(PracticeFeedback.verdict(pair).kept)
    }

    func testForcedAlignmentKeepsKnownTextAt55PercentWhisperAgreement() {
        let pair = alignmentPair(coverage: 0.8)
        XCTAssertTrue(pair.usesAlignedWords)
        XCTAssertTrue(PracticeFeedback.verdict(pair).kept)
    }

    func testLowWhisperAndForcedCoverageRejectThePair() {
        XCTAssertFalse(alignmentPair(coverage: 0.79).usesAlignedWords)
    }

    func testForcedCoverageCannotRescueFreePractice() {
        var pair = alignmentPair(coverage: 1)
        pair.practiceMode = .freeText
        XCTAssertFalse(pair.usesAlignedWords)
    }

    private func alignmentPair(coverage: Double) -> SentencePair {
        let samples = [MotionSample(t: 0, accel: (0, 0, 0), rotation: (0, 0, 0))]
        var pair = SentencePair(visitID: UUID(), promptIDs: [], text: "a bee", role: .training)
        pair.spoken = .init(condition: .normal, samples: samples, cueStart: 0, cueEnd: 1)
        pair.silent = .init(condition: .silent, samples: samples, cueStart: 0, cueEnd: 1)
        pair.transcript = LocalWhisperTranscript(text: "a b", words: [])
        pair.transcriptMatchRate = 0.55
        pair.timedWordCoverage = coverage
        pair.wordTimingSource = .forcedAlignment
        pair.alignment = .init(normalizedCost: 0, durationRatio: 1, path: [],
                               words: [.init(word: "bee", voicedStart: 0, voicedEnd: 1, silentStart: 0, silentEnd: 1)],
                               warnings: [])
        return pair
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
