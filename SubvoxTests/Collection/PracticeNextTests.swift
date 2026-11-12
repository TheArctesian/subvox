import Foundation
import XCTest
@testable import Subvox

final class PracticeNextTests: XCTestCase {
    private let speech = PracticeCorpus.Piece(id: "test-speech", language: "en", kind: "speech", title: "Test speech",
                                              author: "", lines: [.init(text: "Here is my speech.", ipa: [])])
    private let frenchDrill = PracticeCorpus.Drill(id: "test-cafe", language: "fr", title: "Au café",
                                                   phrases: [.init(text: "Un café, s’il vous plaît.", ipa: [])])

    @MainActor func testHomeOffersAllFourModesWithoutPriorPractice() throws {
        try withEmptyBookmarks { saved in
            let deleted = VoicePracticeTarget(mode: TrainMode.speech.rawValue, language: .en, itemID: "deleted", text: "Old.")
            let choices = PracticeNext.suggestions(language: .en, practiceLanguages: [.en], drills: [], pieces: [], pins: [], scripts: [],
                                                   savedPhrases: saved, memorization: MemorizationReview(), recentRuns: [deleted],
                                                   selection: { _, _ in "deleted" })
            XCTAssertEqual(choices.map(\.id), [.language, .speech, .memorization, .free])
            XCTAssertEqual(choices.map(\.destination), choices.map { .practice($0.id) })
        }
    }

    @MainActor func testSavedContextOpensReviewInTheRightLanguageAndReopensMaterial() throws {
        try withEmptyBookmarks { saved in
            let phrase = try XCTUnwrap(frenchDrill.phrases.first)
            saved.toggle(LanguagePhraseID(phrase, drillID: frenchDrill.id, language: .fr))
            var marks = MemorizationReview()
            marks.toggle(try XCTUnwrap(MemorizationSection.sections(of: speech, language: .en).first).id)
            let pin = LiteraturePin(pieceID: speech.id, language: .en, mode: .speech)
            let script = PersonalScript(id: UUID(), revision: UUID(), title: "My script", text: "One sentence.", language: .en)
            let choices = PracticeNext.suggestions(language: .en, practiceLanguages: [.en, .fr], drills: [frenchDrill],
                                                   pieces: [speech], pins: [pin], scripts: [script], savedPhrases: saved,
                                                   memorization: marks, selection: { _, _ in nil })
            XCTAssertEqual(choices.map(\.destination), [
                .review(.language, .fr), .literature(pin), .review(.memorization, .en), .script(script)
            ])
        }
    }

    @MainActor func testRecentRunsReopenValidPracticeContextsWithoutReviewMarks() throws {
        try withEmptyBookmarks { saved in
            let script = PersonalScript(id: UUID(), revision: UUID(), title: "A script", text: "New text.", language: .fr)
            let other = PersonalScript(id: UUID(), revision: UUID(), title: "Another script", text: "Other text.", language: .en)
            let runs = [
                VoicePracticeTarget(mode: TrainMode.language.rawValue, language: .fr, itemID: "phrase/\(frenchDrill.id)", text: "Un café."),
                VoicePracticeTarget(mode: TrainMode.speech.rawValue, language: .en, itemID: speech.id, text: "Here is my speech."),
                VoicePracticeTarget(mode: TrainMode.memorization.rawValue, language: .fr, itemID: script.pieceID, text: script.text),
                VoicePracticeTarget(mode: TrainMode.free.rawValue, language: .fr, itemID: script.pieceID, text: script.text)
            ]
            let choices = PracticeNext.suggestions(language: .en, practiceLanguages: [.en, .fr], drills: [frenchDrill],
                                                   pieces: [speech], pins: [], scripts: [other, script], savedPhrases: saved,
                                                   memorization: MemorizationReview(), recentRuns: runs, selection: { _, _ in nil })
            XCTAssertEqual(choices.map(\.destination), [
                .drill(id: frenchDrill.id, language: .fr),
                .literature(LiteraturePin(pieceID: speech.id, language: .en, mode: .speech)),
                .memorize(pieceID: script.pieceID, language: .fr),
                .script(script)
            ])
        }
    }

    @MainActor func testHomeFeaturesInvictusUntilSomethingIsRecordedOrPinned() throws {
        try withEmptyBookmarks { saved in
            let invictus = PracticeCorpus.Piece(id: "invictus", language: "en", kind: "poem", title: "Invictus",
                                                author: "", lines: [.init(text: "Out of the night that covers me,", ipa: [])])
            @MainActor func speechChoice(runs: [VoicePracticeTarget]) -> PracticeSuggestion.Destination? {
                PracticeNext.suggestions(language: .en, practiceLanguages: [.en], drills: [], pieces: [speech, invictus], pins: [],
                                         scripts: [], savedPhrases: saved, memorization: MemorizationReview(), recentRuns: runs,
                                         selection: { page, _ in page == "speech" ? self.speech.id : nil })
                    .first { $0.id == .speech }?.destination
            }
            // A page the learner only browsed doesn't outrank the featured poem.
            XCTAssertEqual(speechChoice(runs: []), .literature(LiteraturePin(pieceID: "invictus", language: .en, mode: .poetry)))
            // Something they recorded does.
            let run = VoicePracticeTarget(mode: TrainMode.speech.rawValue, language: .en, itemID: speech.id, text: "Here is my speech.")
            XCTAssertEqual(speechChoice(runs: [run]), .literature(LiteraturePin(pieceID: speech.id, language: .en, mode: .speech)))
        }
    }

    @MainActor private func withEmptyBookmarks(_ body: (SavedLanguagePhrases) throws -> Void) throws {
        let suite = "practice-next-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        try body(SavedLanguagePhrases(defaults: defaults))
    }
}
