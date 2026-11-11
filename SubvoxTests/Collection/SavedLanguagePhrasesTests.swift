import Foundation
import XCTest
@testable import Subvox

final class SavedLanguagePhrasesTests: XCTestCase {
    @MainActor
    func testSaveAndRemoveKeepsExactPhraseAndLocaleAcrossRelaunch() throws {
        let suite = "saved-language-tests-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let phrase = PracticeCorpus.Line(text: "Café", ipa: [], gloss: "Coffee")
        let common = PracticeCorpus.Drill(id: "cafe", language: "es", title: "Order coffee", phrases: [phrase])
        let second = PracticeCorpus.Drill(id: "menu", language: "es", title: "Read the menu", phrases: [phrase])
        let cafe = LanguagePhraseID(phrase, drillID: common.id, language: .es)
        let menu = LanguagePhraseID(phrase, drillID: second.id, language: .es)
        let mexico = LanguagePhraseID(phrase, drillID: common.id, language: .esMX)
        let store = SavedLanguagePhrases(defaults: defaults)
        [cafe, menu, mexico].forEach(store.toggle)

        let reopened = SavedLanguagePhrases(defaults: defaults)
        let drills = [common, second]
        XCTAssertEqual(reopened.cards(in: drills, language: .es).map(\.id), [cafe, menu])
        let card = try XCTUnwrap(reopened.cards(in: drills, language: .esMX).first)
        XCTAssertEqual(card.id, mexico)
        XCTAssertEqual(card.translationCue(in: "en"), "Coffee")
        let revised = PracticeCorpus.Line(text: "Cafe\u{301}", ipa: [])
        XCTAssertNotEqual(cafe, LanguagePhraseID(revised, drillID: common.id, language: .es))
        var revisedDrill = common
        revisedDrill.phrases = [revised]
        XCTAssertEqual(reopened.cards(in: [revisedDrill, second], language: .es).map(\.id), [menu])
        reopened.remove(cafe)
        let afterRemoval = SavedLanguagePhrases(defaults: defaults)
        XCTAssertEqual(afterRemoval.cards(in: drills, language: .es).map(\.id), [menu])
        XCTAssertEqual(afterRemoval.cards(in: drills, language: .esMX).map(\.id), [mexico])
    }

    @MainActor
    func testReviewRoundPrioritizesUnseenPhrasesAndPersistsProgress() throws {
        let suite = "language-review-tests-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let lines = ["Je voudrais un café.", "Le menu, s’il vous plaît.", "L’addition, merci.", "C’est combien ?"]
            .map { PracticeCorpus.Line(text: $0, ipa: []) }
        let drill = PracticeCorpus.Drill(id: "cafe", language: "fr", title: "Au café", phrases: lines)
        let ids = lines.map { LanguagePhraseID($0, drillID: drill.id, language: .fr) }
        let store = SavedLanguagePhrases(defaults: defaults)
        ids.forEach(store.toggle)
        let firstRound = store.nextRound(in: [drill], language: .fr)
        XCTAssertEqual(firstRound.map(\.id), Array(ids.prefix(3)))
        for (index, card) in firstRound.enumerated() {
            store.markReviewed(card.id, at: Date(timeIntervalSince1970: Double(index)))
        }

        let reopened = SavedLanguagePhrases(defaults: defaults)
        XCTAssertEqual(reopened.nextRound(in: [drill], language: .fr).map(\.id), [ids[3], ids[0], ids[1]])
        XCTAssertTrue(reopened.nextRound(in: [drill], language: .en).isEmpty)
    }
}
