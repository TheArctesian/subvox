import XCTest
@testable import Subvox

final class MemorizationReviewTests: XCTestCase {
    func testLegacyMarksMigrateAndRatedSectionsReturnWhenDue() throws {
        let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == "invictus" })
        let sections = MemorizationSection.sections(of: piece, language: .en)
        XCTAssertEqual(sections.count, 4)
        let today = Date(timeIntervalSinceReferenceDate: 800_000_000)
        let tomorrow = today.addingTimeInterval(86_400)
        var restored = MemorizationReview(saved: try JSONEncoder().encode(sections.map(\.id)))
        let first = restored.round(in: [piece], language: .en, now: today)
        XCTAssertEqual(first.map(\.id), Array(sections.prefix(3)).map(\.id))
        restored.complete(first[0].id, keepPractising: true, at: today)
        restored.complete(first[1].id, keepPractising: false, at: today)
        restored.complete(first[2].id, keepPractising: true, at: today)

        let next = MemorizationReview(saved: restored.saved)
        XCTAssertEqual(next.round(in: [piece], language: .en, now: today).map(\.id), [sections[3].id])
        XCTAssertEqual(next.round(in: [piece], language: .en, now: tomorrow).map(\.id), [sections[3].id, sections[0].id, sections[1].id])
        XCTAssertTrue(next.isMarked(sections[1].id))
    }

    func testReviewKeepsScriptsSeparateByLocaleAndExactSectionText() throws {
        let script = PersonalScript(id: UUID(), revision: UUID(), title: "My story", text: "First line. Second line.", language: .es)
        let piece = script.memoryPiece
        let sections = MemorizationSection.sections(of: piece, language: .es)
        let first = try XCTUnwrap(sections.first)
        var review = MemorizationReview()
        review.toggle(first.id)

        let restored = MemorizationReview(saved: review.saved)
        XCTAssertEqual(restored.round(in: [piece], language: .es).map(\.id), [first.id])
        XCTAssertTrue(restored.round(in: [piece], language: .esMX).isEmpty)
        XCTAssertTrue(restored.round(in: [], language: .es).isEmpty)

        var edited = piece
        edited.lines[0].text = "First line!"
        XCTAssertTrue(restored.round(in: [edited], language: .es).isEmpty)
    }
}
