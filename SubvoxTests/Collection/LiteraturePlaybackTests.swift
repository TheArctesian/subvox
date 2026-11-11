import XCTest
@testable import Subvox

final class LiteraturePlaybackTests: XCTestCase {
    @MainActor
    func testSpeechPlaybackUsesLatestAvailableTakeForTheSelectedPiece() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "speech-playback-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let audioURL = root.appending(path: "audio.caf")
        try Data().write(to: audioURL)

        let piece = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.practiceKind == .speech && $0.language == "en" })
        let section = try XCTUnwrap(piece.parts(language: .en).first)
        let text = section.map { $0.text(.en) }
        var saved = SentencePair(
            visitID: UUID(), promptIDs: text.map(SentencePrompt.identity), text: text.joined(separator: " "),
            role: .training, directory: root.path, sessionStartUptime: 20, audioOffset: 0.5
        )
        saved.practiceMode = .speech
        saved.language = "en"
        saved.literaturePieceID = piece.id
        saved.spoken = SentenceRecording(condition: .normal, samples: [], cueStart: 22, cueEnd: 23)

        var newerSilentOnly = saved
        newerSilentOnly.id = UUID()
        newerSilentOnly.spoken = nil
        newerSilentOnly.silent = SentenceRecording(condition: .silent, samples: [], cueStart: 24, cueEnd: 25)

        var newerOtherPiece = saved
        newerOtherPiece.id = UUID()
        newerOtherPiece.literaturePieceID = "different-piece-with-the-same-section"

        var newerOtherLocale = saved
        newerOtherLocale.id = UUID()
        newerOtherLocale.language = "fr"

        var retry = saved
        retry.id = UUID()
        retry.sessionStartUptime = 50
        retry.spoken = SentenceRecording(condition: .normal, samples: [], cueStart: 55, cueEnd: 56)

        let take = try XCTUnwrap(SpokenTakePlayer.latestTake(
            for: section, mode: .speech, language: .en, pieceID: piece.id,
            in: [saved, newerSilentOnly, newerOtherPiece, newerOtherLocale]
        ))
        XCTAssertEqual(take.id, saved.id)
        XCTAssertEqual(take.clip.url, audioURL)
        XCTAssertEqual(take.clip.range, 1.5...2.5)

        let comparison = SpokenTakePlayer.recentTakes(
            for: section, mode: .speech, language: .en, pieceID: piece.id,
            in: [saved, newerSilentOnly, retry, newerOtherPiece, newerOtherLocale]
        )
        XCTAssertEqual(comparison.map(\.id), [retry.id, saved.id])
        XCTAssertEqual(comparison.map { $0.clip.range }, [4.5...5.5, 1.5...2.5])

        saved.literaturePieceID = nil
        XCTAssertEqual(SpokenTakePlayer.latestTake(
            for: section, mode: .speech, language: .en, pieceID: piece.id, in: [saved]
        )?.id, saved.id)
        XCTAssertNil(SpokenTakePlayer.latestTake(
            for: section, mode: .poetry, language: .en, pieceID: piece.id, in: [saved]
        ))
    }

    @MainActor
    func testPinnedPieceReopensInTheSelectedLocale() throws {
        let suite = "LiteraturePins-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let simplified = LiteraturePin(pieceID: "fixture", language: .zhHans, mode: .poetry)
        let traditional = LiteraturePin(pieceID: "fixture", language: .zhHant, mode: .poetry)
        let pins = LiteraturePins(defaults: defaults)
        pins.toggle(simplified)
        pins.toggle(traditional)
        XCTAssertEqual(LiteraturePins(defaults: defaults).items, [traditional, simplified])
        pins.toggle(traditional)
        XCTAssertEqual(LiteraturePins(defaults: defaults).items, [simplified])
    }
}
