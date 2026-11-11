import XCTest
@testable import Subvox

final class ArticulationDiagramTests: XCTestCase {
    func testEveryPracticeSoundHasADiagramOrAStatedReasonAndNoStaleEntries() throws {
        let diagrams = try XCTUnwrap(ArticulationDiagrams.bundled, "Corpus/practice/articulation-v1.json is missing or does not decode")
        var expected = Set<String>()
        for (language, sounds) in PracticeCorpus.bundled.sounds {
            for sound in sounds {
                let key = ArticulationDiagrams.key(language: language, ipa: sound.ipa)
                expected.insert(key)
                let drawn = diagrams.diagram(language: language, ipa: sound.ipa) != nil
                XCTAssertNotEqual(drawn, diagrams.withoutDiagram[key] != nil, "\(key) must be drawn or listed without a diagram, not both or neither")
            }
        }
        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(Set(diagrams.sounds.keys).union(diagrams.withoutDiagram.keys), expected,
                       "Regenerate with Tools/chimera after changing the practice sounds")
    }

    func testEveryPostureDrawsEveryPartWithFinitePointsAndTheViewIsAWindow() throws {
        let diagrams = try XCTUnwrap(ArticulationDiagrams.bundled)
        XCTAssertEqual(diagrams.view.count, 4)
        XCTAssertLessThan(diagrams.view[0], diagrams.view[2])
        XCTAssertLessThan(diagrams.view[1], diagrams.view[3])
        XCTAssertTrue(diagrams.parts.contains("tongue") && diagrams.parts.contains("upperLip"))
        for (id, posture) in diagrams.postures {
            for part in diagrams.parts {
                let points = try XCTUnwrap(posture[part], "\(id) lacks \(part)")
                XCTAssertGreaterThanOrEqual(points.count, 2, "\(id) \(part)")
                XCTAssertTrue(points.allSatisfy { $0.x.isFinite && $0.y.isFinite }, "\(id) \(part)")
            }
        }
    }

    func testTonesHaveNoDiagramAndMovingSoundsSayTheyShowTheStart() throws {
        let diagrams = try XCTUnwrap(ArticulationDiagrams.bundled)
        XCTAssertNil(diagrams.diagram(language: "zh", ipa: "˥"))
        XCTAssertEqual(diagrams.diagram(language: "en", ipa: "aɪ")?.sound.note, .start)
        XCTAssertNil(diagrams.diagram(language: "en", ipa: "p")?.sound.note)
        XCTAssertEqual(diagrams.diagram(language: "en", ipa: "p")?.sound.posture, diagrams.diagram(language: "fr", ipa: "p")?.sound.posture)
    }
}
