import XCTest
@testable import Subvox

final class JawUnitsTests: XCTestCase {
    private func samples() -> [MotionSample] {
        (0..<40).map { i in
            MotionSample(t: Double(i) / 50,
                         rotation: (0.1 * sin(Double(i) * 0.2), 0, 0))
        }
    }

    func testAll39KnownPhonesMapToExactlySixUnits() {
        XCTAssertEqual(JawUnits.version, "jaw6-v1")
        XCTAssertEqual(Phoneme.all.count, 39)
        let units = Phoneme.all.compactMap { JawUnits.unit(for: $0.symbol) }
        XCTAssertEqual(units.count, 39)
        XCTAssertEqual(Set(units), Set(["close", "mid", "open", "open>close", "mid>close", "consonant"]))
        for phone in Phoneme.all {
            XCTAssertEqual(JawUnits.unit(for: phone.symbol), Lexicon.jawClass(phone.symbol) ?? "consonant")
        }
        XCTAssertEqual(JawUnits.unit(for: "AA"), JawUnits.unit(for: "AE"))
        XCTAssertEqual(JawUnits.unit(for: "P"), JawUnits.unit(for: "NG"))
    }

    func testUnknownPhonesNeverBecomeConsonants() {
        for phone in ["", "QX", "SIL", "aa", "AA1"] { XCTAssertNil(JawUnits.unit(for: phone)) }
    }
}
