import XCTest
@testable import Subvox

final class MotionTests: XCTestCase {
    func testEveryRoleIsInstantUnderReduceMotion() {
        for role in Motion.allCases {
            XCTAssertNil(role.animation(reduceMotion: true), "\(role) must not animate with Reduce Motion on")
            XCTAssertNotNil(role.animation(reduceMotion: false))
        }
    }

    /// Authored animations go through `Motion`, which owns the durations and the Reduce Motion gate.
    func testAnimationsOnlyComeFromMotionRoles() throws {
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "Subvox")
        let pattern = try NSRegularExpression(pattern:
            #"(?<!TimelineView\()\.animation\(|withAnimation\(|\.spring\(|\.easeOut\(|\.easeIn\(|\.easeInOut\(|\.linear\(duration"#)
        var offenders: [String] = []
        var scanned = 0
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)!
        for case let file as URL in files where file.pathExtension == "swift" && file.lastPathComponent != "Motion.swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            scanned += 1
            for (number, line) in text.components(separatedBy: .newlines).enumerated()
            where pattern.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil {
                offenders.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        XCTAssertGreaterThan(scanned, 50, "Didn't find the app sources at \(sources.path)")
        XCTAssertEqual(offenders, [], "Use a Motion role (.motion(_:value:), withMotion) instead")
    }
}
