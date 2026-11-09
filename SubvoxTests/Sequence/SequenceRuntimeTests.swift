import XCTest
@testable import Subvox

final class SequenceRuntimeTests: XCTestCase {
    @MainActor func testProjectRepairsPartialRuntimeAndPreservesExistingEnvironment() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? manager.removeItem(at: root) }

        let runtime = SequenceRuntime(root: root)
        let project = try runtime.project()
        let lockfile = project.appending(path: "uv.lock")
        let lockfileData = try Data(contentsOf: lockfile)
        let projectManifest = project.appending(path: "pyproject.toml")
        let manifestData = try Data(contentsOf: projectManifest)
        let environment = project.appending(path: ".venv")
        try manager.createDirectory(at: environment, withIntermediateDirectories: false)
        let marker = environment.appending(path: "existing.txt")
        try Data("existing environment".utf8).write(to: marker)
        try manager.removeItem(at: lockfile)
        try Data().write(to: projectManifest)

        XCTAssertEqual(try runtime.project(), project)
        XCTAssertEqual(try Data(contentsOf: lockfile), lockfileData)
        XCTAssertEqual(try Data(contentsOf: projectManifest), manifestData)
        XCTAssertEqual(try String(contentsOf: marker, encoding: .utf8), "existing environment")
    }
}
