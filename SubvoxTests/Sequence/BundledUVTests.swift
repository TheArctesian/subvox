import XCTest
@testable import Subvox

final class BundledUVTests: XCTestCase {
    func testAppBundlesThePinnedUVForEveryMacArchitecture() throws {
        let uv = SequenceRuntime.bundledUV
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: uv.path), "Build phase 'Bundle pinned uv' did not run")
        let child = Process(), pipe = Pipe()
        child.executableURL = uv
        child.arguments = ["--version"]
        child.standardOutput = pipe
        try child.run(); child.waitUntilExit()
        let version = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(child.terminationStatus, 0)
        XCTAssertTrue(version.hasPrefix("uv 0.13.6 "), version)
    }

    func testUVKeepsPythonCachesAndSettingsInsideTheAppsOwnFolder() {
        let root = URL(fileURLWithPath: "/tmp/subvox-runtime-root")
        let environment = SequenceRuntime.uvEnvironment(root: root)
        XCTAssertEqual(environment["UV_PYTHON_PREFERENCE"], "only-managed")
        XCTAssertEqual(environment["UV_NO_CONFIG"], "1")
        for key in ["UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"] {
            XCTAssertTrue(environment[key]?.hasPrefix(root.path + "/") == true, key)
        }
    }
}
