import AVFoundation
import CryptoKit
import XCTest
@testable import Subvox

final class BundledWhisperTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appending(path: "bundled-whisper-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() { try? FileManager.default.removeItem(at: directory) }

    func testAppBundlesAUniversalWhisperServerThatRuns() throws {
        let server = LocalWhisper.bundledServer
        XCTAssertTrue(LocalWhisper.validExecutable(server), "Build phase 'Bundle pinned whisper-server' did not run")
        func archs(_ url: URL) throws -> Set<String> {
            let lipo = Process(), pipe = Pipe()
            lipo.executableURL = URL(fileURLWithPath: "/usr/bin/lipo")
            lipo.arguments = ["-archs", url.path]
            lipo.standardOutput = pipe
            try lipo.run(); lipo.waitUntilExit()
            return Set(String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).split(whereSeparator: \.isWhitespace).map(String.init))
        }
        // Release builds are universal; Debug builds only the active architecture. Either way the
        // helper must run everywhere the app itself does.
        let app = try XCTUnwrap(Bundle.main.executableURL)
        XCTAssertTrue(try archs(app).isSubset(of: archs(server)), "whisper-server lacks an architecture the app has")
        let help = Process(), output = Pipe()
        help.executableURL = server
        help.arguments = ["--help"]
        help.standardOutput = output
        help.standardError = output
        try help.run(); help.waitUntilExit()
        let usage = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(help.terminationStatus, 0)
        for flag in ["--dtw", "--no-flash-attn", "--inference-path", "--prompt"] { XCTAssertTrue(usage.contains(flag), flag) }
    }

    func testModelChecksumRejectsWrongSizeOrContent() throws {
        let file = directory.appending(path: "ggml-test.bin"), contents = Data("pinned model".utf8)
        try contents.write(to: file)
        let sha = SHA256.hash(data: contents).map { String(format: "%02x", $0) }.joined()
        XCTAssertTrue(try LocalWhisper.verifyModel(file, size: contents.count, sha256: sha))
        XCTAssertFalse(try LocalWhisper.verifyModel(file, size: contents.count + 1, sha256: sha))
        XCTAssertFalse(try LocalWhisper.verifyModel(file, size: contents.count, sha256: String(repeating: "0", count: 64)))
    }

    func testPinnedModelIsAnAlignableRevisionPinnedDownload() {
        let pinned = LocalWhisper.pinnedModel
        XCTAssertEqual(pinned.sha256.count, 64)
        XCTAssertEqual(LocalWhisper.pinnedModelURL.host, "huggingface.co")
        XCTAssertTrue(LocalWhisper.pinnedModelURL.path.hasSuffix("/resolve/\(pinned.revision)/\(pinned.name)"))
        XCTAssertEqual(LocalWhisper.alignmentPreset(modelPath: pinned.name), "base")
        XCTAssertTrue(LocalWhisper.modelsDirectory.path.contains("/Application Support/Subvox/"))
    }

    @MainActor func testAppsOwnFilesAreTheDefaultAndAValidOverrideWins() throws {
        let name = "Subvox.BundledWhisperTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let models = directory.appending(path: "models"), server = directory.appending(path: "whisper-server")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try Data("#!/bin/sh\n".utf8).write(to: server)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: server.path)

        let missing = LocalWhisper(defaults: defaults, modelDirectory: models, server: server)
        XCTAssertEqual(missing.resolvedExecutable, server)
        XCTAssertNil(missing.resolvedModel, "Without a download there is no model")
        XCTAssertFalse(missing.isConfigured)

        // The app's model counts only at the pinned size.
        let downloaded = models.appending(path: LocalWhisper.pinnedModel.name)
        FileManager.default.createFile(atPath: downloaded.path, contents: nil)
        try FileHandle(forWritingTo: downloaded).truncate(atOffset: UInt64(LocalWhisper.pinnedModel.size))
        let service = LocalWhisper(defaults: defaults, modelDirectory: models, server: server)
        XCTAssertEqual(service.resolvedModel, downloaded)
        XCTAssertTrue(service.isConfigured)

        // An override that can't be used falls back to the app's own file.
        service.modelPath = "/missing/ggml-base.bin"
        service.executablePath = "/missing/whisper-server"
        XCTAssertEqual(service.resolvedModel, downloaded)
        XCTAssertEqual(service.resolvedExecutable, server)

        // A usable override wins.
        let own = directory.appending(path: "ggml-small.bin"), ownServer = directory.appending(path: "my-server")
        FileManager.default.createFile(atPath: own.path, contents: Data([1]))
        try Data("#!/bin/sh\n".utf8).write(to: ownServer)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: ownServer.path)
        service.modelPath = own.path
        service.executablePath = ownServer.path
        XCTAssertEqual(service.resolvedModel, own)
        XCTAssertEqual(service.resolvedExecutable, ownServer)
        service.modelPath = ""
        XCTAssertEqual(service.resolvedModel, downloaded)
    }

    /// End to end with the bundled server and the app-downloaded model; skipped until the model is downloaded.
    @MainActor func testBundledServerTranscribesASavedTake() async throws {
        let model = LocalWhisper.modelsDirectory.appending(path: LocalWhisper.pinnedModel.name)
        guard (try? model.resourceValues(forKeys: [.fileSizeKey]).fileSize) == LocalWhisper.pinnedModel.size else {
            throw XCTSkip("Download the speech recognition model in the app to run this check")
        }
        let speech = directory.appending(path: "take.aiff")
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-v", "Samantha", "-o", speech.path, "The quick brown fox jumps over the lazy dog."]
        try say.run(); say.waitUntilExit()
        try XCTSkipUnless(say.terminationStatus == 0, "No system voice to synthesise a take")
        let name = "Subvox.BundledWhisperTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let whisper = LocalWhisper(defaults: defaults)
        XCTAssertEqual(whisper.resolvedExecutable, LocalWhisper.bundledServer)
        XCTAssertEqual(whisper.resolvedModel, model)
        let transcript = try await whisper.transcribe(audioURL: speech, outputDirectory: directory.appending(path: "out"))
        let heard = transcript.text.lowercased()
        for word in ["quick", "brown", "fox", "lazy", "dog"] { XCTAssertTrue(heard.contains(word), heard) }
        XCTAssertFalse(transcript.words.isEmpty)
    }
}
