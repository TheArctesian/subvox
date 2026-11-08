import CryptoKit
import Foundation

/// The app's local Python runtime (ml/subvox_ml, run with `uv`). Shared by neural training and
/// community upload. Source files are content-hashed and copied once into Application Support.
@MainActor final class SequenceRuntime {
    enum Failure: LocalizedError {
        case runtime, backend(String)
        var errorDescription: String? {
            switch self {
            case .runtime: "Install uv locally to run the pinned Python runtime."
            case .backend(let message): "Local Python operation failed: " + message
            }
        }
    }

    let root: URL
    private var process: Process?
    init(root: URL) { self.root = root }

    func project() throws -> URL {
        let bundled = Bundle.main.resourceURL!.appending(path: "SequenceRuntime")
        // This file sits at Subvox/Sequence/, so three levels up is the repository root.
        let development = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appending(path: "ml")
        let isBundled = FileManager.default.fileExists(atPath: bundled.appending(path: "uv.lock").path)
        let source = isBundled ? bundled : development
        // evaluation.py reads the distractor bank from <runtime>/Corpus, so it travels with the runtime.
        let corpus = isBundled ? Bundle.main.resourceURL!.appending(path: "Corpus") : development.deletingLastPathComponent().appending(path: "Corpus")
        let python = ["pyproject.toml", "uv.lock"] + (try FileManager.default.contentsOfDirectory(atPath: source.appending(path: "subvox_ml").path))
            .filter { $0.hasSuffix(".py") }.sorted().map { "subvox_ml/" + $0 }
        let files = python.map { ($0, source.appending(path: $0)) } + [("Corpus/diphones.txt", corpus.appending(path: "diphones.txt"))]
        var hash = SHA256()
        for (file, url) in files { hash.update(data: Data(file.utf8)); hash.update(data: try Data(contentsOf: url)) }
        let identity = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let destination = root.appending(path: "Runtime/\(identity)")
        if !FileManager.default.fileExists(atPath: destination.path) {
            for directory in ["subvox_ml", "Corpus"] { try FileManager.default.createDirectory(at: destination.appending(path: directory), withIntermediateDirectories: true) }
            for (file, url) in files { try FileManager.default.copyItem(at: url, to: destination.appending(path: file)) }
        }
        return destination
    }

    /// Runs `subvox-ml <arguments>`. `environment` is added to the app's own (secrets go here, never
    /// in arguments, which other processes can read). `status` receives the latest log line.
    func run(_ arguments: [String], environment: [String: String] = [:], status: (String) -> Void = { _ in }) async throws {
        let uv = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"].first { FileManager.default.isExecutableFile(atPath: $0) }
        guard let uv else { throw Failure.runtime }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let project = try project(), child = Process(), log = root.appending(path: "process-\(UUID()).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
        child.executableURL = URL(fileURLWithPath: uv)
        child.arguments = ["run", "--frozen", "--project", project.path, "subvox-ml"] + arguments
        if !environment.isEmpty { child.environment = ProcessInfo.processInfo.environment.merging(environment) { $1 } }
        child.standardOutput = handle; child.standardError = handle
        try child.run(); process = child
        defer { if child.isRunning { child.terminate() }; process = nil }
        while child.isRunning {
            try Task.checkCancellation()
            if let data = try? Data(contentsOf: log), let line = String(data: data, encoding: .utf8)?.split(separator: "\n").last {
                status(line.hasPrefix("{") ? "Finishing local operation…" : String(line.prefix(200)))
            }
            try await Task.sleep(for: .milliseconds(300))
        }
        try Task.checkCancellation()
        guard child.terminationStatus == 0 else {
            let detail = (try? String(contentsOf: log, encoding: .utf8)).map { String($0.suffix(1600)) } ?? "See \(log.path)"
            throw Failure.backend(detail)
        }
    }

    func terminate() { if let process, process.isRunning { process.terminate() } }
}
