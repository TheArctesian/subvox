import CryptoKit
import Foundation

/// The app's local Python runtime (ml/subvox_ml, run with `uv`). Shared by neural training and
/// community upload. Source files are content-hashed and copied once into Application Support.
/// uv ships inside the app (Contents/Helpers/uv, pinned by Tools/runtime/bundle-uv.sh); the Python
/// it downloads, its package cache and each environment live under `root`, so users install nothing.
@MainActor final class SequenceRuntime {
    enum Failure: LocalizedError {
        case runtime, backend(String)
        var errorDescription: String? {
            switch self {
            case .runtime: "This copy of Subvox is missing its bundled Python runner. Reinstall Subvox."
            case .backend(let message): "Local Python operation failed: " + message
            }
        }
    }

    let root: URL
    private var process: Process?
    init(root: URL) { self.root = root }

    func project(copy: Bool = true) throws -> URL {
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
        let projectFiles = try files.map { file, source in
            let data = try Data(contentsOf: source)
            hash.update(data: Data(file.utf8))
            hash.update(data: data)
            return (path: file, source: source, size: data.count)
        }
        let identity = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let destination = root.appending(path: "Runtime/\(identity)")
        if copy { try installProject(projectFiles, at: destination) }
        return destination
    }

    private typealias ProjectFile = (path: String, source: URL, size: Int)

    private func installProject(_ files: [ProjectFile], at destination: URL) throws {
        let manager = FileManager.default
        func isComplete(_ file: ProjectFile) -> Bool {
            let target = destination.appending(path: file.path)
            let values = try? target.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            return values?.isRegularFile == true && values?.fileSize == file.size
        }

        if manager.fileExists(atPath: destination.path) {
            // Repair legacy partial copies without replacing an existing environment.
            for file in files where !isComplete(file) {
                let target = destination.appending(path: file.path)
                try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data(contentsOf: file.source).write(to: target, options: .atomic)
            }
            return
        }

        let staging = destination.deletingLastPathComponent().appending(path: ".staging-\(UUID())")
        defer { try? manager.removeItem(at: staging) }
        for file in files {
            let target = staging.appending(path: file.path)
            try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try manager.copyItem(at: file.source, to: target)
        }
        do {
            try manager.moveItem(at: staging, to: destination)
        } catch {
            // Another app instance may have published the same content while we staged it.
            guard files.allSatisfy(isComplete) else { throw error }
        }
    }

    /// Runs `subvox-ml <arguments>`. `environment` is added to the app's own (secrets go here, never
    /// in arguments, which other processes can read). `status` receives the latest log line.
    /// The uv inside the app bundle.
    nonisolated static var bundledUV: URL { Bundle.main.bundleURL.appending(path: "Contents/Helpers/uv") }

    /// Settings that keep uv to the app's own files: Python it installs itself (never one found on the
    /// Mac, so every user gets the same interpreter), caches under `root`, and no user uv config.
    nonisolated static func uvEnvironment(root: URL) -> [String: String] {
        ["UV_PYTHON_PREFERENCE": "only-managed",
         "UV_PYTHON_INSTALL_DIR": root.appending(path: "uv/python").path,
         "UV_CACHE_DIR": root.appending(path: "uv/cache").path,
         "UV_NO_CONFIG": "1"]
    }

    func run(_ arguments: [String], environment: [String: String] = [:], offline: Bool = false, status: (String) -> Void = { _ in }) async throws {
        let uv = Self.bundledUV.path
        guard FileManager.default.isExecutableFile(atPath: uv) else { throw Failure.runtime }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let project = try project(), child = Process(), log = root.appending(path: "process-\(UUID()).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log); defer { try? handle.close() }
        child.executableURL = URL(fileURLWithPath: uv)
        child.arguments = ["run", "--frozen"] + (offline ? ["--offline"] : []) + ["--project", project.path, "subvox-ml"] + arguments
        child.environment = ProcessInfo.processInfo.environment.merging(Self.uvEnvironment(root: root)) { $1 }.merging(environment) { $1 }
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
