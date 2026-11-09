import CryptoKit
import Foundation

@MainActor @Observable final class ForcedAligner {
    nonisolated static let model = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    nonisolated static let revision = "2c733782da5604684829819a5eb744c193fe9398"
    nonisolated static let weightsHash = "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551"
    nonisolated private static let files: [(name: String, size: Int, sha256: String)] = [
        ("config.json", 1856, "4609fb49b7e1d28aecb2840da1926c40bd915bc6f1120a940afacf7159bbfb13"),
        ("preprocessor_config.json", 212, "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"),
        ("vocab.json", 4637, "d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0"),
        ("pytorch_model.bin", 1263535127, weightsHash),
    ]
    private let directory = SentenceStore.directory.appending(path: "ForcedAligner/\(ForcedAligner.revision)")
    private let runtime = SequenceRuntime(root: SentenceStore.directory.appending(path: "SequenceLearning"))
    private(set) var isInstalled = false
    private(set) var isInstalling = false
    private(set) var installError: String?

    init() {
        let ready = (try? runtime.project(copy: false))?.appending(path: ".forced-aligner-ready")
        isInstalled = ready.map { FileManager.default.fileExists(atPath: $0.path) } == true
            && Self.files.allSatisfy {
                (try? directory.appending(path: $0.name).resourceValues(forKeys: [.fileSizeKey]).fileSize) == $0.size
            }
    }

    func install() async {
        guard !isInstalling else { return }
        isInstalling = true
        installError = nil
        defer { isInstalling = false }
        do {
            let directory = directory
            try await Task.detached(priority: .utility) {
                try await Self.downloadFiles(to: directory)
            }.value
            // Provision the pinned Python dependencies during setup. Alignment uses uv's offline mode.
            try await runtime.run(["phones", "--text", "Ready", "--language", "en"])
            try Data().write(to: runtime.project().appending(path: ".forced-aligner-ready"), options: .atomic)
            isInstalled = true
        } catch {
            installError = error.localizedDescription
        }
    }

    nonisolated private static func downloadFiles(to directory: URL) async throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, size, expectedHash) in files {
            let destination = directory.appending(path: name)
            if (try? verify(destination, size: size, sha256: expectedHash)) == true { continue }
            let url = URL(string: "https://huggingface.co/\(model)/resolve/\(revision)/\(name)")!
            let (download, response) = try await URLSession.shared.download(from: url)
            defer { try? manager.removeItem(at: download) }
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  try verify(download, size: size, sha256: expectedHash) else {
                throw Failure.invalidDownload
            }
            if manager.fileExists(atPath: destination.path) {
                _ = try manager.replaceItemAt(destination, withItemAt: download)
            } else {
                try manager.moveItem(at: download, to: destination)
            }
        }
    }

    nonisolated private static func verify(_ file: URL, size: Int, sha256: String) throws -> Bool {
        guard try file.resourceValues(forKeys: [.fileSizeKey]).fileSize == size else { return false }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined() == sha256
    }

    enum Failure: LocalizedError {
        case invalidDownload, invalidResult
        var errorDescription: String? {
            switch self {
            case .invalidDownload: String(localized: "The word alignment model download could not be verified.")
            case .invalidResult: String(localized: "Word alignment returned invalid timing.")
            }
        }
    }

    struct Result: Decodable {
        struct Word: Decodable {
            struct Phone: Decodable {
                var ipa: String
                var start: Double
                var end: Double
                var score: Double
            }
            var text: String
            var start: Double?
            var end: Double?
            var timed: Bool
            var phones: [Phone]
        }
        var words: [Word]
        var coverage: Double
        var accepted: Bool
        var model: String
        var revision: String

        func anchors(origin: Double, cue: ClosedRange<Double>) throws -> [MotionWordAnchor] {
            var previousEnd = cue.lowerBound
            return try words.filter(\.timed).map { word in
                guard let start = word.start.map({ $0 + origin }), let end = word.end.map({ $0 + origin }),
                      start.isFinite, end.isFinite, start >= previousEnd - 1e-7,
                      end > start, end <= cue.upperBound + 1e-7 else { throw Failure.invalidResult }
                previousEnd = end
                return MotionWordAnchor(word: word.text.lowercased().replacingOccurrences(of: "’", with: "'"), start: start, end: end)
            }
        }

        func phoneAlignment(origin: Double, cue: ClosedRange<Double>, profileHash: String) throws -> PhoneAlignment? {
            let phones = words.enumerated().flatMap { index, word in
                word.timed ? word.phones.map {
                    PhoneBoundary(ipa: $0.ipa, start: $0.start + origin, end: $0.end + origin,
                                  backendScore: $0.score, word: word.text, wordIndex: index)
                } : []
            }
            guard !phones.isEmpty else { return nil }
            let alignment = PhoneAlignment(condition: .normal, phones: phones, provenance: .init(
                method: .localAcousticAlignment, backend: model + "@" + revision, modelSHA256: ForcedAligner.weightsHash,
                pronunciationSHA256: profileHash, note: "Local IPA CTC forced alignment"))
            try alignment.validate(in: cue)
            return alignment
        }
    }

    func align(audio: URL, range: ClosedRange<Double>, text: String, language: String) async throws -> Result {
        let temporary = FileManager.default.temporaryDirectory.appending(path: "subvox-align-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let wav = temporary.appending(path: "audio.wav")
        let output = temporary.appending(path: "alignment.json")
        let conversion = Task.detached(priority: .userInitiated) {
            try LocalWhisper.convertToWAV(audio, destination: wav, timeRange: range)
        }
        try await withTaskCancellationHandler(operation: { try await conversion.value }, onCancel: { conversion.cancel() })
        try await runtime.run(["align", "--audio", wav.path, "--start", "0", "--end",
                               String(range.upperBound - range.lowerBound), "--text=\(text)", "--language", language,
                               "--output", output.path], offline: true)
        let result = try JSONDecoder().decode(Result.self, from: Data(contentsOf: output))
        guard result.model == Self.model, result.revision == Self.revision else { throw Failure.invalidResult }
        return result
    }
}
