import CryptoKit
import Foundation

/// Local only is the default. Automatic sharing requires a separate account and dataset grant.
@MainActor @Observable final class ContributionSettings {
    enum Mode: String, CaseIterable, Identifiable {
        case local = "Local only", anonymous = "Anonymous", huggingFace = "Hugging Face"
        var id: String { rawValue }
    }
    /// Bump when the consent text changes; contributors then confirm again before uploading.
    static let consentVersion = "consent-v1-draft"
    static let automaticConsentVersion = "consent-v2-automatic"
    struct AutomaticGrant: Codable, Equatable {
        let id: UUID
        let username: String
        let repo: String
        let version: String
    }
    static let defaultRepo = "stephenokita/subvox-community-test"
    /// Where contributors ask for their shared data to be removed.
    static let removalContact = "me@stephenokita.com"

    @ObservationIgnored private let defaults: UserDefaults
    var mode: Mode {
        didSet {
            defaults.set(mode.rawValue, forKey: "contribution.mode")
            if mode != .huggingFace { stopAutomaticSharing() }
        }
    }
    var repo: String {
        didSet {
            defaults.set(repo, forKey: "contribution.repo")
            if repo != oldValue { stopAutomaticSharing() }
        }
    }
    private(set) var consentedVersion: String?

    private(set) var automaticGrant: AutomaticGrant?

    var hasConsent: Bool { consentedVersion == Self.consentVersion }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        mode = Mode(rawValue: defaults.string(forKey: "contribution.mode") ?? "") ?? .local
        repo = defaults.string(forKey: "contribution.repo") ?? Self.defaultRepo
        consentedVersion = defaults.string(forKey: "contribution.consent")
        automaticGrant = defaults.data(forKey: "contribution.automaticGrant").flatMap {
            try? JSONDecoder().decode(AutomaticGrant.self, from: $0)
        }
    }

    func giveConsent() { consentedVersion = Self.consentVersion; defaults.set(Self.consentVersion, forKey: "contribution.consent") }
    func withdrawConsent() {
        stopAutomaticSharing()
        consentedVersion = nil
        defaults.removeObject(forKey: "contribution.consent")
    }

    func activeAutomaticGrant(for username: String?) -> AutomaticGrant? {
        guard mode == .huggingFace, let grant = automaticGrant, grant.username == username,
              grant.repo == repo, grant.version == Self.automaticConsentVersion else { return nil }
        return grant
    }

    func enableAutomaticSharing(for username: String) {
        guard mode == .huggingFace, !username.isEmpty, !repo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let grant = AutomaticGrant(id: UUID(), username: username, repo: repo, version: Self.automaticConsentVersion)
        guard let data = try? JSONEncoder().encode(grant) else { return }
        automaticGrant = grant
        defaults.set(data, forKey: "contribution.automaticGrant")
    }

    func stopAutomaticSharing() {
        automaticGrant = nil
        defaults.removeObject(forKey: "contribution.automaticGrant")
    }
}

/// One exported session waiting for review, or already sent as a pull request.
struct CommunityUpload: Codable, Identifiable, Equatable, Sendable {
    enum Status: String, Codable, Sendable { case review, uploading, uploaded, failed }
    var id: String            // dataset session ID (a hash; not the local session UUID)
    var folder: String
    var takes: Int
    var month: String
    var status: Status = .review
    var dropped: [String] = []
    var prNumber: Int?
    var prURL: String?
    var error: String?
}

struct CommunityTake: Identifiable, Equatable, Sendable {
    var id: String
    var promptID: String
    var mode: String
    var eligible: Bool
    var cameraEligible: Bool
    var text: String
    var source: String?
}

/// Review queue and pull-request uploads. Exports and the queue live under
/// Application Support/Subvox/Community; the queue survives restarts and failed sends can be retried.
@MainActor @Observable final class CommunityUploader {
    private(set) var items: [CommunityUpload] = []
    private(set) var busy = false
    private(set) var status = ""
    private(set) var error: String?
    let root: URL
    @ObservationIgnored private let runtime: SequenceRuntime
    @ObservationIgnored private let corpus: CommunityCorpus
    @ObservationIgnored private let export: ([String]) async throws -> Void
    @ObservationIgnored private let upload: ([String], String) async throws -> Void

    init(root: URL? = nil, runtimeRoot: URL? = nil, corpus: CommunityCorpus = .bundled,
         export: (([String]) async throws -> Void)? = nil,
         upload: (([String], String) async throws -> Void)? = nil) {
        self.root = root ?? SentenceStore.directory.appending(path: "Community")
        // Shares the neural runtime's installed copy, so uploading doesn't set up Python twice.
        let runtime = SequenceRuntime(root: runtimeRoot ?? SentenceStore.directory.appending(path: "SequenceLearning"))
        self.runtime = runtime
        self.export = export ?? { try await runtime.run($0) }
        self.upload = upload ?? { arguments, token in
            try await runtime.run(arguments, environment: ["SUBVOX_HF_TOKEN": token])
        }
        self.corpus = corpus
        load()
    }

    private var queueURL: URL { root.appending(path: "queue.json") }
    private var exportRoot: URL { root.appending(path: "export") }

    private var queueReadable = true

    func load() {
        do {
            items = try JSONDecoder().decode([CommunityUpload].self, from: Data(contentsOf: queueURL))
            for index in items.indices where items[index].status == .uploading {
                items[index].status = .failed
                items[index].error = "The previous upload was interrupted."
            }
        } catch let failure as NSError where failure.domain == NSCocoaErrorDomain && failure.code == NSFileReadNoSuchFileError {
            items = []
        } catch {
            queueReadable = false
            self.error = "The sharing queue could not be read. Uploads are paused."
        }
    }
    private func save() {
        guard queueReadable else { return }
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try JSONEncoder().encode(items).write(to: queueURL, options: .atomic)
        } catch { self.error = error.localizedDescription }
    }

    static var promptsDirectory: URL? { Bundle.main.resourceURL?.appending(path: "Corpus/community") }

    /// Exported IDs match the Python dataset schema and never expose the local session UUID.
    static func publicSessionID(_ id: UUID) -> String {
        SHA256.hash(data: Data("subvox-community:session:\(id.uuidString)".utf8))
            .prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    /// Re-exports community sessions. Automatic runs consume only the current export manifest.
    @discardableResult
    func prepare(contributor: String, automatic grant: ContributionSettings.AutomaticGrant? = nil) async -> [CommunityUpload]? {
        guard !busy, queueReadable else { return nil }
        busy = true; error = nil; status = "Looking for new sessions…"
        defer { busy = false }
        let output = grant.map { exportRoot.appending(path: $0.id.uuidString) } ?? exportRoot
        let manifest = root.appending(path: "export-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: manifest) }
        do {
            var arguments = ["export-community", "--root", SentenceStore.directory.path, "--output", output.path,
                             "--contributor", contributor, "--consent-version", grant?.version ?? ContributionSettings.consentVersion]
            if let prompts = Self.promptsDirectory { arguments += ["--prompts-dir", prompts.path] }
            let shared = self.shared.map(\.id)
            if !shared.isEmpty { arguments += ["--skip", shared.joined(separator: ",")] }
            if let grant { arguments += ["--automatic-grant", grant.id.uuidString, "--manifest", manifest.path] }
            try await export(arguments)
            let found: [CommunityUpload]
            if grant != nil {
                let paths = try JSONDecoder().decode([String].self, from: Data(contentsOf: manifest))
                found = paths.compactMap { loadSession(URL(fileURLWithPath: $0)) }
            } else {
                found = scan(contributor: contributor)
            }
            merge(found)
            status = found.isEmpty ? "No new community recordings to send." : ""
            return found
        } catch {
            self.error = error.localizedDescription
            status = ""
            return nil
        }
    }

    func scan(contributor: String) -> [CommunityUpload] {
        let folder = exportRoot.appending(path: "data/\(contributor)")
        let sessions = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return sessions.compactMap(loadSession).sorted { $0.id < $1.id }
    }

    private func loadSession(_ url: URL) -> CommunityUpload? {
        guard let data = try? Data(contentsOf: url.appending(path: "session.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = json["session_id"] as? String else { return nil }
        return CommunityUpload(id: id, folder: url.path, takes: json["takes"] as? Int ?? 0, month: json["month"] as? String ?? "")
    }

    func merge(_ found: [CommunityUpload]) {
        for item in found {
            if let i = items.firstIndex(where: { $0.id == item.id }) {
                if items[i].status != .uploaded && items[i].status != .uploading { items[i].takes = item.takes; items[i].folder = item.folder }
            } else { items.append(item) }
        }
        save()
    }

    func takes(_ item: CommunityUpload) -> [CommunityTake] {
        guard let text = try? String(contentsOf: URL(fileURLWithPath: item.folder).appending(path: "takes.jsonl"), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            guard let row = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let id = row["take_id"] as? String, let prompt = row["prompt_id"] as? String else { return nil }
            let entry = corpus.prompt(prompt)
            return CommunityTake(id: id, promptID: prompt, mode: row["mode"] as? String ?? "", eligible: row["eligible"] as? Bool ?? false,
                                 cameraEligible: row["camera_eligible"] as? Bool ?? false, text: entry?.text ?? "Unknown prompt",
                                 source: entry.map { $0.passageID == nil ? "Calibration" : "\($0.title) — \($0.author)" })
        }
    }

    func toggleDrop(_ take: String, in id: String) {
        guard !busy, let i = items.firstIndex(where: { $0.id == id }), items[i].status != .uploaded, items[i].status != .uploading else { return }
        if items[i].dropped.contains(take) { items[i].dropped.removeAll { $0 == take } } else { items[i].dropped.append(take) }
        save()
    }

    func restoreTakes(in id: String) {
        guard !busy, let index = items.firstIndex(where: { $0.id == id && $0.status != .uploaded }) else { return }
        items[index].dropped.removeAll()
        save()
    }

    /// Sends one reviewed session as a pull request. A retry after a failure adds to the same PR.
    private func send(_ id: String, repo: String, token: String, automatic: Bool) async {
        guard !busy, queueReadable, let i = items.firstIndex(where: { $0.id == id }), items[i].status != .uploaded, items[i].takes > items[i].dropped.count else { return }
        busy = true; error = nil; items[i].status = .uploading; items[i].error = nil; save()
        defer { busy = false }
        let item = items[i]
        let output = root.appending(path: "upload-\(UUID()).json")
        var arguments = ["upload-community", "--session", item.folder, "--repo", repo, "--output", output.path]
        if automatic { arguments.append("--automatic") }
        if !item.dropped.isEmpty { arguments += ["--drop", item.dropped.joined(separator: ",")] }
        if let pr = item.prNumber { arguments += ["--pr", String(pr)] }
        if let prompts = Self.promptsDirectory { arguments += ["--prompts-dir", prompts.path] }
        do {
            status = "Sending session \(id.prefix(8))…"
            try await upload(arguments, token)
            let result = try JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any]
            try? FileManager.default.removeItem(at: output)
            guard let j = items.firstIndex(where: { $0.id == id }) else { return }
            items[j].status = .uploaded
            items[j].prNumber = (result?["pr_num"] as? Int) ?? items[j].prNumber
            items[j].prURL = (result?["pr_url"] as? String) ?? items[j].prURL
            status = "Sent as a pull request."
        } catch {
            if let j = items.firstIndex(where: { $0.id == id }) { items[j].status = .failed; items[j].error = error.localizedDescription }
            self.error = error.localizedDescription; status = ""
        }
        save()
    }

    var unsent: [CommunityUpload] { items.filter { $0.status != .uploaded } }
    var pending: [CommunityUpload] { unsent.filter { $0.takes > $0.dropped.count } }
    var shared: [CommunityUpload] { items.filter { $0.status == .uploaded } }

    func sendReviewed(_ ids: [String], repo: String, token: String, automatic: Bool = false,
                      canContinue: () -> Bool = { true }) async {
        guard !busy else { return }
        var sent = 0
        var failed = 0
        for id in ids {
            guard canContinue(), !Task.isCancelled else { break }
            await send(id, repo: repo, token: token, automatic: automatic)
            if shared.contains(where: { $0.id == id }) { sent += 1 } else { failed += 1 }
        }
        error = failed == 0 ? nil : automatic
            ? "Some sessions could not be sent. Subvox will retry while automatic sharing is on."
            : "Some sessions could not be sent. Review the failed sessions and retry."
        status = "\(sent) sent · \(failed) failed"
    }

    func stop() { runtime.terminate() }
}
