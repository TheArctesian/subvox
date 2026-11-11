import XCTest
@testable import Subvox

final class StubProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) -> (Int, Data))?
    nonisolated(unsafe) static var requests: [URLRequest] = []
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        var request = self.request
        if request.httpBody == nil, let stream = request.httpBodyStream {
            stream.open(); var data = Data(); var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable { let n = stream.read(&buffer, maxLength: buffer.count); if n <= 0 { break }; data.append(buffer, count: n) }
            stream.close(); request.httpBody = data
        }
        Self.requests.append(request)
        let (status, body) = Self.handler?(request) ?? (500, Data())
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class MemoryStore: TokenStore, @unchecked Sendable {
    var data: Data?
    func load() -> Data? { data }
    func save(_ data: Data) throws { self.data = data }
    func delete() { data = nil }
}

final class HuggingFaceAuthTests: XCTestCase {
    func testPKCEMatchesTheRFC7636TestVector() {
        XCTAssertEqual(HuggingFaceOAuth.challenge(for: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")
        let a = HuggingFaceOAuth.randomToken(), b = HuggingFaceOAuth.randomToken()
        XCTAssertNotEqual(a, b)
        XCTAssertNil(a.rangeOfCharacter(from: CharacterSet(charactersIn: "+/=")), "base64url, no padding")
    }

    func testAuthorizationURLAsksOnlyForTheMinimalScopes() throws {
        let url = HuggingFaceOAuth.authorization(redirect: "http://127.0.0.1:5123/oauth/callback", state: "s1", challenge: "c1")
        let items = Dictionary(uniqueKeysWithValues: try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems).map { ($0.name, $0.value!) })
        XCTAssertEqual(url.host, "huggingface.co")
        XCTAssertEqual(items["client_id"], "ae146936-11cf-4196-a5ac-7288fafbf4ba")
        XCTAssertEqual(items["scope"], "openid profile write-discussions")
        XCTAssertEqual(items["code_challenge_method"], "S256")
        XCTAssertEqual(items["redirect_uri"], "http://127.0.0.1:5123/oauth/callback")
        XCTAssertNil(items["client_secret"])
    }

    func testCallbackChecksStateAndSurfacesDenial() throws {
        XCTAssertEqual(try HuggingFaceOAuth.code(fromRequestLine: "GET /oauth/callback?code=abc&state=s1 HTTP/1.1", expectedState: "s1"), "abc")
        XCTAssertThrowsError(try HuggingFaceOAuth.code(fromRequestLine: "GET /oauth/callback?code=abc&state=evil HTTP/1.1", expectedState: "s1")) {
            XCTAssertEqual($0 as? HuggingFaceOAuth.Failure, .stateMismatch)
        }
        XCTAssertThrowsError(try HuggingFaceOAuth.code(fromRequestLine: "GET /oauth/callback?error=access_denied&state=s1 HTTP/1.1", expectedState: "s1")) {
            XCTAssertEqual($0 as? HuggingFaceOAuth.Failure, .denied("access_denied"))
        }
        XCTAssertThrowsError(try HuggingFaceOAuth.code(fromRequestLine: "GET /favicon.ico HTTP/1.1", expectedState: "s1"))
    }

    func testTokenExchangeIsAPublicClientRequest() throws {
        let request = HuggingFaceOAuth.exchange(code: "c&1", verifier: "v", redirect: "http://127.0.0.1:9/oauth/callback")
        let body = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertTrue(body.contains("grant_type=authorization_code") && body.contains("code=c%261") && body.contains("code_verifier=v"))
        XCTAssertTrue(body.contains("client_id=ae146936"))
        XCTAssertFalse(body.contains("client_secret"))
        let token = try HuggingFaceOAuth.token(from: Data(#"{"access_token":"hf_x","expires_in":3600,"scope":"openid profile write-discussions"}"#.utf8),
                                               now: Date(timeIntervalSince1970: 0))
        XCTAssertEqual(token.expiresAt, Date(timeIntervalSince1970: 3600))
        XCTAssertNil(token.refreshToken)
    }

    @MainActor func testExpiredTokensRefreshOrRequireSignInAgain() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        let store = MemoryStore()
        let old = HuggingFaceToken(accessToken: "old", refreshToken: "r1", expiresAt: Date(timeIntervalSince1970: 100), scope: nil, username: "stephenokita")
        try store.save(JSONEncoder().encode(old))
        StubProtocol.requests = []
        let account = HuggingFaceAccount(store: store, session: URLSession(configuration: configuration), open: { _ in })
        XCTAssertEqual(account.username, "stephenokita")
        StubProtocol.handler = { _ in (503, Data()) }
        do { _ = try await account.accessToken(now: Date(timeIntervalSince1970: 200)); XCTFail("expected temporary failure") }
        catch { XCTAssertEqual(error as? HuggingFaceOAuth.Failure, .http(503, "")) }
        XCTAssertEqual(account.username, "stephenokita", "A temporary failure keeps credentials available for background retry")
        StubProtocol.handler = { _ in (200, Data(#"{"access_token":"new","expires_in":28800}"#.utf8)) }
        let token = try await account.accessToken(now: Date(timeIntervalSince1970: 200))
        XCTAssertEqual(token, "new")
        XCTAssertTrue(String(decoding: StubProtocol.requests.last?.httpBody ?? Data(), as: UTF8.self).contains("grant_type=refresh_token"))
        let saved = try JSONDecoder().decode(HuggingFaceToken.self, from: XCTUnwrap(store.data))
        XCTAssertEqual(saved.refreshToken, "r1", "A refresh without a new refresh token keeps the old one")

        try store.save(JSONEncoder().encode(HuggingFaceToken(accessToken: "x", refreshToken: nil, expiresAt: .distantPast, scope: nil, username: "stephenokita")))
        do { _ = try await account.accessToken(); XCTFail("expected sign-in") }
        catch { XCTAssertEqual(error as? HuggingFaceOAuth.Failure, .needsSignIn) }
        XCTAssertEqual(account.state, .signedOut)
        XCTAssertNil(store.data)
    }

    @MainActor func testPendingLinkCanBeReopenedAndCancelled() async throws {
        var opened: [URL] = []
        let account = HuggingFaceAccount(store: MemoryStore(), open: { opened.append($0) })
        let task = Task { await account.signIn() }
        for _ in 0..<100 where account.pendingURL == nil { try await Task.sleep(for: .milliseconds(20)) }
        let link = try XCTUnwrap(account.pendingURL)
        XCTAssertEqual(account.state, .signingIn)
        account.reopenSignIn()
        XCTAssertEqual(opened, [link, link], "Open again uses the same link (same state and PKCE challenge)")
        account.cancelSignIn()
        await task.value
        XCTAssertEqual(account.state, .signedOut)
        XCTAssertNil(account.pendingURL)
        XCTAssertNil(account.error, "Cancelling isn't an error")
    }

    func testLoopbackRedirectReceivesTheBrowserRequest() async throws {
        let redirect = try LoopbackRedirect()
        let uri = try await redirect.start()
        XCTAssertTrue(uri.hasPrefix("http://127.0.0.1:") && uri.hasSuffix("/oauth/callback"))
        async let line = redirect.requestLine(timeout: .seconds(10))
        let (data, response) = try await URLSession.shared.data(from: URL(string: uri + "?code=xyz&state=s9")!)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(String(decoding: data, as: UTF8.self).contains("return to Subvox"))
        let received = try await line
        XCTAssertEqual(try HuggingFaceOAuth.code(fromRequestLine: received, expectedState: "s9"), "xyz")
    }
}

final class CommunityUploaderTests: XCTestCase {
    @MainActor func testQueueFindsSessionsKeepsSentOnesAndRemembersDroppedTakes() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "uploader-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let session = root.appending(path: "export/data/stephenokita/abc123")
        try FileManager.default.createDirectory(at: session, withIntermediateDirectories: true)
        try Data(#"{"session_id":"abc123","takes":2,"month":"2026-11"}"#.utf8).write(to: session.appending(path: "session.json"))
        let prompt = CommunityCorpus.bundled.calibration[0]
        try Data((#"{"take_id":"t1","prompt_id":"\#(prompt.promptID)","mode":"mouthed","eligible":true,"camera_eligible":true}"# + "\n").utf8)
            .write(to: session.appending(path: "takes.jsonl"))
        let uploader = CommunityUploader(root: root, runtimeRoot: root.appending(path: "runtime"))
        uploader.merge(uploader.scan(contributor: "stephenokita"))
        XCTAssertEqual(uploader.items.map(\.id), ["abc123"])
        XCTAssertEqual(uploader.takes(uploader.items[0]).first?.text, prompt.text)
        uploader.toggleDrop("t1", in: "abc123")
        let reloaded = CommunityUploader(root: root, runtimeRoot: root.appending(path: "runtime"))
        XCTAssertEqual(reloaded.items.first?.dropped, ["t1"], "The queue survives a restart")
        reloaded.merge(reloaded.scan(contributor: "stephenokita"))
        XCTAssertEqual(reloaded.items.count, 1, "Re-scanning doesn't duplicate a session")
        reloaded.toggleDrop("t2", in: "abc123")
        XCTAssertTrue(reloaded.pending.isEmpty, "An empty upload cannot be sent")
        XCTAssertEqual(reloaded.unsent.map(\.id), ["abc123"], "The session remains available for recovery")
        reloaded.restoreTakes(in: "abc123")
        XCTAssertEqual(reloaded.pending.map(\.id), ["abc123"])
    }

    @MainActor func testMixedUploadResultsRetainFailureAndRetryOnlyThatSession() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "uploader-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        var failFirst = true
        var attempted: [String] = []
        let uploader = CommunityUploader(root: root, runtimeRoot: root) { arguments, _ in
            let session = arguments[try XCTUnwrap(arguments.firstIndex(of: "--session")) + 1]
            attempted.append(session)
            if session == "first", failFirst { throw URLError(.notConnectedToInternet) }
            let result = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            try Data(#"{"pr_num":1}"#.utf8).write(to: URL(fileURLWithPath: result))
        }
        uploader.merge([
            CommunityUpload(id: "first", folder: "first", takes: 1, month: "2026-11"),
            CommunityUpload(id: "second", folder: "second", takes: 1, month: "2026-11"),
        ])
        await uploader.sendReviewed(["first", "second"], repo: "test/dataset", token: "fixture")
        XCTAssertEqual(uploader.status, "1 sent · 1 failed")
        XCTAssertNotNil(uploader.error)
        XCTAssertEqual(uploader.pending.map(\.id), ["first"])
        XCTAssertEqual(uploader.pending.first?.status, .failed)
        XCTAssertEqual(uploader.shared.map(\.id), ["second"])

        failFirst = false
        await uploader.sendReviewed(["first"], repo: "test/dataset", token: "fixture")
        XCTAssertEqual(uploader.status, "1 sent · 0 failed")
        XCTAssertNil(uploader.error)
        XCTAssertEqual(attempted, ["first", "second", "first"])
    }

    @MainActor func testAutomaticManifestResumesAfterInterruptedUploadAndStopsOnOptOut() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "automatic-uploader-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "subvox-test-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = ContributionSettings(defaults: defaults)
        settings.mode = .huggingFace
        settings.repo = "test/dataset"
        settings.enableAutomaticSharing(for: "alice")
        let grant = try XCTUnwrap(settings.activeAutomaticGrant(for: "alice"))
        func argument(_ name: String, in arguments: [String]) throws -> String {
            let index = try XCTUnwrap(arguments.firstIndex(of: name))
            return arguments[index + 1]
        }

        var attempts: [String] = []
        let uploader = CommunityUploader(root: root, runtimeRoot: root, export: { arguments in
            XCTAssertEqual(try argument("--automatic-grant", in: arguments), grant.id.uuidString)
            let output = URL(fileURLWithPath: try argument("--output", in: arguments))
            let folders = try ["first", "second", "stale"].map { id in
                let folder = output.appending(path: "data/alice/\(id)")
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let metadata = ["session_id": id, "month": "2026-11", "takes": 2] as [String: Any]
                try JSONSerialization.data(withJSONObject: metadata).write(to: folder.appending(path: "session.json"))
                return folder.path
            }
            let manifest = try argument("--manifest", in: arguments)
            try JSONEncoder().encode(Array(folders.prefix(2))).write(to: URL(fileURLWithPath: manifest))
        }, upload: { arguments, _ in
            attempts.append(URL(fileURLWithPath: try argument("--session", in: arguments)).lastPathComponent)
            throw URLError(.networkConnectionLost)
        })
        let exported = await uploader.prepare(contributor: "alice", automatic: grant)
        let ids = try XCTUnwrap(exported).map(\.id)
        XCTAssertEqual(ids, ["first", "second"], "A stale folder outside the manifest must stay out of the queue")
        await uploader.sendReviewed(["first"], repo: grant.repo, token: "fixture", automatic: true)
        XCTAssertEqual(uploader.pending.first?.status, .failed)

        // Reopen the queue and simulate the Hub returning the existing PR after retry.
        let restored = CommunityUploader(root: root, runtimeRoot: root) { arguments, token in
            XCTAssertEqual(token, "fixture")
            XCTAssertTrue(arguments.contains("--automatic"))
            XCTAssertEqual(try argument("--repo", in: arguments), grant.repo)
            attempts.append(URL(fileURLWithPath: try argument("--session", in: arguments)).lastPathComponent)
            let receipt = try argument("--output", in: arguments)
            try Data(#"{"pr_num":7,"pr_url":"https://hub.test/pr/7"}"#.utf8).write(to: URL(fileURLWithPath: receipt))
            settings.stopAutomaticSharing()
        }
        XCTAssertEqual(restored.pending.first?.status, .failed)
        await restored.sendReviewed(ids, repo: grant.repo, token: "fixture", automatic: true) {
            settings.activeAutomaticGrant(for: "alice") == grant
        }
        XCTAssertEqual(attempts, ["first", "first"])
        XCTAssertEqual(restored.shared.first?.prNumber, 7)
        XCTAssertEqual(restored.pending.map(\.id), ["second"])
        XCTAssertNil(ContributionSettings(defaults: defaults).activeAutomaticGrant(for: "alice"))
    }

    @MainActor func testAutomaticSharingRequiresItsOwnAccountAndDatasetGrant() throws {
        let suite = "subvox-test-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = ContributionSettings(defaults: defaults)
        settings.mode = .huggingFace
        settings.giveConsent()
        XCTAssertNil(settings.activeAutomaticGrant(for: "alice"))
        settings.enableAutomaticSharing(for: "alice")
        let first = try XCTUnwrap(settings.activeAutomaticGrant(for: "alice"))
        XCTAssertNil(settings.activeAutomaticGrant(for: "bob"))
        XCTAssertEqual(ContributionSettings(defaults: defaults).activeAutomaticGrant(for: "alice"), first)
        settings.repo = "owner/new-dataset"
        XCTAssertNil(settings.activeAutomaticGrant(for: "alice"))
        settings.enableAutomaticSharing(for: "alice")
        XCTAssertNotEqual(settings.activeAutomaticGrant(for: "alice")?.id, first.id)
        settings.mode = .local
        settings.mode = .huggingFace
        XCTAssertNil(settings.activeAutomaticGrant(for: "alice"))
    }

    @MainActor func testLibraryQueueRequiresTheOriginalGrantAndKeepsSectionTakesTogether() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "library-uploader-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "subvox-test-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = ContributionSettings(defaults: defaults)
        settings.enableAutomaticSharing(for: "alice")
        XCTAssertNil(settings.activeLibraryGrant(for: "alice"))
        settings.enableLibrarySharing(for: "alice")
        let automatic = try XCTUnwrap(settings.activeAutomaticGrant(for: "alice"))
        let library = try XCTUnwrap(settings.activeLibraryGrant(for: "alice"))
        XCTAssertEqual(ContributionSettings(defaults: defaults).activeLibraryGrant(for: "alice"), library)
        XCTAssertNil(settings.activeLibraryGrant(for: "bob"))
        let prompt = try XCTUnwrap(LibraryPilotCatalog.bundled.sections(pieceID: "invictus", language: "en").first)
        var uploads = 0
        let uploader = CommunityUploader(root: root, runtimeRoot: root, export: { arguments in
            func argument(_ name: String) throws -> String {
                arguments[try XCTUnwrap(arguments.firstIndex(of: name)) + 1]
            }
            XCTAssertEqual(try argument("--automatic-grant"), automatic.id.uuidString)
            XCTAssertEqual(try argument("--library-grant"), library.id.uuidString)
            let folder = URL(fileURLWithPath: try argument("--output")).appending(path: "data/alice/pilot")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data(#"{"session_id":"pilot","consent_version":"consent-v3-library-pilot","takes":2}"#.utf8)
                .write(to: folder.appending(path: "session.json"))
            let rows = ["spoken", "mouthed"].map { mode in
                #"{"take_id":"\#(mode)","prompt_id":"\#(prompt.promptID)","mode":"\#(mode)","eligible":true}"#
            }.joined(separator: "\n")
            try Data(rows.utf8).write(to: folder.appending(path: "takes.jsonl"))
            try JSONEncoder().encode([folder.path]).write(to: URL(fileURLWithPath: try argument("--manifest")))
        }, upload: { _, _ in
            uploads += 1
            throw URLError(.notConnectedToInternet)
        })
        let prepared = await uploader.prepare(contributor: "alice", automatic: automatic, library: library)
        let item = try XCTUnwrap(prepared?.first)
        XCTAssertEqual(item.libraryGrantID, library.id)
        XCTAssertEqual(item.libraryTakeIDs, Set(["spoken", "mouthed"]))
        XCTAssertEqual(uploader.takes(item).first?.text, prompt.text)
        uploader.toggleDrop("spoken", in: item.id)
        XCTAssertEqual(Set(uploader.items[0].dropped), Set(["spoken", "mouthed"]))
        uploader.restoreTakes(in: item.id)

        await uploader.sendReviewed([item.id], repo: library.repo, token: "fixture")
        XCTAssertEqual(uploads, 0)
        settings.stopLibrarySharing()
        settings.enableLibrarySharing(for: "alice")
        let renewed = try XCTUnwrap(settings.activeLibraryGrant(for: "alice"))
        XCTAssertNotEqual(renewed.id, library.id)
        await uploader.sendReviewed([item.id], repo: renewed.repo, token: "fixture", automatic: true, library: renewed)
        XCTAssertEqual(uploads, 0)
        settings.repo = "test/new-dataset"
        XCTAssertNil(settings.activeLibraryGrant(for: "alice"))
    }

    @MainActor func testRecoveredLibraryReceiptUsesSubmittedTakeIDs() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "library-receipt-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let grant = ContributionSettings.AutomaticGrant(id: UUID(), username: "alice", repo: "test/dataset",
                                                         version: ContributionSettings.libraryConsentVersion)
        let uploader = CommunityUploader(root: root, runtimeRoot: root) { arguments, _ in
            let index = try XCTUnwrap(arguments.firstIndex(of: "--output"))
            let result = URL(fileURLWithPath: arguments[index + 1])
            try Data(#"{"pr_num":7,"take_ids":["a","b","c","d"]}"#.utf8).write(to: result)
        }
        var item = CommunityUpload(id: "pilot", folder: "fixture", takes: 4, month: "2026-11")
        item.libraryGrantID = grant.id
        item.libraryTakeIDs = ["a", "b", "c", "d"]
        item.dropped = ["c", "d"]
        uploader.merge([item])
        await uploader.sendReviewed([item.id], repo: grant.repo, token: "fixture", automatic: true, library: grant)

        let reopened = CommunityUploader(root: root, runtimeRoot: root)
        let receipt = try XCTUnwrap(reopened.shared.first)
        XCTAssertEqual(receipt.prNumber, 7)
        XCTAssertEqual(receipt.libraryTakeIDs, Set(["a", "b", "c", "d"]))
        XCTAssertTrue(receipt.dropped.isEmpty)
    }

    @MainActor func testSharingDefaultsToHuggingFaceAndConsentIsVersioned() {
        let defaults = UserDefaults(suiteName: "subvox-test-\(UUID())")!
        let settings = ContributionSettings(defaults: defaults)
        XCTAssertEqual(settings.mode, .huggingFace, "Sharing goes through Hugging Face; recording requires it")
        XCTAssertFalse(settings.hasConsent)
        settings.giveConsent()
        XCTAssertTrue(ContributionSettings(defaults: defaults).hasConsent)
        defaults.set("consent-v0", forKey: "contribution.consent")
        XCTAssertFalse(ContributionSettings(defaults: defaults).hasConsent, "A new consent text needs confirming again")
    }
}
