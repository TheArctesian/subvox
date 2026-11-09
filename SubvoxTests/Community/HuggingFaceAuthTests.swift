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
        StubProtocol.handler = { _ in (200, Data(#"{"access_token":"new","expires_in":28800}"#.utf8)) }
        let account = HuggingFaceAccount(store: store, session: URLSession(configuration: configuration), open: { _ in })
        XCTAssertEqual(account.username, "stephenokita")
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

    @MainActor func testSharingDefaultsToLocalAndConsentIsVersioned() {
        let defaults = UserDefaults(suiteName: "subvox-test-\(UUID())")!
        let settings = ContributionSettings(defaults: defaults)
        XCTAssertEqual(settings.mode, .local)
        XCTAssertFalse(settings.hasConsent)
        settings.giveConsent()
        XCTAssertTrue(ContributionSettings(defaults: defaults).hasConsent)
        defaults.set("consent-v0", forKey: "contribution.consent")
        XCTAssertFalse(ContributionSettings(defaults: defaults).hasConsent, "A new consent text needs confirming again")
    }
}
