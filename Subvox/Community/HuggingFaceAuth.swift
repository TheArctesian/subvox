import AppKit
import CryptoKit
import Foundation
import Network
import Security

/// "Sign in with Hugging Face" for a native app (RFC 8252): authorization code + PKCE (S256) with a
/// loopback redirect, opened in the user's default browser. The OAuth app is public (no client
/// secret), so the app never holds a write token of its own; it only ever acts as the signed-in user.
enum HuggingFaceOAuth {
    static let clientID = "ae146936-11cf-4196-a5ac-7288fafbf4ba"
    /// write-discussions is the documented scope for opening pull requests (docs/community-dataset-design.md, section 1).
    static let scopes = "openid profile write-discussions"
    static let authorizeURL = URL(string: "https://huggingface.co/oauth/authorize")!
    static let tokenURL = URL(string: "https://huggingface.co/oauth/token")!
    static let userinfoURL = URL(string: "https://huggingface.co/oauth/userinfo")!
    static let callbackPath = "/oauth/callback"

    static func base64url(_ data: Data) -> String {
        data.base64EncodedString().replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
    static func randomToken(bytes: Int = 32) -> String {
        var data = Data(count: bytes)
        _ = data.withUnsafeMutableBytes { SecRandomCopyBytes(kSecRandomDefault, bytes, $0.baseAddress!) }
        return base64url(data)
    }
    static func challenge(for verifier: String) -> String { base64url(Data(SHA256.hash(data: Data(verifier.utf8)))) }

    static func authorization(redirect: String, state: String, challenge: String) -> URL {
        var components = URLComponents(url: authorizeURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            .init(name: "client_id", value: clientID), .init(name: "redirect_uri", value: redirect),
            .init(name: "response_type", value: "code"), .init(name: "scope", value: scopes), .init(name: "state", value: state),
            .init(name: "code_challenge", value: challenge), .init(name: "code_challenge_method", value: "S256"),
        ]
        return components.url!
    }

    enum Failure: LocalizedError, Equatable {
        case denied(String), stateMismatch, malformed, timedOut, needsSignIn, http(Int, String)
        var errorDescription: String? {
            switch self {
            case .denied(let reason): "Hugging Face sign-in was not completed (\(reason))."
            case .stateMismatch: "The sign-in response didn't match this request, so it was ignored."
            case .malformed: "Hugging Face returned an unexpected response."
            case .timedOut: "Sign-in timed out. Try again."
            case .needsSignIn: "Sign in with Hugging Face to upload."
            case .http(let code, let body): "Hugging Face returned HTTP \(code): \(body.prefix(200))"
            }
        }
    }

    /// The authorization code from the browser's redirect request line ("GET /oauth/callback?... HTTP/1.1").
    static func code(fromRequestLine line: String, expectedState: String) throws -> String {
        let parts = line.split(separator: " ")
        guard parts.count >= 2, parts[0] == "GET", let components = URLComponents(string: "http://127.0.0.1" + parts[1]),
              components.path == callbackPath else { throw Failure.malformed }
        let items = Dictionary((components.queryItems ?? []).map { ($0.name, $0.value ?? "") }, uniquingKeysWith: { a, _ in a })
        if let error = items["error"] { throw Failure.denied(items["error_description"] ?? error) }
        guard items["state"] == expectedState else { throw Failure.stateMismatch }
        guard let code = items["code"], !code.isEmpty else { throw Failure.malformed }
        return code
    }

    static func form(_ fields: [(String, String)]) -> Data {
        var allowed = CharacterSet.alphanumerics; allowed.insert(charactersIn: "-._~")
        return Data(fields.map { "\($0)=\($1.addingPercentEncoding(withAllowedCharacters: allowed)!)" }.joined(separator: "&").utf8)
    }
    static func tokenRequest(_ fields: [(String, String)]) -> URLRequest {
        var request = URLRequest(url: tokenURL)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = form(fields + [("client_id", clientID)])
        return request
    }
    static func exchange(code: String, verifier: String, redirect: String) -> URLRequest {
        tokenRequest([("grant_type", "authorization_code"), ("code", code), ("redirect_uri", redirect), ("code_verifier", verifier)])
    }
    static func refresh(_ token: String) -> URLRequest { tokenRequest([("grant_type", "refresh_token"), ("refresh_token", token)]) }

    static func token(from data: Data, now: Date = Date()) throws -> HuggingFaceToken {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let access = json["access_token"] as? String, !access.isEmpty else { throw Failure.malformed }
        let lifetime = (json["expires_in"] as? NSNumber)?.doubleValue ?? 8 * 3600
        return HuggingFaceToken(accessToken: access, refreshToken: json["refresh_token"] as? String,
                                expiresAt: now.addingTimeInterval(lifetime), scope: json["scope"] as? String, username: nil)
    }
}

struct HuggingFaceToken: Codable, Equatable, Sendable {
    var accessToken: String
    var refreshToken: String?
    var expiresAt: Date
    var scope: String?
    var username: String?
    func expired(at now: Date = Date()) -> Bool { now.addingTimeInterval(60) >= expiresAt }
}

/// Generic-password Keychain item holding the token JSON.
protocol TokenStore: Sendable {
    func load() -> Data?
    func save(_ data: Data) throws
    func delete()
}

struct KeychainTokenStore: TokenStore {
    var service = "org.subvox.huggingface"
    var account = "oauth-token"
    private var query: [String: Any] { [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account] }
    func load() -> Data? {
        var result: AnyObject?
        let status = SecItemCopyMatching(query.merging([kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]) { $1 } as CFDictionary, &result)
        return status == errSecSuccess ? result as? Data : nil
    }
    func save(_ data: Data) throws {
        SecItemDelete(query as CFDictionary)
        let status = SecItemAdd(query.merging([kSecValueData as String: data, kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]) { $1 } as CFDictionary, nil)
        guard status == errSecSuccess else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }
    func delete() { SecItemDelete(query as CFDictionary) }
}

/// One-shot HTTP listener on 127.0.0.1 for the OAuth redirect.
final class LoopbackRedirect: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "subvox.oauth.loopback")
    private var continuation: CheckedContinuation<String, Error>?

    init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    /// Starts listening and returns the redirect URI (with the port the system assigned).
    func start() async throws -> String {
        try await withCheckedThrowingContinuation { (ready: CheckedContinuation<String, Error>) in
            var resumed = false
            listener.stateUpdateHandler = { [listener] state in
                guard !resumed else { return }
                switch state {
                case .ready:
                    resumed = true
                    ready.resume(returning: "http://127.0.0.1:\(listener.port?.rawValue ?? 0)\(HuggingFaceOAuth.callbackPath)")
                case .failed(let error):
                    resumed = true; ready.resume(throwing: error)
                default: break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in self?.handle(connection) }
            listener.start(queue: queue)
        }
    }

    /// The first request line the browser sends to the callback.
    func requestLine(timeout: Duration = .seconds(300)) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { try await withCheckedThrowingContinuation { c in self.queue.async { self.continuation = c } } }
            group.addTask { try await Task.sleep(for: timeout); throw HuggingFaceOAuth.Failure.timedOut }
            defer { group.cancelAll(); stop() }
            return try await group.next()!
        }
    }

    func stop() {
        listener.cancel()
        queue.async { self.continuation?.resume(throwing: CancellationError()); self.continuation = nil }
    }

    private func handle(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [weak self] data, _, _, _ in
            let text = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
            let line = text.components(separatedBy: "\r\n").first ?? ""
            let page = "<!doctype html><meta charset=utf-8><title>Subvox</title><body style=\"font:16px -apple-system;margin:3em\">"
                + "<h2>You can return to Subvox.</h2><p>This window can be closed.</p></body>"
            let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: \(page.utf8.count)\r\nConnection: close\r\n\r\n" + page
            connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in connection.cancel() })
            guard line.hasPrefix("GET ") else { return }
            self?.continuation?.resume(returning: line); self?.continuation = nil
        }
    }
}

/// The signed-in Hugging Face account. The token lives only in the Keychain.
@MainActor @Observable final class HuggingFaceAccount {
    enum State: Equatable { case signedOut, signingIn, signedIn(String) }
    private(set) var state: State = .signedOut
    private(set) var error: String?
    @ObservationIgnored private let store: TokenStore
    @ObservationIgnored private let session: URLSession
    @ObservationIgnored private let open: (URL) -> Void

    var username: String? { if case .signedIn(let name) = state { name } else { nil } }

    init(store: TokenStore = KeychainTokenStore(), session: URLSession = .shared, open: @escaping (URL) -> Void = { NSWorkspace.shared.open($0) }) {
        self.store = store; self.session = session; self.open = open
        if let token = stored(), let name = token.username { state = .signedIn(name) }
    }

    private func stored() -> HuggingFaceToken? { store.load().flatMap { try? JSONDecoder().decode(HuggingFaceToken.self, from: $0) } }

    func signIn() async {
        guard state != .signingIn else { return }
        state = .signingIn; error = nil
        do {
            let redirect = try LoopbackRedirect()
            let uri = try await redirect.start()
            let verifier = HuggingFaceOAuth.randomToken(bytes: 48), expected = HuggingFaceOAuth.randomToken()
            open(HuggingFaceOAuth.authorization(redirect: uri, state: expected, challenge: HuggingFaceOAuth.challenge(for: verifier)))
            let code = try HuggingFaceOAuth.code(fromRequestLine: try await redirect.requestLine(), expectedState: expected)
            var token = try HuggingFaceOAuth.token(from: try await send(HuggingFaceOAuth.exchange(code: code, verifier: verifier, redirect: uri)))
            token.username = try await username(token.accessToken)
            try store.save(try JSONEncoder().encode(token))
            state = .signedIn(token.username ?? "Hugging Face user")
        } catch {
            self.error = error.localizedDescription
            state = stored()?.username.map { .signedIn($0) } ?? .signedOut
        }
    }

    func signOut() { store.delete(); state = .signedOut; error = nil }

    /// A usable access token, refreshed when possible; otherwise the user must sign in again.
    func accessToken(now: Date = Date()) async throws -> String {
        guard var token = stored() else { state = .signedOut; throw HuggingFaceOAuth.Failure.needsSignIn }
        if !token.expired(at: now) { return token.accessToken }
        guard let refresh = token.refreshToken else { signOut(); throw HuggingFaceOAuth.Failure.needsSignIn }
        do {
            let renewed = try HuggingFaceOAuth.token(from: try await send(HuggingFaceOAuth.refresh(refresh)), now: now)
            token = HuggingFaceToken(accessToken: renewed.accessToken, refreshToken: renewed.refreshToken ?? refresh,
                                     expiresAt: renewed.expiresAt, scope: renewed.scope ?? token.scope, username: token.username)
            try store.save(try JSONEncoder().encode(token))
            return token.accessToken
        } catch { signOut(); throw HuggingFaceOAuth.Failure.needsSignIn }
    }

    private func username(_ access: String) async throws -> String? {
        var request = URLRequest(url: HuggingFaceOAuth.userinfoURL)
        request.setValue("Bearer \(access)", forHTTPHeaderField: "Authorization")
        let json = try JSONSerialization.jsonObject(with: try await send(request)) as? [String: Any]
        return json?["preferred_username"] as? String ?? json?["name"] as? String
    }

    private func send(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw HuggingFaceOAuth.Failure.http(status, String(decoding: data, as: UTF8.self)) }
        return data
    }
}
