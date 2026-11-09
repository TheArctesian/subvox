import AppKit
import SwiftUI

struct ContributeView: View {
    @Environment(AppModel.self) private var model
    @State private var expanded: String?
    @State private var expandedTakes: [CommunityTake] = []
    @State private var review: UploadReview?
    @State private var sending = false
    @State private var sendError: String?

    private struct UploadReview: Identifiable {
        let id = UUID()
        let repo: String
        let username: String
        let sessions: [Session]

        struct Session: Identifiable {
            let id: String
            let takes: [CommunityTake]
        }
    }

    private var canShare: Bool { settings.mode == .huggingFace && account.username != nil && settings.hasConsent }
    private var working: Bool { uploader.busy || sending }

    private var settings: ContributionSettings { model.contribution }
    private var uploader: CommunityUploader { model.uploader }
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            HStack {
                AppPageHeading(title: "Share recordings", subtitle: "Help build an open dataset for silent speech. Nothing leaves this Mac until you press Send.")
                Spacer()
                Button("Back to training") { model.showContribute = false }.disabled(working)
            }
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    SharingSetupView().disabled(working)
                    LocalRecordingHistory()
                    if let sendError { Label(sendError, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                    if canShare || !uploader.unsent.isEmpty || !uploader.shared.isEmpty { queueSection }
                }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
            }
            .sheet(item: $review) { selection in
                VStack(alignment: .leading, spacing: 14) {
                    Text("Review public upload").font(.title2.weight(.semibold))
                    LabeledContent("Dataset", value: selection.repo)
                    LabeledContent("Account", value: selection.username)
                    Text("\(selection.sessions.count) sessions · \(selection.sessions.reduce(0) { $0 + $1.takes.count }) takes")
                    List(selection.sessions) { session in
                        Section("Session \(session.id.prefix(8))") {
                            ForEach(session.takes) { take in
                                Text("\(take.mode == "spoken" ? "Spoken" : "Mouthed"): \(take.text)")
                            }
                        }
                    }
                    HStack {
                        Spacer()
                        Button("Cancel") { review = nil }.keyboardShortcut(.cancelAction)
                        Button("Send to public dataset", systemImage: "arrow.up.circle") {
                            review = nil
                            send(selection)
                        }.buttonStyle(.borderedProminent).disabled(!canShare || working)
                    }
                }.padding(24).frame(width: 640, height: 460)
            }
        }
    }

    private var queueSection: some View {
        @Bindable var settings = settings
        let pending = uploader.pending.filter { belongsToCurrentAccount($0) }
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Ready to send").font(.title3.weight(.semibold))
                Spacer()
                Button { refresh() } label: { Label("Check again", systemImage: "arrow.clockwise").labelStyle(.iconOnly).frame(minWidth: 28, minHeight: 28) }.buttonStyle(.borderless).help("Check for new sessions").disabled(working || !canShare)
                Button("Review all (\(pending.count))") { prepareReview(pending) }.buttonStyle(.borderedProminent).disabled(working || !canShare || pending.isEmpty)
            }
            if !uploader.status.isEmpty { Text(uploader.status).font(.callout).foregroundStyle(.secondary) }
            if let error = uploader.error { Text(error).font(.callout).foregroundStyle(.orange).textSelection(.enabled) }
            if canShare && pending.isEmpty && !working && uploader.status.isEmpty {
                Text("Nothing new to send. Record more community passages and they'll appear here.").font(.callout).foregroundStyle(.secondary)
            }
            ForEach(uploader.unsent) { item in sessionRow(item) }
            if !uploader.shared.isEmpty {
                DisclosureGroup("Already shared (\(uploader.shared.count))") {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(uploader.shared) { item in
                            HStack {
                                Label("Session \(item.id.prefix(8)) · \(item.month) · \(item.takes - item.dropped.count) takes", systemImage: "checkmark.circle.fill")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if let url = item.prURL.flatMap(URL.init(string:)) { Link("Pull request #\(item.prNumber ?? 0)", destination: url) }
                            }.font(.callout)
                        }
                    }.padding(.top, 6)
                }
            }
            DisclosureGroup("Dataset") {
                TextField("Dataset", text: $settings.repo).textFieldStyle(.roundedBorder).frame(maxWidth: 360).disabled(working).padding(.top, 4)
            }.font(.callout)
        }
        .task(id: "\(account.username ?? "")|\(settings.hasConsent)") { refresh() }
    }

    private func refresh() {
        guard let name = account.username, settings.hasConsent, !uploader.busy else { return }
        Task { await uploader.prepare(contributor: name) }
    }

    private func belongsToCurrentAccount(_ item: CommunityUpload) -> Bool {
        let contributor = URL(fileURLWithPath: item.folder).deletingLastPathComponent().lastPathComponent
        return contributor == account.username
    }

    private func prepareReview(_ items: [CommunityUpload]) {
        guard let username = account.username, canShare, !working, items.allSatisfy(belongsToCurrentAccount) else { return }
        let sessions = items.map { item in
            UploadReview.Session(id: item.id, takes: uploader.takes(item).filter { !item.dropped.contains($0.id) })
        }
        guard !sessions.isEmpty, zip(items, sessions).allSatisfy({ item, session in
            !session.takes.isEmpty && session.takes.count == item.takes - item.dropped.count
        }) else {
            sendError = "The take list could not be verified. Check for sessions again before sharing."
            return
        }
        sendError = nil
        review = UploadReview(repo: settings.repo, username: username, sessions: sessions)
    }

    private func sessionRow(_ item: CommunityUpload) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Session \(item.id.prefix(8)) · \(item.month)").font(.headline)
                Text("\(item.takes - item.dropped.count) of \(item.takes) takes").foregroundStyle(.secondary)
                Spacer()
                switch item.status {
                case .uploaded:
                    if let url = item.prURL.flatMap(URL.init(string:)) { Link("Pull request #\(item.prNumber ?? 0)", destination: url) }
                    else { Label("Sent", systemImage: "checkmark") }
                case .uploading: ProgressView().controlSize(.small)
                case .review, .failed:
                    Button(expanded == item.id ? "Hide takes" : "Review takes") {
                        expanded = expanded == item.id ? nil : item.id
                        expandedTakes = expanded == nil ? [] : uploader.takes(item)
                    }
                    if item.takes == item.dropped.count {
                        Button("Restore all") { uploader.restoreTakes(in: item.id) }.disabled(working)
                    } else {
                        Button(item.status == .failed ? "Review retry" : "Review upload") { prepareReview([item]) }
                            .buttonStyle(.borderedProminent).disabled(working || !canShare || !belongsToCurrentAccount(item))
                    }
                }
            }
            if let error = item.error, item.status == .failed { Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3) }
            if expanded == item.id {
                ForEach(expandedTakes) { take in
                    Toggle(isOn: Binding(get: { !item.dropped.contains(take.id) }, set: { _ in uploader.toggleDrop(take.id, in: item.id) })) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(take.mode == "spoken" ? "Spoken" : "Mouthed"): \(take.text)")
                            Text([take.source, take.eligible ? nil : "motion not eligible", take.cameraEligible ? nil : "no camera points"]
                                .compactMap { $0 }.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .disabled(working)
                Text("Untick any take you don't want to share. Dropped takes are left out of the upload; your local recording isn't changed.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12).background(Color(nsColor: .controlBackgroundColor).opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
    }

    private func send(_ selection: UploadReview) {
        guard canShare, !working, account.username == selection.username else { return }
        sending = true
        Task {
            defer { sending = false }
            do {
                let token = try await account.accessToken()
                guard canShare, account.username == selection.username else { return }
                guard selection.sessions.allSatisfy({ session in
                    guard let item = uploader.pending.first(where: { $0.id == session.id }), belongsToCurrentAccount(item) else { return false }
                    let current = uploader.takes(item).filter { !item.dropped.contains($0.id) }
                    return current == session.takes
                }) else {
                    sendError = "The queue changed. Review these takes again before sending."
                    return
                }
                await uploader.sendReviewed(selection.sessions.map(\.id), repo: selection.repo, token: token)
            } catch {
                sendError = "Sign in again to send your reviewed sessions. Your queue is still here."
            }
        }
    }
}

struct LocalRecordingHistory: View {
    @Environment(AppModel.self) private var model
    @State var expanded = false

    var body: some View {
        DisclosureGroup("Saved on this Mac", isExpanded: $expanded) {
            let pairs = model.sentenceTraining.data.pairs.reversed().filter { $0.spoken != nil || $0.silent != nil }
            if pairs.isEmpty { Text("Your saved takes will appear here.").foregroundStyle(.secondary) }
            ForEach(pairs.prefix(30)) { pair in
                VStack(alignment: .leading, spacing: 3) {
                    Text(pair.text).lineLimit(2)
                    Text(pair.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption).foregroundStyle(.secondary)
                    if pair.practiceMode != nil {
                        Text("Practice recording · local only").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.vertical, 3)
            }
            if pairs.count > 30 { Text("Showing the 30 most recent recordings.").font(.caption).foregroundStyle(.secondary) }
            Text("Only eligible community readings appear in the upload queue.").font(.callout).foregroundStyle(.secondary)
        }
    }
}

/// Sharing mode, Hugging Face sign-in and consent. Shown on first launch and on the Share page.
struct SharingSetupView: View {
    @Environment(AppModel.self) private var model
    @State private var adult = false
    @State private var publicData = false
    @State private var linkedName = false

    private var settings: ContributionSettings { model.contribution }
    private var uploader: CommunityUploader { model.uploader }
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            modeSection
            if settings.mode == .huggingFace {
                accountSection
                if account.username != nil { consentSection }
            }
        }
    }

    private var modeSection: some View {
        @Bindable var settings = settings
        return VStack(alignment: .leading, spacing: 10) {
            Text("How to share").font(.title2.weight(.semibold))
            Picker("Sharing", selection: $settings.mode) {
                Text("Local only").tag(ContributionSettings.Mode.local)
                Text("Anonymous (coming soon)").tag(ContributionSettings.Mode.anonymous)
                Text("Hugging Face account").tag(ContributionSettings.Mode.huggingFace)
            }.pickerStyle(.radioGroup)
            switch settings.mode {
            case .local:
                Text("Recordings stay on this Mac. You can still record community passages for your own use.").font(.callout).foregroundStyle(.secondary)
            case .anonymous:
                Text("Anonymous sharing needs a relay server that isn't running yet. Choose Local only or Hugging Face for now.")
                    .font(.callout).foregroundStyle(.orange)
            case .huggingFace:
                Text("Each session becomes a pull request under your Hugging Face account, so you're credited as a contributor.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var accountSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Hugging Face account").font(.title3.weight(.semibold))
            switch account.state {
            case .signedIn(let name):
                HStack {
                    Label("Signed in as \(name)", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
                    Spacer()
                    Button("Sign out", action: account.signOut).disabled(uploader.busy)
                }
            case .signingIn:
                HStack { ProgressView().controlSize(.small); Text("Finish signing in in your browser…") }
                if let link = account.pendingURL {
                    Text("Didn't open, or opened a different page? Open the link again, or copy it into any browser where you're signed in to Hugging Face.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Button("Open link again", action: account.reopenSignIn)
                        Button("Copy link") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(link.absoluteString, forType: .string)
                        }
                        Button("Cancel", action: account.cancelSignIn)
                    }
                    Text(link.absoluteString).font(.caption.monospaced()).foregroundStyle(.tertiary).lineLimit(2).truncationMode(.middle).textSelection(.enabled)
                }
            case .signedOut:
                Button { Task { await account.signIn() } } label: { Label("Sign in with Hugging Face", systemImage: "person.badge.key") }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                Text("Opens huggingface.co in your browser. Subvox asks only to read your public profile and open pull requests on your behalf; it can't change your repositories or settings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = account.error { Text(error).font(.callout).foregroundStyle(.orange) }
        }
    }

    private var consentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Before you share").font(.title3.weight(.semibold))
            if settings.hasConsent {
                HStack {
                    Label("You agreed to share (\(ContributionSettings.consentVersion)).", systemImage: "checkmark.circle")
                    Spacer()
                    Button("Withdraw agreement", action: settings.withdrawConsent).disabled(uploader.busy)
                }
                Text("Withdrawing stops future uploads. To remove data you've already shared, email \(ContributionSettings.removalContact). Copies others have downloaded can't be recalled.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text("What is shared: AirPods head-motion, numeric lip and jaw points from the camera, which sentence you read, your AirPods model and firmware, and the month. Never video, images, audio, file names, device names or exact times.")
                    .fixedSize(horizontal: false, vertical: true)
                Toggle("I am 18 or older.", isOn: $adult)
                Toggle("I understand the dataset is public (CC BY 4.0): anyone can download and reuse what I share, and published copies can't be fully withdrawn.", isOn: $publicData)
                Toggle("I understand my contributions are linked to my Hugging Face username.", isOn: $linkedName)
                Button("I agree to share", action: settings.giveConsent).buttonStyle(.borderedProminent).disabled(!(adult && publicData && linkedName))
                Text("Draft wording pending a data-protection review.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
