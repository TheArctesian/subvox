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

    private var canShare: Bool {
        settings.mode == .huggingFace && account.username != nil && settings.hasConsent && model.automaticSharingGrant == nil
    }
    private var working: Bool { uploader.busy || sending }

    private var settings: ContributionSettings { model.contribution }
    private var uploader: CommunityUploader { model.uploader }
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            AppPageHeading(title: "Sharing Settings", subtitle: "Manage your Hugging Face account and the recordings you submit.")
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    SharingSetupView(showsDataset: true)
                    LocalRecordingHistory()
                    if let sendError { Label(sendError, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                    if canShare || model.automaticSharingGrant != nil || !uploader.unsent.isEmpty || !uploader.shared.isEmpty { queueSection }
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
        let unsent = uploader.unsent.filter(belongsToCurrentAccount)
        let pending = unsent.filter { !$0.isLibraryPilot && $0.takes > $0.dropped.count }
        let submitted = uploader.shared.filter(belongsToCurrentAccount)
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(model.automaticSharingGrant == nil ? "Ready to send" : "Community submissions").font(.title3.weight(.semibold))
                Spacer()
                if canShare {
                    Button("Check for sessions", systemImage: "arrow.clockwise", action: refresh)
                        .disabled(working)
                    Button("Review all (\(pending.count))") { prepareReview(pending) }
                        .buttonStyle(.borderedProminent).disabled(working || pending.isEmpty)
                }
            }
            if !uploader.status.isEmpty { Text(uploader.status).font(.callout).foregroundStyle(.secondary) }
            if let error = uploader.error { Text(error).font(.callout).foregroundStyle(.orange).textSelection(.enabled) }
            if canShare && pending.isEmpty && !working && uploader.status.isEmpty {
                Text("Nothing new to send. Record more community passages and they'll appear here.").font(.callout).foregroundStyle(.secondary)
            }
            if model.automaticSharingGrant != nil {
                Text("To review community reading takes before sending, turn off automatic sharing above.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(unsent) { item in sessionRow(item) }
            if !submitted.isEmpty {
                MotionDisclosure("Submitted pull requests (\(submitted.count))") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Open a pull request on Hugging Face to edit its details or close it.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(submitted) { item in
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Label(item.isLibraryPilot ? "Library session submitted" : "Community session submitted", systemImage: "arrow.up.circle.fill")
                                    Text("\(item.month) · \(item.takes - item.dropped.count) takes · \(item.id.prefix(8))")
                                }
                                .foregroundStyle(.secondary)
                                Spacer()
                                if let url = item.prURL.flatMap(URL.init(string:)) {
                                    Link(destination: url) {
                                        Label(item.prNumber.map { String(localized: "Open PR #\($0)") } ?? String(localized: "Open pull request"),
                                              systemImage: "arrow.up.right")
                                    }
                                }
                            }.font(.callout)
                        }
                    }.padding(.top, 6)
                }
            }
        }
        .task(id: canShare ? account.username : nil) { refresh() }
    }

    private func refresh() {
        guard let name = account.username, canShare, !working else { return }
        Task { await uploader.prepare(contributor: name) }
    }

    private func belongsToCurrentAccount(_ item: CommunityUpload) -> Bool {
        let contributor = URL(fileURLWithPath: item.folder).deletingLastPathComponent().lastPathComponent
        return contributor == account.username
    }

    private func prepareReview(_ items: [CommunityUpload]) {
        guard let username = account.username, canShare, !working,
              items.allSatisfy({ !$0.isLibraryPilot && belongsToCurrentAccount($0) }) else { return }
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
        let pilotActive = item.isLibraryPilot && item.libraryGrantID == model.librarySharingGrant?.id
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(item.isLibraryPilot ? "Library session · \(item.month)" : "Session \(item.id.prefix(8)) · \(item.month)")
                    .font(.headline)
                if item.isLibraryPilot {
                    Text("\((item.takes - item.dropped.count) / 2) eligible sections").foregroundStyle(.secondary)
                } else {
                    Text("\(item.takes - item.dropped.count) of \(item.takes) takes").foregroundStyle(.secondary)
                }
                Spacer()
                switch item.status {
                case .uploaded:
                    if let url = item.prURL.flatMap(URL.init(string:)) { Link("Pull request #\(item.prNumber ?? 0)", destination: url) }
                    else { Label("Sent", systemImage: "checkmark") }
                case .uploading: ProgressView().controlSize(.small)
                case .review, .failed:
                    let reviewTitle: LocalizedStringKey = item.isLibraryPilot
                        ? (expanded == item.id ? "Hide sections" : "Review sections")
                        : (expanded == item.id ? "Hide takes" : "Review takes")
                    Button(reviewTitle) {
                        expanded = expanded == item.id ? nil : item.id
                        expandedTakes = expanded == nil ? [] : uploader.takes(item)
                    }
                    if item.isLibraryPilot {
                        if !pilotActive { Text("Sharing ended").foregroundStyle(.secondary) }
                        else if item.takes == item.dropped.count {
                            Button("Restore sections") { uploader.restoreTakes(in: item.id) }.disabled(working)
                        } else if item.status == .failed { Text("Will retry").foregroundStyle(.secondary) }
                    } else if model.automaticSharingGrant != nil {
                        if item.status == .failed { Text("Will retry").foregroundStyle(.secondary) }
                    } else if item.takes == item.dropped.count {
                        Button("Restore all") { uploader.restoreTakes(in: item.id) }.disabled(working)
                    } else {
                        Button(item.status == .failed ? "Review retry" : "Review upload") { prepareReview([item]) }
                            .buttonStyle(.borderedProminent).disabled(working || !canShare || !belongsToCurrentAccount(item))
                    }
                }
            }
            if let error = item.error, item.status == .failed { Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3) }
            if expanded == item.id {
                if item.isLibraryPilot {
                    ForEach(expandedTakes.filter { $0.mode == "spoken" }) { take in
                        Toggle(isOn: includeTake(take, in: item)) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(take.text)
                                if let source = take.source { Text(source).font(.caption).foregroundStyle(.secondary) }
                            }
                        }
                    }
                    .disabled(working || !pilotActive)
                    Text("Each section includes both the spoken and mouthed takes. Your local recordings stay on this Mac.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    ForEach(expandedTakes) { take in
                        Toggle(isOn: includeTake(take, in: item)) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(take.mode == "spoken" ? "Spoken" : "Mouthed"): \(take.text)")
                                Text([take.source, take.eligible ? nil : "motion not eligible", take.cameraEligible ? nil : "no camera points"]
                                    .compactMap { $0 }.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .disabled(working || model.automaticSharingGrant != nil)
                    Text(model.automaticSharingGrant == nil
                         ? "Untick any take you don't want to share. Dropped takes are left out of the upload; your local recording isn't changed."
                         : "Turn off automatic sharing to change which takes are included.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(12).background(Color(nsColor: .controlBackgroundColor).opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
    }

    private func includeTake(_ take: CommunityTake, in item: CommunityUpload) -> Binding<Bool> {
        Binding(get: { !item.dropped.contains(take.id) }, set: { _ in uploader.toggleDrop(take.id, in: item.id) })
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
                await uploader.sendReviewed(selection.sessions.map(\.id), repo: selection.repo, token: token) {
                    canShare && account.username == selection.username && settings.repo == selection.repo
                }
            } catch {
                sendError = "Sign in again to send your reviewed sessions. Your queue is still here."
            }
        }
    }
}

struct LocalRecordingHistory: View {
    @Environment(AppModel.self) private var model
    @State var expanded = false
    @State private var playback = SpokenTakePlayer()

    var body: some View {
        DisclosureGroup("Saved on this Mac", isExpanded: $expanded.motion(.disclosure)) {
            if expanded { recordings }
        }
        .onChange(of: expanded) { if !expanded { playback.stop() } }
        .modifier(SpokenPlaybackLifecycle(playback: playback))
    }

    private var recordings: some View {
        let workflow = model.sentenceTraining
        let pairs = workflow.data.pairs.reversed().filter { $0.spoken != nil || $0.silent != nil }
        let uploads = Dictionary(model.uploader.items.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        return VStack(alignment: .leading, spacing: 8) {
            if pairs.isEmpty { Text("Your saved takes will appear here.").foregroundStyle(.secondary) }
            if let playbackError = playback.error { Text(playbackError).font(.caption).foregroundStyle(.orange) }
            ForEach(pairs.prefix(30)) { pair in
                let title = title(for: pair)
                let library = libraryReceipt(for: pair, uploads: uploads)
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(title ?? pair.text).font(.headline).lineLimit(2)
                        Spacer()
                        if let clip = SpokenTakePlayer.clip(for: pair) {
                            Button(playback.playing == pair.id ? "Stop" : "Play spoken take",
                                   systemImage: playback.playing == pair.id ? "stop.fill" : "play.fill") {
                                model.referenceSpeaker.stop()
                                playback.toggle(clip, id: pair.id)
                            }
                            .disabled(!workflow.canStartRecording)
                        }
                    }
                    if title != nil { Text(pair.text).lineLimit(2) }
                    Text(pair.createdAt.formatted(date: .abbreviated, time: .shortened))
                    HStack {
                        Text(status(for: pair, uploads: uploads, library: library))
                        if let library, library.status == .uploaded,
                           let url = library.prURL.flatMap(URL.init(string:)) {
                            Link("Open pull request", destination: url)
                        }
                    }
                }
                .font(.caption).padding(.vertical, 4)
            }
            if pairs.count > 30 { Text("Showing the 30 most recent recordings.").font(.caption).foregroundStyle(.secondary) }
        }
    }

    private func title(for pair: SentencePair) -> String? {
        if pair.practiceMode?.isLiterature == true {
            let language = AppLanguage(rawValue: pair.language ?? "en") ?? .en
            for piece in model.practice.pieces + model.personalPracticePieces
            where piece.practiceKind == pair.practiceMode && piece.language == language.corpus
                && (pair.literaturePieceID == nil || pair.literaturePieceID == piece.id) {
                let parts = piece.parts(language: language)
                if let index = parts.firstIndex(where: {
                    $0.map { $0.text(language) }.joined(separator: " ") == pair.text
                }) {
                    return parts.count == 1 ? piece.title(language)
                        : String(localized: "\(piece.title(language)) · Part \(index + 1) of \(parts.count)")
                }
            }
        }
        guard pair.practiceMode == nil, pair.promptIDs.count == 1,
              let prompt = model.sentenceTraining.community.prompt(pair.promptIDs[0]) else { return nil }
        return prompt.passageID == nil ? String(localized: "Community calibration") : prompt.title
    }

    private func libraryReceipt(for pair: SentencePair, uploads: [String: CommunityUpload]) -> CommunityUpload? {
        guard pair.practiceMode?.isLiterature == true,
              let session = pair.recordingSessionID, let spoken = pair.spoken, let silent = pair.silent,
              let item = uploads[CommunityUploader.publicSessionID(session)],
              let exported = item.libraryTakeIDs else { return nil }
        let takeIDs = [CommunityUploader.publicTakeID(spoken.id), CommunityUploader.publicTakeID(silent.id)]
        guard takeIDs.allSatisfy({ exported.contains($0) && !item.dropped.contains($0) }) else { return nil }
        return item
    }

    private func status(for pair: SentencePair, uploads: [String: CommunityUpload], library: CommunityUpload?) -> String {
        let checking: String
        if model.sentenceTraining.checkingPairIDs.contains(pair.id) {
            checking = String(localized: "Checking")
        } else if pair.needsBackgroundCheck == true {
            checking = String(localized: "Check pending")
        } else if pair.error != nil {
            checking = String(localized: "Needs review")
        } else if pair.transcript != nil || pair.alignment != nil {
            checking = String(localized: "Checked")
        } else {
            checking = String(localized: "Saved")
        }
        if let library {
            if library.status != .uploaded, library.libraryGrantID != model.librarySharingGrant?.id {
                return String(localized: "\(checking) · Library sharing ended")
            }
            switch library.status {
            case .uploaded: return String(localized: "\(checking) · Submitted as a pull request")
            case .uploading: return String(localized: "\(checking) · Uploading section")
            case .failed: return String(localized: "\(checking) · Eligible section, upload failed")
            case .review: return String(localized: "\(checking) · Eligible section, queued for sharing")
            }
        }
        guard pair.practiceMode == nil, pair.role != .finalTest, pair.promptIDs.count == 1,
              model.sentenceTraining.community.prompt(pair.promptIDs[0]) != nil,
              let session = pair.recordingSessionID else {
            return String(localized: "\(checking) · Local only")
        }
        let upload = uploads[CommunityUploader.publicSessionID(session)]
        switch upload?.status {
        case .uploaded: return String(localized: "\(checking) · Session submitted")
        case .uploading: return String(localized: "\(checking) · Session uploading")
        case .failed: return String(localized: "\(checking) · Session upload failed")
        case .review: return String(localized: "\(checking) · Session in sharing queue")
        case nil: return String(localized: "\(checking) · Community session")
        }
    }

}

/// Sharing mode, Hugging Face sign-in and consent. Shown on first launch and in Settings.
struct SharingSetupView: View {
    @Environment(AppModel.self) private var model
    var showsDataset = false
    @State private var adult = false
    @State private var confirmAutomatic = false
    @State private var confirmStop = false
    @State private var confirmLibrary = false

    private var settings: ContributionSettings { model.contribution }
    private var uploader: CommunityUploader { model.uploader }
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            accountSection
            if account.username != nil {
                if showsDataset { datasetSection }
                automaticSharingSection
                if showsDataset, let grant = model.automaticSharingGrant { libraryPilotSection(grant) }
            }
        }
    }

    private func libraryPilotSection(_ grant: ContributionSettings.AutomaticGrant) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Library sharing pilot").font(.title3.weight(.semibold))
            Text("Share new English recordings of Invictus and the Gettysburg Address excerpt. Subvox opens a pull request on \(grant.repo) under \(grant.username) when at least one section has both a spoken and a mouthed take from the same recording session and passes recording and alignment checks.")
                .fixedSize(horizontal: false, vertical: true)
            if model.librarySharingGrant != nil {
                HStack {
                    Label("Library sharing is on", systemImage: "arrow.up.circle.fill")
                    Spacer()
                    Button("Stop library sharing", action: model.stopLibrarySharing)
                }
                Text("Turning this off stops new and queued library submissions. A submission already in progress may finish.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                Text("The public dataset receives AirPods motion, numeric points for the lips, jaw, and nose if recorded, prompt and pronunciation labels, language and headphone details, and the recording month. Audio, images, and video stay on this Mac.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Toggle("I agree to share future pilot recordings under CC BY 4.0, linked to my Hugging Face username.", isOn: $confirmLibrary)
                Button("Enable library sharing", systemImage: "arrow.up.circle") {
                    model.enableLibrarySharing()
                    confirmLibrary = false
                }
                .buttonStyle(.borderedProminent)
                .disabled(!confirmLibrary || uploader.busy || model.navigationLocked)
            }
        }
        .onChange(of: grant) { confirmLibrary = false }
        .onDisappear { confirmLibrary = false }
    }

    private var accountSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Hugging Face account").font(.title3.weight(.semibold))
            HuggingFaceSignInView()
        }
    }

    private var datasetSection: some View {
        @Bindable var settings = settings
        return VStack(alignment: .leading, spacing: 8) {
            Text("Pull request destination").font(.title3.weight(.semibold))
            TextField("Dataset", text: $settings.repo, prompt: Text("owner/dataset"))
                .textFieldStyle(.roundedBorder).frame(maxWidth: 360)
                .disabled(uploader.busy || model.automaticSharingGrant != nil)
            if model.automaticSharingGrant != nil {
                Text("Turn off automatic sharing to change the destination.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var automaticSharingSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let grant = model.automaticSharingGrant {
                HStack {
                    Label("Automatic sharing is on", systemImage: "arrow.up.circle.fill")
                    Spacer()
                    Button("Stop sharing…") { confirmStop = true }.disabled(model.navigationLocked)
                }
                Text("New community reading and calibration sessions are submitted to \(grant.repo) after you finish recording. Subvox collects recordings for research, so practice needs sharing on. A submission already in progress may finish.")
                    .font(.callout).foregroundStyle(.secondary)
            } else if let username = account.username {
                Text("Share future sessions automatically").font(.title3.weight(.semibold))
                Text("When you finish a new community reading or calibration session, Subvox will open one pull request on the public dataset \(settings.repo) under \(username). Previous recordings and personal practice stay outside automatic sharing.")
                    .fixedSize(horizontal: false, vertical: true)
                Text("The submission includes AirPods motion, numeric lip and jaw points if recorded, prompt and training labels, headphone details, and the recording month. Audio, images, and video stay on this Mac.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Toggle("I am 18 or older.", isOn: $adult)
                Toggle("I understand these recordings can be publicly downloaded and reused under CC BY 4.0 and linked to my Hugging Face username.", isOn: $confirmAutomatic)
                Button("Enable automatic sharing", systemImage: "arrow.up.circle") {
                    settings.enableAutomaticSharing(for: username)
                    confirmAutomatic = false
                    Task { await model.sendAutomaticContributions() }
                }.buttonStyle(.borderedProminent).disabled(!adult || !confirmAutomatic || uploader.busy)
                Text("You can turn this off in Subvox Settings. Copies others have downloaded can’t be recalled.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .confirmationDialog("Stop sharing recordings?", isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop sharing", role: .destructive, action: model.stopAutomaticSharing)
        } message: {
            Text("Subvox will return to the welcome screen until you turn sharing back on. Recordings already shared stay in the public dataset.")
        }
        .onChange(of: account.username) { adult = false; confirmAutomatic = false }
        .onChange(of: settings.repo) { confirmAutomatic = false }
    }

}

/// Sign in to Hugging Face, including the browser hand-off. Used by Welcome and Sharing Settings.
struct HuggingFaceSignInView: View {
    @Environment(AppModel.self) private var model
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch account.state {
            case .signedIn(let name):
                HStack {
                    Label("Signed in as \(name)", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
                    Spacer()
                    Button("Sign out", action: model.signOutOfHuggingFace).disabled(model.uploader.busy || model.navigationLocked)
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
                Text("Signed out").font(.callout).foregroundStyle(.secondary)
                Button { Task { await account.signIn() } } label: { Label("Sign in with Hugging Face", systemImage: "person.badge.key") }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                Text("Opens huggingface.co in your browser. Subvox asks only to read your public profile and open pull requests on your behalf; it can't change your repositories or settings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = account.error { Text(error).font(.callout).foregroundStyle(.orange) }
        }
    }
}
