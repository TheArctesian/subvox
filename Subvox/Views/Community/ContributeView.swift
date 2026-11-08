import AppKit
import SwiftUI

struct ContributeView: View {
    @Environment(AppModel.self) private var model
    @State private var expanded: String?

    private var settings: ContributionSettings { model.contribution }
    private var uploader: CommunityUploader { model.uploader }
    private var account: HuggingFaceAccount { model.huggingFace }

    var body: some View {
        AppPageLayout {
            HStack {
                AppPageHeading(title: "Share recordings", subtitle: "Help build an open dataset for silent speech. Nothing leaves this Mac until you press Send.")
                Spacer()
                Button("Back to training") { model.showContribute = false }.disabled(uploader.busy)
            }
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    SharingSetupView()
                    if settings.mode == .huggingFace && account.username != nil && settings.hasConsent { queueSection }
                }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
            }
        }
    }

    private var queueSection: some View {
        @Bindable var settings = settings
        return VStack(alignment: .leading, spacing: 12) {
            Text("Your sessions").font(.title3.weight(.semibold))
            HStack {
                TextField("Dataset", text: $settings.repo).textFieldStyle(.roundedBorder).frame(maxWidth: 360)
                Button("Find community sessions") { if let name = account.username { Task { await uploader.prepare(contributor: name) } } }
                    .disabled(uploader.busy)
            }
            if uploader.busy { HStack { ProgressView().controlSize(.small); Text(uploader.status).font(.callout.monospaced()).lineLimit(1) } }
            else if !uploader.status.isEmpty { Text(uploader.status).font(.callout).foregroundStyle(.secondary) }
            if let error = uploader.error { Text(error).font(.callout).foregroundStyle(.orange).textSelection(.enabled) }
            ForEach(uploader.items) { item in sessionRow(item) }
        }
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
                    Button(expanded == item.id ? "Hide takes" : "Review takes") { expanded = expanded == item.id ? nil : item.id }
                    Button(item.status == .failed ? "Retry" : "Send") { send(item) }.buttonStyle(.borderedProminent)
                        .disabled(uploader.busy || item.takes == item.dropped.count)
                }
            }
            if let error = item.error, item.status == .failed { Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3) }
            if expanded == item.id {
                ForEach(uploader.takes(item)) { take in
                    Toggle(isOn: Binding(get: { !item.dropped.contains(take.id) }, set: { _ in uploader.toggleDrop(take.id, in: item.id) })) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(take.mode == "spoken" ? "Spoken" : "Mouthed"): \(take.text)")
                            Text([take.source, take.eligible ? nil : "motion not eligible", take.cameraEligible ? nil : "no camera points"]
                                .compactMap { $0 }.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                Text("Untick any take you don't want to share. Dropped takes are left out of the upload; your local recording isn't changed.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12).background(Color(nsColor: .controlBackgroundColor).opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
    }

    private func send(_ item: CommunityUpload) {
        Task {
            do { await uploader.send(item.id, repo: settings.repo, token: try await account.accessToken()) }
            catch { model.huggingFace.signOut() }
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
                Text("Withdrawing stops future uploads. To remove data already shared, ask for its removal on the dataset; copies others downloaded can't be recalled.")
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
