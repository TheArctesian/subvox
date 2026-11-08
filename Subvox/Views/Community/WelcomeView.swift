import SwiftUI

/// First launch: what Subvox does, and whether to share recordings, before anything else.
struct WelcomeView: View {
    @Environment(AppModel.self) private var model

    private var ready: Bool {
        switch model.contribution.mode {
        case .local: true
        case .anonymous: false
        case .huggingFace: model.huggingFace.username != nil && model.contribution.hasConsent
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome to Subvox").font(.largeTitle.weight(.semibold))
                    Text("Subvox is an open research project teaching computers to understand silently mouthed words, from AirPods head motion and the shape of your lips.")
                        .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                VStack(alignment: .leading, spacing: 12) {
                    Label("You read short sentences aloud, then mouth them silently. About ten minutes a session.", systemImage: "text.bubble")
                    Label("AirPods motion and numeric lip points are recorded. Never video or photos.", systemImage: "lock")
                    Label("Everything stays on this Mac unless you choose to share it.", systemImage: "desktopcomputer")
                }.font(.body)
                Divider()
                SharingSetupView()
                HStack {
                    Button(model.contribution.mode == .huggingFace ? "Continue" : "Start using Subvox") { model.onboarded = true }
                        .buttonStyle(.borderedProminent).controlSize(.large).disabled(!ready)
                    if model.contribution.mode != .local {
                        Button("Skip for now") { model.contribution.mode = .local; model.onboarded = true }.buttonStyle(.borderless)
                    }
                }
                Text("You can change this any time from Record › Share recordings.").font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: 680, alignment: .leading).padding(40).frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
