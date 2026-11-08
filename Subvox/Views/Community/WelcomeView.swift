import AVFoundation
import SwiftUI

/// First launch: what Subvox does for a learner, which languages, what's needed, and (optionally)
/// sharing for research. Nothing here blocks starting: sharing can be set up later.
struct WelcomeView: View {
    @Environment(AppModel.self) private var model
    @State private var goal = Goal.practise
    @State private var showSharing = false

    enum Goal: Hashable { case practise, contribute }

    private var sharingReady: Bool {
        switch model.contribution.mode {
        case .local: true
        case .anonymous: false
        case .huggingFace: model.huggingFace.username != nil && model.contribution.hasConsent
        }
    }

    var body: some View {
        @Bindable var model = model
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome to Subvox").font(.largeTitle.weight(.semibold)).accessibilityAddTraits(.isHeader)
                    Text("Practise pronunciation in English, French, Spanish and Mandarin: hear each sound, say words and phrases, read poems and speeches. You can also help an open research project that teaches computers to understand silently mouthed words.")
                        .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Your languages").font(.headline).accessibilityAddTraits(.isHeader)
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                        GridRow {
                            Text("I speak")
                            Picker("I speak", selection: $model.nativeLanguage) {
                                ForEach(["en", "fr", "es", "zh"], id: \.self) { Text(LanguageMenu.nativeName($0)).tag($0) }
                            }.labelsHidden().fixedSize()
                        }
                        GridRow {
                            Text("I want to practise")
                            Picker("I want to practise", selection: $model.language) {
                                ForEach(AppLanguage.allCases) { Text($0.name).tag($0) }
                            }.labelsHidden().fixedSize()
                        }
                    }
                    Text("You can change these any time from the globe menu at the top right.").font(.callout).foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("What you'll need").font(.headline).accessibilityAddTraits(.isHeader)
                    need(ok: model.motion.isConnected || model.motion.status == .streaming, symbol: "airpods",
                         title: "AirPods Pro, AirPods (3rd generation or later) or AirPods Max",
                         detail: "For recording. Listening to sounds and reading guides works without them.")
                    need(ok: AVCaptureDevice.authorizationStatus(for: .video) != .denied, symbol: "camera",
                         title: "The Mac's camera",
                         detail: "Reads lip points while you record. Never video or photos.")
                    need(ok: model.localWhisper.isConfigured, symbol: "waveform",
                         title: "Speech recognition on this Mac (a local Whisper model)",
                         detail: "Checks what you said. Nothing is uploaded.")
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("What would you like to do first?").font(.headline).accessibilityAddTraits(.isHeader)
                    Picker("What would you like to do first?", selection: $goal) {
                        Text("Practise my pronunciation").tag(Goal.practise)
                        Text("Contribute recordings to research").tag(Goal.contribute)
                    }.pickerStyle(.radioGroup).labelsHidden()
                }

                DisclosureGroup(isExpanded: $showSharing) {
                    SharingSetupView().padding(.top, 8)
                } label: {
                    Text("Share recordings for research (optional)").font(.headline)
                }

                HStack {
                    Button("Start") {
                        if !sharingReady { model.contribution.mode = .local }  // finish sharing setup later
                        model.trainMode = goal == .practise ? .sounds : .reading
                        model.onboarded = true
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large).keyboardShortcut(.defaultAction)
                    if !sharingReady {
                        Text("Recordings stay on this Mac until you finish setting up sharing.").font(.callout).foregroundStyle(.secondary)
                    }
                }
                Text("You can change sharing any time with Share Recordings… on the Reading page.").font(.callout).foregroundStyle(.secondary)
            }.frame(maxWidth: 680, alignment: .leading).padding(40).frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func need(ok: Bool, symbol: String, title: LocalizedStringKey, detail: LocalizedStringKey) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol).frame(width: 22).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(title)
                    Image(systemName: ok ? "checkmark.circle.fill" : "circle.dashed").foregroundStyle(ok ? .green : .secondary)
                        .accessibilityLabel(ok ? "Ready" : "Not set up yet")
                }
                Text(detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
