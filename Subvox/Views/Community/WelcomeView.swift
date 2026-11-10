import AVFoundation
import SwiftUI

/// First launch, and whenever sign-in or sharing is missing. Subvox collects recordings for an
/// open research dataset first, so the app opens only after Hugging Face sign-in and a sharing grant.
struct WelcomeView: View {
    @Environment(AppModel.self) private var model
    @State private var adult = false
    @State private var agreed = false

    private var signedIn: Bool { model.huggingFace.username != nil }
    private var sharingOn: Bool { model.automaticSharingGrant != nil }
    private var shareAgreed: Bool { sharingOn || (adult && agreed) }
    private var canStart: Bool { signedIn && shareAgreed && !model.contribution.repo.trimmingCharacters(in: .whitespaces).isEmpty }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header.padding(.bottom, 8)
                WelcomeStep(number: 1, title: "Sign in with Hugging Face", done: signedIn, enabled: true) {
                    Text("Your recordings are contributed to the public dataset under your Hugging Face account.")
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    HuggingFaceSignInView()
                }
                WelcomeStep(number: 2, title: "Your languages", done: true, enabled: true) { languages }
                WelcomeStep(number: 3, title: "Share your recordings", done: signedIn && shareAgreed, enabled: signedIn) { sharing }
                requirements
                footer
            }
            .frame(maxWidth: 640, alignment: .leading).padding(.horizontal, 40).padding(.vertical, 48).frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear { model.contribution.mode = .huggingFace }
        .onChange(of: model.huggingFace.username) { adult = false; agreed = false }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 18) {
            Image(systemName: "mouth.fill").font(.system(size: 30, weight: .semibold)).foregroundStyle(.white)
                .frame(width: 64, height: 64).background(Color.accentColor.gradient, in: RoundedRectangle(cornerRadius: 16))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 6) {
                Text("Teach computers to read silent speech").font(.largeTitle.weight(.semibold)).accessibilityAddTraits(.isHeader)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Subvox records how your AirPods move while you read passages aloud and then mouth them silently, and adds those recordings to an open research dataset. You get pronunciation practice in English, French, Spanish and Mandarin along the way.")
                    .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var languages: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("I speak").font(.subheadline.weight(.semibold))
                ChipFlow {
                    ForEach(AppModel.spokenLanguageCodes, id: \.self) { code in
                        LanguageChip(title: LanguageMenu.nativeName(code), selected: model.spokenLanguages.contains(code)) {
                            model.toggleSpokenLanguage(code)
                        }
                    }
                }
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("I want to practise").font(.subheadline.weight(.semibold))
                ChipFlow {
                    ForEach(AppLanguage.allCases) { item in
                        LanguageChip(title: item.name, selected: model.practiceLanguages.contains(item)) {
                            model.togglePracticeLanguage(item)
                        }
                    }
                }
            }
            if model.spokenLanguages.count > 1 {
                @Bindable var model = model
                Picker("Hints and translations in", selection: $model.nativeLanguage) {
                    ForEach(model.spokenLanguages, id: \.self) { Text(LanguageMenu.nativeName($0)).tag($0) }
                }.fixedSize()
            }
            Text("Choose as many as you like. Switch between them from the globe menu at the top right.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var sharing: some View {
        if let grant = model.automaticSharingGrant {
            Label("Sharing new sessions with \(grant.repo)", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                Text("After each community reading or calibration session, Subvox opens one pull request on the public dataset \(model.contribution.repo).")
                    .fixedSize(horizontal: false, vertical: true)
                Text("Shared: AirPods motion, numeric lip and jaw points if the camera is on, the prompt, your headphone model and the month. Audio, images and video stay on this Mac.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Toggle("I am 18 or older.", isOn: $adult)
                Toggle("I understand these recordings can be publicly downloaded and reused under CC BY 4.0 and are linked to my Hugging Face username.", isOn: $agreed)
                Text("To remove data you've shared, email \(ContributionSettings.removalContact). Copies others have downloaded can't be recalled.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var requirements: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("To record you'll also need").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                need(model.motion.status == .streaming, "airpods", "AirPods with head tracking")
                need(AVCaptureDevice.authorizationStatus(for: .video) == .authorized, "camera", "Camera")
                need(model.localWhisper.isConfigured, "waveform", "Local Whisper")
            }
            Text("Camera frames are processed on this Mac and only numeric points are kept. Speech recognition runs locally.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.padding(.horizontal, 4)
    }

    private var footer: some View {
        HStack(spacing: 14) {
            Button {
                if model.automaticSharingGrant == nil, let username = model.huggingFace.username {
                    model.contribution.mode = .huggingFace
                    model.contribution.enableAutomaticSharing(for: username)
                }
                model.openHome()
                model.onboarded = true
            } label: { Text("Continue to Home").frame(minWidth: 140) }
                .buttonStyle(.borderedProminent).controlSize(.large).keyboardShortcut(.defaultAction)
                .disabled(!canStart)
            Text(!signedIn ? "Sign in to continue." : !shareAgreed ? "Confirm both statements to continue." : "You can stop sharing in Subvox Settings.")
                .font(.callout).foregroundStyle(.secondary)
        }.padding(.top, 4)
    }

    private func need(_ ok: Bool, _ symbol: String, _ title: LocalizedStringKey) -> some View {
        Label {
            Text(title)
        } icon: {
            Image(systemName: ok ? "checkmark.circle.fill" : symbol).foregroundStyle(ok ? Color.green : Color.secondary)
        }
        .font(.callout)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.quaternary.opacity(0.5), in: Capsule())
        .accessibilityElement(children: .combine)
        .accessibilityValue(ok ? Text("Ready") : Text("Not set up yet"))
    }
}

/// A numbered card; the number becomes a check when the step is complete.
private struct WelcomeStep<Content: View>: View {
    var number: Int
    var title: LocalizedStringKey
    var done: Bool
    var enabled: Bool
    @ViewBuilder var content: () -> Content

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle().fill(done ? Color.green : enabled ? Color.accentColor : Color.secondary.opacity(0.4))
                if done { Image(systemName: "checkmark").font(.system(size: 13, weight: .bold)) }
                else { Text("\(number)").font(.system(size: 14, weight: .semibold)) }
            }
            .foregroundStyle(.white).frame(width: 28, height: 28).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 12) {
                Text(title).font(.title3.weight(.semibold)).accessibilityAddTraits(.isHeader)
                    .accessibilityValue(done ? Text("Done") : Text(""))
                content()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 14))
        .opacity(enabled ? 1 : 0.5)
        .disabled(!enabled)
    }
}

/// A selectable capsule for multi-select language lists.
private struct LanguageChip: View {
    var title: String
    var selected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if selected { Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)) }
                Text(title)
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            .foregroundStyle(selected ? Color.white : Color.primary)
            .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.quaternary.opacity(0.6)), in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// Lays chips out left to right, wrapping onto new rows.
private struct ChipFlow: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0,
                      height: rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(0, rows.count - 1)))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> [(indices: [Int], width: CGFloat, height: CGFloat)] {
        var rows: [(indices: [Int], width: CGFloat, height: CGFloat)] = []
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            if var last = rows.last, last.width + spacing + size.width <= width {
                last.indices.append(index); last.width += spacing + size.width; last.height = max(last.height, size.height)
                rows[rows.count - 1] = last
            } else {
                rows.append(([index], size.width, size.height))
            }
        }
        return rows
    }
}
