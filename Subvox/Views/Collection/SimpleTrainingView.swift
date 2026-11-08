import AVFoundation
import SwiftUI

/// The default Train page: one way to record, for someone who just installed the app.
struct SimpleTrainingView: View {
    @Environment(AppModel.self) private var model

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var motionReady: Bool { model.motion.status == .streaming }
    private var calibrationLeft: Int { workflow.calibrationRemaining.count }
    private var recordedSentences: Int { workflow.data.mouthedPromptIDs.filter { workflow.community.prompt($0) != nil }.count }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            HStack {
                AppPageHeading(title: "Reading", subtitle: "Read a short sentence aloud, then mouth it silently. About ten minutes a session.")
                Spacer()
                Button("Share Recordings…") { model.showContribute = true }.disabled(workflow.isBusy)
                Button { model.startSentenceTraining() } label: {
                    Label(calibrationLeft > 0 ? "Start Recording" : "Record Next Passages", systemImage: "record.circle")
                }
                .buttonStyle(.borderedProminent).controlSize(.large).disabled(!motionReady || workflow.plan.isEmpty)
                .help("⌘↩")
            }
        } content: {
            if workflow.isBusy {
                VStack(spacing: 14) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting…" : workflow.processingText.isEmpty ? "Saving…" : workflow.processingText)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 28) {
                        checklist
                        progress
                        VStack(alignment: .leading, spacing: 8) {
                            if let error = workflow.error { Text(error).font(.callout).foregroundStyle(.orange) }
                        }
                        Divider()
                        Toggle("Advanced tools for researchers", isOn: Binding(get: { model.researchTools }, set: { model.researchTools = $0 }))
                            .toggleStyle(.switch).font(.callout)
                    }.frame(maxWidth: 720, alignment: .leading).padding(32).frame(maxWidth: .infinity)
                }
            }
        }
    }

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 10) {
            row(ok: motionReady, title: motionReady ? "AirPods motion is coming through" : model.motion.statusText,
                hint: motionReady ? nil : airpodsHint)
            row(ok: cameraAllowed, title: cameraAllowed ? "The camera can read lip shape" : "Camera access is needed to record",
                hint: cameraAllowed ? nil : "macOS asks the first time you record. If you said no, allow Subvox in System Settings › Privacy & Security › Camera.")
            PrivacySummary()
        }
    }

    private var cameraAllowed: Bool { AVCaptureDevice.authorizationStatus(for: .video) != .denied && AVCaptureDevice.authorizationStatus(for: .video) != .restricted }

    private var airpodsHint: String {
        switch model.motion.status {
        case .denied: "Allow Subvox in System Settings › Privacy & Security › Motion & Fitness, then reopen the app."
        case .unsupported: "This Mac can't read headphone motion."
        default:
            // macOS only streams head motion while Spatial Audio head tracking is active for the AirPods.
            model.motion.isConnected
                ? "Put both AirPods in your ears. If it stays here, play something in Music or Safari and set Spatial Audio to Head Tracked in the AirPods menu."
                : "Use AirPods Pro, AirPods (3rd generation or later) or AirPods Max as this Mac's sound output. Then play something in Music or Safari and set Spatial Audio to Head Tracked in the AirPods menu in the menu bar."
        }
    }

    private func row(ok: Bool, title: String, hint: String?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.circle").foregroundStyle(ok ? .green : .orange).font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                if let hint { Text(hint).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
            }
        }
    }

    private var progress: some View {
        let total = workflow.community.calibration.count
        return VStack(alignment: .leading, spacing: 8) {
            if calibrationLeft > 0 {
                Text("Warm-up: \(total - calibrationLeft) of \(total) sentences").font(.title3.weight(.semibold))
                ProgressView(value: Double(total - calibrationLeft), total: Double(max(1, total)))
                Text("Everyone reads the same \(total) short sentences first. After that you get passages from books, poems and speeches.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                Text("\(recordedSentences) sentences recorded").font(.title3.weight(.semibold))
                if let next = workflow.communityPlan.first?.first.flatMap({ workflow.community.prompt($0.id) }) {
                    Text("Next: \(next.title) — \(next.author)").font(.callout).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// What a recording saves, where, and what leaves the Mac, shown before the first take.
struct PrivacySummary: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("What's saved, and where", systemImage: "lock.shield").font(.headline)
            Text("On this Mac: your voice as audio, AirPods head motion, and numeric lip and jaw points from the camera. Never video or photos.")
            Text("Speech recognition runs on this Mac. Nothing leaves it unless you press Send in Share Recordings, and audio is never shared.")
            HStack {
                Text(SentenceStore.directory.path).font(.callout.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([SentenceStore.directory]) }
            }
        }
        .font(.callout).fixedSize(horizontal: false, vertical: true)
        .padding(12).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }
}
