import SwiftUI

/// The default Train page: one way to record, for someone who just installed the app.
struct SimpleTrainingView: View {
    @Environment(AppModel.self) private var model

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var motionReady: Bool { model.motion.status == .streaming }
    private var calibrationLeft: Int { workflow.calibrationRemaining.count }
    private var recordedSentences: Int { workflow.data.mouthedPromptIDs.filter { workflow.community.prompt($0) != nil }.count }

    var body: some View {
        AppPageLayout {
            HStack {
                AppPageHeading(title: "Record", subtitle: "Read a short sentence aloud, then mouth it silently. About ten minutes a session.")
                Spacer()
                Button("Share recordings…") { model.showContribute = true }.disabled(workflow.isBusy)
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
                            Button { model.startSentenceTraining() } label: {
                                Label(calibrationLeft > 0 ? "Start recording" : "Record the next passages", systemImage: "record.circle")
                                    .frame(minWidth: 220)
                            }
                            .buttonStyle(.borderedProminent).controlSize(.large).disabled(!motionReady || workflow.plan.isEmpty)
                            if let error = workflow.error { Text(error).font(.callout).foregroundStyle(.orange) }
                            Text("Recordings stay on this Mac unless you choose to share them.").font(.caption).foregroundStyle(.secondary)
                        }
                        Divider()
                        Toggle("Show research tools", isOn: Binding(get: { model.researchTools }, set: { model.researchTools = $0 }))
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
            row(ok: true, title: "The camera reads lip shape while you record", hint: "Only numeric points are kept. No video or photos are saved.")
        }
    }

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
