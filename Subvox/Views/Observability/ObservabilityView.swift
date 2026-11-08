import SwiftUI

struct ObservabilityView: View {
    @Environment(AppModel.self) private var model
    private var workflow: ObservabilityWorkflow { model.observability }

    var body: some View {
        if workflow.isAcquiring { recordingScreen } else { page }
    }

    private var page: some View {
        AppPageLayout {
            HStack {
                AppPageHeading(title: "Mouth signal check", subtitle: "Can your AirPods tell mouth shapes apart at all?")
                Spacer()
                Button("Back to training") { model.showObservability = false }.disabled(workflow.isBusy)
            }
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    if workflow.phase == .analysing {
                        HStack(spacing: 12) { ProgressView(); Text("Scoring \(workflow.permutations) label shuffles on this Mac…") }
                    }
                    if let error = workflow.error { Text(error).foregroundStyle(.orange) }
                    if let latest = workflow.latestReport, let report = latest.report { reportView(report, session: latest) }
                    protocolSection
                    if workflow.history.count > 1 { historySection }
                }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
            }
        }
    }

    private var protocolSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("How it works").font(.title2.weight(.semibold))
            Text("Each block shows all seven gestures once, in a random order. Relax for \(Int(ObservabilityWorkflow.relaxSeconds)) s, then do the gesture during the \(Int(ObservabilityWorkflow.cueSeconds)) s cue. Everything is silent: no voice, no camera, motion only.")
            HStack(spacing: 10) {
                ForEach(ObservabilityGesture.allCases) { gesture in
                    VStack(spacing: 6) {
                        Image(systemName: gesture.symbol).font(.title2)
                        Text(gesture.title).font(.caption)
                    }.frame(maxWidth: .infinity).padding(.vertical, 10)
                        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
                }
            }
            Text("Afterwards a simple classifier is trained on all blocks but one and tested on the block left out. Its accuracy is compared with \(workflow.permutations) runs where the labels were shuffled within each block. “bat” and “pat” look the same on the mouth, so if they separate, something else is leaking in.")
                .font(.callout).foregroundStyle(.secondary)
            Stepper("\(workflow.blocks) blocks · \(workflow.blocks * ObservabilityGesture.allCases.count) takes · about \(workflow.estimatedMinutes) min",
                    value: Binding(get: { workflow.blocks }, set: { workflow.blocks = $0 }), in: ObservabilityAnalysis.minimumBlocks...30)
            Text("Keep your head still and the AirPods seated the whole session. Use ← to redo a take and → to start or continue.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Start mouth signal check") {
                    model.representation.camera.stop(); model.signalCheck.camera.stop(); model.sentenceTraining.camera.stop()
                    workflow.start(audio: model.audio, recorder: model.recorder)
                }.buttonStyle(.borderedProminent).controlSize(.large).disabled(!canStart)
                if model.motion.status != .streaming { Label("Connect your AirPods to begin", systemImage: "headphones").foregroundStyle(.orange) }
            }
        }
    }

    private var canStart: Bool {
        model.motion.status == .streaming && !workflow.isBusy && !model.recorder.isRecording &&
            !model.sentenceTraining.isBusy && !model.signalCheck.isBusy && !model.representation.isBusy && !model.sequenceLearning.isBusy
    }

    private func reportView(_ report: ObservabilityReport, session: ObservabilitySession) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text("Latest result").font(.title2.weight(.semibold))
                Spacer()
                Text(Date(timeIntervalSince1970: session.startedUnix).formatted(date: .abbreviated, time: .shortened)).foregroundStyle(.secondary)
            }
            Label(report.summary, systemImage: verdictSymbol(report.verdict))
                .font(.headline).foregroundStyle(verdictColor(report.verdict)).fixedSize(horizontal: false, vertical: true)
            ForEach(report.results) { result in contrastRow(result) }
            if !report.insufficient.isEmpty {
                Text("Not tested (fewer than \(ObservabilityAnalysis.minimumBlocks) valid blocks): " +
                     report.insufficient.compactMap { id in ObservabilityContrast.all.first { $0.id == id }?.title }.joined(separator: ", "))
                    .font(.callout).foregroundStyle(.secondary)
            }
            Text("\(session.validTakes.count) valid takes of \(session.planned.count) planned. Significance threshold p < \(String(format: "%.4f", ObservabilityContrast.alpha)) (0.05 split across three contrasts). Accuracy is balanced across gestures.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(18).background(Color(nsColor: .controlBackgroundColor).opacity(0.6), in: RoundedRectangle(cornerRadius: 16))
    }

    private func contrastRow(_ result: ObservabilityContrastResult) -> some View {
        let isControl = result.contrast.kind == .control
        let good = isControl ? !result.significant : result.significant
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: good ? "checkmark.circle.fill" : "xmark.circle").foregroundStyle(good ? .green : (isControl ? .red : .secondary))
                Text(result.contrast.title).font(.headline)
                Spacer()
                Text(String(format: "%.0f%% vs %.0f%% chance · p = %.3f", result.balancedAccuracy * 100, result.chance * 100, result.pValue))
                    .font(.callout.monospacedDigit())
            }
            Text(result.contrast.question).font(.callout).foregroundStyle(.secondary)
            ProgressView(value: result.balancedAccuracy).tint(good ? .green : .gray)
            Text("Confusion (rows = gesture, columns = guess): " + result.confusion.enumerated().map { i, row in
                "\(result.contrast.gestures[i].title) " + row.map(String.init).joined(separator: "/")
            }.joined(separator: " · ")).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
        }
    }

    private var historySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Earlier sessions").font(.title3.weight(.semibold))
            ForEach(workflow.history.dropFirst()) { session in
                HStack {
                    Text(Date(timeIntervalSince1970: session.startedUnix).formatted(date: .abbreviated, time: .shortened))
                    Text("\(session.validTakes.count) takes").foregroundStyle(.secondary)
                    if let report = session.report { Text(report.verdict.rawValue).foregroundStyle(verdictColor(report.verdict)) }
                    Spacer()
                    Button("Re-score") { workflow.reanalyse(session.id) }.disabled(workflow.isBusy)
                }.font(.callout)
            }
        }
    }

    private func verdictSymbol(_ verdict: ObservabilityReport.Verdict) -> String {
        switch verdict {
        case .articulation: "waveform.badge.checkmark"
        case .jawOnly: "arrow.up.and.down.circle"
        case .noSignal: "waveform.slash"
        case .artifact: "exclamationmark.triangle"
        case .insufficient: "questionmark.circle"
        }
    }
    private func verdictColor(_ verdict: ObservabilityReport.Verdict) -> Color {
        switch verdict {
        case .articulation: .green
        case .jawOnly: .blue
        case .noSignal, .insufficient: .secondary
        case .artifact: .red
        }
    }

    // MARK: Recording

    private var cueColor: Color {
        if workflow.takeIssue != nil { return .orange }
        return switch workflow.phase { case .relax: .orange; case .recording: .purple; case .ended: .green; default: .cyan }
    }
    private var stage: String {
        if workflow.takeIssue != nil { return "Redo required" }
        return switch workflow.phase { case .relax: "Relax"; case .recording: "Now"; case .ended: "Saved"; default: "Ready" }
    }
    private var timer: String {
        switch workflow.phase {
        case .ready: "Press → when ready"
        case .ended: "Press → for the next gesture"
        default: String(format: "%.1f s", workflow.secondsRemaining)
        }
    }

    private var recordingScreen: some View {
        RecordingViewport {
            VStack(spacing: 8) {
                HStack {
                    Text("Mouth signal check").font(.headline)
                    Spacer()
                    if let current = workflow.current { Text("Block \(current.block + 1) of \(workflow.blocks)").font(.subheadline).foregroundStyle(.secondary) }
                    Text("\(workflow.index + 1) of \(workflow.total)").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                }
                ProgressView(value: Double(workflow.index), total: Double(max(1, workflow.total))).tint(cueColor)
            }
        } prompt: {
            RecordingPromptPanel(color: cueColor) {
                VStack(spacing: 18) {
                    RecordingCueStatus(timer: timer, title: stage, color: cueColor)
                    if let gesture = workflow.current?.gesture {
                        Image(systemName: gesture.symbol).font(.system(size: 54)).foregroundStyle(cueColor)
                        Text(gesture.instruction).font(.largeTitle.weight(.semibold)).multilineTextAlignment(.center)
                    }
                    Label("Silent · keep your head still", systemImage: "speaker.slash").font(.callout).foregroundStyle(.secondary)
                }
            }
        } actions: {
            VStack(spacing: 10) {
                HStack(spacing: 12) {
                    Button("Stop and score", action: workflow.stop).buttonStyle(.borderless).foregroundStyle(.secondary)
                    Spacer()
                    Button(action: workflow.redo) { Label("Redo", systemImage: "arrow.counterclockwise") }.disabled(!workflow.canRedo)
                    Button(action: workflow.next) {
                        Label(workflow.phase == .ended ? "Next" : "Start", systemImage: "chevron.right").frame(minWidth: 120)
                    }.buttonStyle(.borderedProminent).disabled(!workflow.canAdvance || model.motion.status != .streaming)
                }.controlSize(.large)
                if let error = workflow.error { Text(error).font(.caption).foregroundStyle(.orange) }
                else { Text("Space or → start or next  ·  ← redo").font(.caption).foregroundStyle(.tertiary) }
            }
        } motion: {
            LiveMotionPanel(plotHeight: 44)
        }
    }
}
