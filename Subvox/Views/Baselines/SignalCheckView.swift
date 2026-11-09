import AppKit
import SwiftUI

struct SignalCheckView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.colorScheme) private var scheme
    @State private var channel = MotionPlotChannel.rotation
    private var workflow: SignalCheckWorkflow { model.signalCheck }
    private var green: Color { PhaseColor.speak(scheme) }
    private var blue: Color { PhaseColor.mouth(scheme) }
    private var amber: Color { PhaseColor.relax(scheme) }
    private var cueColor: Color {
        workflow.phase == .relax ? amber : workflow.current?.condition == .normal ? green : blue
    }

    var body: some View {
        if workflow.showsCaptureCue { capture }
        else if workflow.controlledActive { controlledBody }
        else { normalBody }
    }

    private var controlledBody: some View {
        AppPageLayout {
            HStack {
                Text(workflow.allowsHeadMovement ? "Camera-assisted fit comparison" : "Controlled fit comparison").font(.largeTitle.weight(.semibold))
                Spacer()
                if !workflow.isBusy { Button("Back to signal checks", action: workflow.exitControlled) }
            }
        } content: {
            if workflow.phase == .controlSetup || workflow.phase == .controlBreak {
                ControlledSignalSetupView()
            } else if workflow.isBusy {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(workflow.processingText).font(.title3)
                    Text("Predictions stay hidden until both blocks finish.").foregroundStyle(.secondary)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let comparison = workflow.comparison {
                ScrollView { ControlledSignalResultsView(comparison: comparison).frame(maxWidth: 1000).padding(24).frame(maxWidth: .infinity) }
            }
        }
    }

    private var normalBody: some View {
        AppPageLayout {
            HStack {
                AppPageHeading(title:"Motion signal",subtitle:"Three phrases, repeated aloud and silently mouthed.")
                Spacer()
                Button("Back to recording") { model.showSignalCheck = false;model.showRepresentation=false }.disabled(workflow.isBusy)
            }
        } content: {
            if workflow.phase == .reseat { reseat.padding(24) }
            else if workflow.isBusy {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting recording…" : workflow.processingText).font(.title3)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        if workflow.run?.report != nil || workflow.reference != nil || workflow.referenceError != nil { referenceControls }
                        instructions
                        if let comparison = workflow.comparison {
                            MotionDisclosure("Latest controlled comparison") { ControlledSignalResultsView(comparison: comparison) }
                        }
                        if let error = workflow.error { Text(error).foregroundStyle(.orange) }
                        if let run = workflow.run { results(run) }
                    }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
                }
            }
        }
    }

    private var referenceControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Test after reseating").font(.title2.weight(.semibold))
            if let reference = workflow.reference, let link = workflow.referenceLink {
                Text("Reference fixed from " + reference.sourceStartedAt.formatted()).font(.headline)
                Text("9 spoken and 9 mouthed reference takes. Templates, channel scales and scoring settings are locked.")
                    .foregroundStyle(.secondary)
                Text("Reference fingerprint: " + String(link.sha256.prefix(12))).font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Freeze the completed session before removing your AirPods. The second session will be scored only against those reference recordings.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if workflow.reference != nil {
                Button("Camera teacher · train A and test B") { workflow.prepareControlled() }
                    .buttonStyle(.borderedProminent).controlSize(.large).disabled(workflow.isBusy || !workflow.storageReady)
            }
            Button(workflow.reference == nil ? "Freeze this session for reseat test" : "Simple reseat check") { workflow.prepareReseat() }
                .buttonStyle(.borderedProminent).controlSize(.large).disabled(!workflow.canPrepareReseat)
            if let error = workflow.referenceError { Text(error).foregroundStyle(.orange) }
        }
    }

    private var reseat: some View {
        VStack(spacing: 22) {
            Label("Reference frozen", systemImage: "lock.fill").font(.title.weight(.semibold)).foregroundStyle(green)
            Text("Reseat both AirPods").font(.system(size: 40, weight: .semibold))
                .multilineTextAlignment(.center)
            Text("Remove both AirPods, then put them back in naturally. Wait for motion to reconnect, then start the second session.")
                .font(.title2).multilineTextAlignment(.center)
            if let reference = workflow.reference {
                Text("Using the reference recorded " + reference.sourceStartedAt.formatted()).foregroundStyle(.secondary)
            }
            Text("The same three phrases, three times each. New takes are scored after you finish; they never update the reference.")
                .font(.title3).multilineTextAlignment(.center).foregroundStyle(.secondary)
            if model.motion.status != .streaming {
                Label("Waiting for AirPods motion", systemImage: "headphones").foregroundStyle(amber)
            }
            Button("I’ve reseated the AirPods — start second session") { model.startSignalCheck(againstReference: true) }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(model.motion.status != .streaming || model.recorder.isRecording)
            Button("Back", action: workflow.stop)
        }.padding(32).frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 18))
    }

    private var instructions: some View {
        return VStack(alignment: .leading, spacing: 12) {
            Text("Read the phrase before starting. Keep your head still and your AirPods in the same position.")
                .font(.title3)
            Text("Start take begins a three-second still baseline. Wait for START, say or mouth the phrase once, then choose Done immediately. END confirms the saved boundary. Redo replaces a mistaken take. A 12-second timeout is saved but excluded from comparisons.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 24) {
                Label("Relax", systemImage: "leaf").foregroundStyle(amber)
                Label("Speak aloud", systemImage: "speaker.wave.2").foregroundStyle(green)
                Label("Mouth silently", systemImage: "mouth").foregroundStyle(blue)
            }.font(.headline)
            ForEach(SignalCheckRun.sentences, id: \.self) { Text($0).font(.title2) }
            Text("Signal checks save numeric camera landmarks and AirPods motion only. Audio and video are not recorded.")
                .font(.callout).foregroundStyle(.secondary)
            Text("These recordings stay separate from model training. Your 25% final test remains reserved.")
                .font(.callout).foregroundStyle(.secondary)
            Button(workflow.run == nil ? "Start signal check" : "Start a separate reference collection") { model.startSignalCheck() }
                .controlSize(.large)
                .disabled(model.motion.status != .streaming || !workflow.storageReady || model.recorder.isRecording)
            if model.motion.status != .streaming { Text("Connect your AirPods to begin.").foregroundStyle(.orange) }
        }
    }

    private var capture: some View {
        let phase = workflow.phase
        return RecordingViewport {
            VStack(spacing:8) {
                HStack {
                    Text(workflow.usesCameraTeacher ? (workflow.run?.control?.block == .reseated ? "B · frozen transfer" : "A · training") : workflow.run?.reference == nil ? "Reference collection" : "Frozen-reference test").font(.headline)
                    Spacer()
                    Text("Take \(workflow.index+1) of \(workflow.plan.count)").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                }
                ProgressView(value: Double(workflow.index + (workflow.usesCameraTeacher && workflow.run?.control?.block == .reseated ? 18 : 0)), total: Double(max(1, workflow.usesCameraTeacher ? 36 : workflow.plan.count))).tint(cueColor)
            }
        } prompt: {
            VStack(spacing:8) {
                if workflow.isPaced { CameraPoseGuide(camera: workflow.camera, target: workflow.comparison?.target, compact: true, allowsMovement: workflow.allowsHeadMovement).frame(height:70).padding(.horizontal,12) }
                RecordingPromptPanel(color:cueColor) {
                    VStack(spacing:18) {
                        RecordingCueStatus(timer:phase == .ready ? "When you’re ready" : phase == .ended ? (workflow.pendingWrites > 0 ? "Saving on this Mac…" : "Saved on this Mac") : String(format:"%.1f s",workflow.secondsRemaining),
                            title:phase == .ready ? "Ready" : phase == .relax ? "Relax" : phase == .ended ? (workflow.pendingWrites > 0 ? "END · Saving" : "END · Saved") : "START · Recording",color:cueColor)
                        Label(workflow.current?.condition == .normal ? "Speak aloud" : "Mouth silently", systemImage: workflow.current?.condition == .normal ? "speaker.wave.2" : "mouth")
                            .font(.title3.weight(.medium)).foregroundStyle(cueColor)
                        WordFocusPrompt(text:workflow.currentText,activeWordIndex:workflow.isPaced ? workflow.activeWordIndex : nil,color:cueColor)
                        Text(captureHelp).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        if phase == .ended, let take = workflow.latestTake {
                            MotionDisclosure("View saved motion") { markedPlot(take.recording, title: "Saved take") }
                        }
                        if let error=workflow.error {Text(error).font(.caption).foregroundStyle(.orange)}
                    }
                }
            }
        } actions: {
            VStack(spacing:10) {
                HStack(spacing:12) {
                    Button("Stop and save",action:workflow.stop).buttonStyle(.borderless).foregroundStyle(.secondary).keyboardShortcut(.cancelAction)
                    Spacer()
                    Button(action:workflow.redo) {Label("Redo",systemImage:"arrow.counterclockwise")}.disabled(!workflow.canRedo)
                    Button(nextTitle,action:workflow.next)
                        .buttonStyle(.borderedProminent)
                        .disabled(phase == .relax || (workflow.isPaced && phase == .recording) || (workflow.isPaced && phase == .ready && (!workflow.poseReady || !workflow.mouthReady)))
                }.controlSize(.large)
                Text("← / ↑ Redo     → / ↓ Start, finish or continue").font(.caption).foregroundStyle(.tertiary)
            }
        } motion: {LiveMotionPanel(plotHeight:44)}
    }
    private var nextTitle:String {
        if workflow.phase == .ready {return "Start take"}
        if workflow.phase == .ended {
            if workflow.index+1 == workflow.plan.count {return workflow.usesCameraTeacher ? (workflow.run?.control?.block == .unchanged ? "Train and continue to B" : "Finish test") : "Finish block"}
            return "Continue"
        }
        return workflow.isPaced ? "Fixed timer running" : "Done"
    }
    private var captureHelp:String {
        switch workflow.phase {
        case .ready:return "Read first. Start take when ready, then wait for the recording cue."
        case .relax:return workflow.allowsHeadMovement ? "Relax your mouth. Move comfortably and keep your face visible." : "Keep your head and jaw still. Wait for the recording cue."
        case .ended:
            if workflow.pendingWrites > 0 { return "You can continue or redo while this take saves." }
            return workflow.latestTake?.endReason == .timeout ? "Timed out: excluded from comparisons. Redo to add a manual end marker." : "Take saved. Continue, or redo any mistake."
        default:return workflow.isPaced ? (workflow.allowsHeadMovement ? "Follow the highlighted words. Move naturally with your face visible. Recording ends automatically." : "Follow the highlighted words. Keep your head still. Recording ends automatically.") : "Read once naturally. Choose Done as soon as you finish."
        }
    }

    private func results(_ run: SignalCheckRun) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Divider()
            Text(run.reference == nil ? "Latest signal check" : "Second-session results against frozen reference").font(.title2.weight(.semibold))
            Text(run.startedAt.formatted() + " · " + (run.recordsAudio == false ? "Points and motion only" : run.audioDevice ?? "Unknown microphone")).foregroundStyle(.secondary)
            if let report = run.report {
                ForEach(report.conditions.indices, id: \.self) { i in
                    let result = report.conditions[i]
                    VStack(alignment: .leading, spacing: 6) {
                        Text(result.condition == .normal ? "Spoken phrase matching" : "Mouthed phrase matching").font(.headline)
                        if let planned = result.plannedTakes, let collected = result.collectedTakes {
                            Text("\(result.correct) / \(planned) planned takes matched against the frozen reference")
                            Text("\(result.evaluated) scored · \(collected - result.validTakes) invalid · \(planned - collected) unrecorded · \(result.rejectedTakes ?? 0) unmatched valid takes")
                                .font(.caption).foregroundStyle(.secondary)
                            Text("Time-shift control: \(result.shiftedCorrect) / \(result.shiftedEvaluated)")
                                .font(.caption).foregroundStyle(.secondary)
                        } else if result.evaluated > 0 {
                            Text("\(result.correct) / \(result.evaluated) phrases matched · time-shift control \(result.shiftedCorrect) / \(result.shiftedEvaluated)")
                        } else { Text("Not enough usable repetitions of all three phrases to compare.").foregroundStyle(.orange) }
                        if report.reference == nil {
                            Text("\(result.validTakes) usable takes. Nearest repeated phrase, with the entire repetition held out.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                if let reference = report.reference {
                    Text("Fixed reference fingerprint: " + String(reference.sha256.prefix(12))).font(.caption).foregroundStyle(.secondary)
                }
                ForEach(report.warnings, id: \.self) { Text($0).font(.callout).foregroundStyle(.secondary) }
            }
            HStack {
                Text("Recordings and boundaries").font(.title3.weight(.medium))
                Spacer()
                Picker("Motion channels", selection: $channel) {
                    ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
                }.labelsHidden().frame(maxWidth: 190)
            }
            ForEach(run.takes.filter { !$0.discarded }) { take in
                VStack(alignment: .leading, spacing: 10) {
                    Text("\(SignalCheckRun.sentences[take.prompt.sentence]) · \(take.prompt.condition == .normal ? "Spoken" : "Mouthed") · repetition \(take.prompt.repetition + 1)").font(.headline)
                    if let prediction = run.report?.predictions?.first(where: { $0.takeID == take.id }) {
                        if let sentence = prediction.predictedSentence {
                            Text("Matched: " + SignalCheckRun.sentences[sentence])
                                .foregroundStyle(sentence == prediction.expectedSentence ? green : amber)
                        }
                        if let reason = prediction.reason { Text(reason).font(.caption).foregroundStyle(.orange) }
                    }
                    markedPlot(take.recording, title: "Rotation and explicit markers")
                    let metrics = SignalCheckAnalysis.metrics(take)
                    Text(String(format: "%.2f s · %.1f Hz · largest gap %.0f ms · strongest 1%% carries %.0f%% of rotation energy", metrics.duration, metrics.sampleRate ?? 0, metrics.maximumGap * 1000, metrics.topOnePercentEnergyShare * 100))
                        .font(.caption).foregroundStyle(.secondary)
                    if let baseline = metrics.baselineRMS {
                        Text(String(format: "Rotation RMS %.4f rad/s · still baseline %.4f rad/s", metrics.rotationRMS, baseline))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(metrics.issues, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
                    if let transcript = take.transcript, let directory = run.session?.directory {
                        Text("Heard: " + transcript.text).font(.callout)
                        AudioTimingView(transcript: transcript, audioURL: directory.appending(path: "audio.caf"))
                    }
                    if let error = take.audioError { Text(error).font(.caption).foregroundStyle(.orange) }
                }.padding(.vertical, 12)
                Divider()
            }
            if let directory = run.session?.directory {
                Button("Open raw signal check") { NSWorkspace.shared.open(directory) }
            }
        }
    }

    private func markedPlot(_ recording: SentenceRecording, title: String) -> some View {
        let start = recording.cueStart - 3
        let duration = max(0.01, recording.cueEnd - start)
        return VStack(alignment: .leading, spacing: 4) {
            SignalPlot(title: title, samples: recording.samples, channels: channel.channels,
                span: start...max(start + 0.01, recording.cueEnd), minScale: channel.minScale, height: 70, showsHeader: false)
                .overlay {
                    Canvas { context, size in
                        for (time, color) in [(recording.cueStart, Color.green), (recording.cueEnd, Color.red)] {
                            let x = min(size.width - 1, max(1, (time - start) / duration * size.width))
                            var path = Path(); path.move(to: CGPoint(x: x, y: 0)); path.addLine(to: CGPoint(x: x, y: size.height))
                            context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
                        }
                    }
                }
            HStack { Text("Still baseline −3 s"); Spacer(); Text("START 0 s").foregroundStyle(green); Spacer(); Text(String(format: "END %.2f s", recording.cueEnd - recording.cueStart)).foregroundStyle(.red) }
                .font(.caption)
        }
    }
}
