import SwiftUI
import AppKit

struct SequenceLearningView:View {
    var collectionOnly=false
    @State private var previewMovement=NonSpeechMovement.nodAndTurn
    @Environment(AppModel.self) private var model
    private var workflow:SequenceLearningWorkflow {model.sequenceLearning}
    var body:some View {
        if workflow.isAcquiring {recordingScreen} else {regularBody}
    }
    private var regularBody:some View {
        VStack(spacing:0) {
            HStack {
                VStack(alignment:.leading,spacing:5) {
                    Text(collectionOnly ? "Record non-speech motion":"Advanced · neural model training").font(.largeTitle.weight(.semibold))
                    Text(collectionOnly ? "Teach rejection using actual non-speech movements":"Real motion → IPA phonemes and transitions → words").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Back to recording") {model.showRepresentation=false;model.showSignalCheck=false;model.showSequenceLearning=false;model.showNonSpeechCollection=false}.disabled(workflow.isAcquiring)
            }.padding(20).fixedSize(horizontal:false,vertical:true).layoutPriority(1)
            Divider()
            if collectionOnly {
                VStack(spacing:0) {
                    ScrollView {
                        VStack(alignment:.leading,spacing:18) {
                            Text("Preview a movement, then start when ready. During recording, choose Done when finished and redo any mistake.")
                            Picker("Movement example",selection:$previewMovement) {
                                ForEach(NonSpeechMovement.allCases) {movement in Text(movement.title).tag(movement)}
                            }.pickerStyle(.segmented)
                            NonSpeechMovementGuide(movement:previewMovement,color:.cyan,demonstrating:true)
                                .frame(maxWidth:.infinity).padding(.vertical,12)
                            Text(previewMovement.instruction).font(.title2).frame(maxWidth:.infinity,alignment:.center)
                            Text("Four movements, twice for each model condition. Keep speech and silent word articulation off.")
                            Text("Only AirPods motion is saved. Microphone, camera and video are off. The illustrations are examples, not a live camera feed.").foregroundStyle(.secondary)
                            Picker("Session role",selection:Binding(get:{workflow.negativeRole},set:{workflow.negativeRole=$0})) {
                                Text("Training").tag(DataRole.training);Text("Development").tag(DataRole.validation)
                            }.pickerStyle(.segmented)
                            if let error=workflow.error {Text(error).foregroundStyle(.orange)}
                            if let visit=workflow.visit,workflow.recordingNonSpeech,visit.complete {Text("\(visit.takes.filter {$0.discardedReason==nil}.count) non-speech takes saved. Include them in a new dataset snapshot; existing models stay fixed.")}
                        }.frame(maxWidth:1000).padding(24).frame(maxWidth:.infinity)
                    }.frame(minHeight:0,maxHeight:.infinity)
                    if model.motion.status != .streaming {Text("Connect your AirPods to record. You can preview the movements now.").font(.callout).foregroundStyle(.orange)}
                    HStack {
                        if workflow.resumableNonSpeech != nil {
                            Button("Resume \(workflow.resumableNonSpeechCount) remaining movements") {
                                model.representation.camera.stop();model.signalCheck.camera.stop();model.sentenceTraining.camera.stop()
                                workflow.resumeNonSpeech(audio:model.audio,recorder:model.recorder)
                            }.buttonStyle(.borderedProminent).disabled(!canPrepareCollection)
                        }
                        Button("Start a new non-speech session") {workflow.mode = .nonSpeech;startVisit()}
                            .disabled(!canCollect)
                    }.controlSize(.large).fixedSize(horizontal:false,vertical:true).padding(16).layoutPriority(1)
                }
            } else {
                ScrollView {
                    VStack(alignment:.leading,spacing:20) {
                        controls
                        if workflow.phase == .working {ProgressView();Text(workflow.status).font(.callout.monospaced()).textSelection(.enabled);Button("Stop local job",action:workflow.stop)}
                        else {Text(workflow.status).foregroundStyle(.secondary)}
                        if let error=workflow.error {Text(error).foregroundStyle(.orange).textSelection(.enabled)}
                        if let comparison=workflow.comparison {results(comparison)}
                        inference
                        collection
                        teachers
                        corrections
                        Text("This trains a small convolutional/GRU network with CTC. Sentence labels supply phone sequences, with editable pronunciation alternatives; phone boundaries are not spread over word durations. Diphone context uses a next-phone auxiliary head, not a giant pair vocabulary. Spoken and mouthed models are separate. Camera/audio are absent at inference.")
                            .font(.caption).foregroundStyle(.secondary)
                        Text("Training snapshots exclude the reserved 25% sentence final test and quarantine overlapping session/sentence cells. Prior diagnostic tests explicitly become training data for this new model when included. Development scores are used for checkpoint selection and cannot establish fresh-session accuracy. All prompts and teacher labels remain fallible.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth:1100,alignment:.leading).padding(24).frame(maxWidth:.infinity)
                }
            }
            Divider();LiveMotionPanel(plotHeight:60).padding(12).fixedSize(horizontal:false,vertical:true).layoutPriority(1)
        }
    }
    private var controls:some View {
        @Bindable var workflow=workflow
        return VStack(alignment:.leading,spacing:12) {
            Text("1 · Prepare real recordings   2 · Train and compare   3 · Test frozen models").font(.headline)
            HStack {
                Button("Prepare dataset snapshot",action:workflow.prepareDataset)
                Stepper("\(workflow.epochs) epochs",value:$workflow.epochs,in:10...200,step:10).frame(width:170)
                Button("Train neural comparisons",action:workflow.train).buttonStyle(.borderedProminent).disabled(!workflow.datasetCompatible)
            }.disabled(workflow.isBusy || model.sentenceTraining.isBusy || model.signalCheck.isBusy || model.representation.isBusy)
            Text("Compares plain CTC, diphone context, train-only augmentation, and activity supervision. PyTorch runs locally on CPU; the first run prepares a pinned local runtime. Original recordings and frozen baselines stay intact.")
                .font(.callout).foregroundStyle(.secondary)
            if let dataset=workflow.datasetURL {Button("Review dataset manifest and labels") {NSWorkspace.shared.open(dataset)}}
            if workflow.datasetURL != nil && !workflow.datasetCompatible {
                Text("This older snapshot is preserved. Prepare a new dataset snapshot to use cross-validation; saved model inference remains available.")
                    .font(.callout).foregroundStyle(.orange)
            }
            if let run=workflow.runURL {Button("Open model weights, training curves and error reports") {NSWorkspace.shared.open(run)}}
        }
    }
    private var inference:some View {
        @Bindable var workflow=workflow
        return VStack(alignment:.leading,spacing:12) {
            Divider();Text("Try motion → phonemes → words").font(.title2.weight(.semibold))
            HStack {
                Picker("Frozen model",selection:$workflow.experiment) {
                    Text("Plain CTC").tag("plain");Text("Diphone context").tag("diphone")
                    Text("Augmented").tag("augmented");Text("Activity supervised").tag("activity")
                    Text("Camera teacher").tag("teacher")
                }.frame(maxWidth:320)
                Picker("Condition",selection:$workflow.inferenceCondition) {ForEach(SpeechCondition.allCasesForSequence) {Text($0.rawValue).tag($0)}}
            }
            TextField("Optional expected sentence, for focus and evaluation after inference",text:$workflow.expectedText)
            Button("Record a sentence for the frozen model") {
                workflow.mode = .inference;startVisit()
            }.buttonStyle(.borderedProminent).disabled(!canCollect || !workflow.hasModel)
            if let p=workflow.prediction {
                Text(p.accepted ? p.text:"Rejected / uncertain").font(.title)
                Text("IPA hypotheses: "+p.phonemes.joined(separator:" ")).textSelection(.enabled)
                Text("With source-only word LM: "+(p.textWithLm.isEmpty ? "Rejected":p.textWithLm)).font(.callout)
                if let text=p.textWithTransitions {Text("With learned transitions: "+(text.isEmpty ? "Rejected":text)).font(.callout)}
                if let alternatives=p.homophones,!alternatives.isEmpty {Text("Same-phone spellings: "+alternatives.joined(separator:", ")).font(.caption)}
                if let reason=p.rejectionReason {Text(reason).font(.caption).foregroundStyle(.secondary)}
                if p.invariantFallback {Text("Unseen/unknown sensor side: frozen invariant fallback used.").font(.caption).foregroundStyle(.orange)}
                Text("Scores and margins are uncalibrated. A plausible language-model sentence is not evidence that the motion was decoded correctly.").font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(workflow.isBusy)
    }
    private var collection:some View {
        VStack(alignment:.leading,spacing:12) {
            Divider();Text("Collect more motion and test transfer").font(.title2.weight(.semibold))
            HStack {
                ForEach([SequenceLearningWorkflow.Mode.training,.development],id:\.self) {mode in
                    Button(mode.title) {workflow.mode=mode;startVisit()}
                }
            }
            HStack {
                Button("AirPods unchanged · fresh same-fit test") {workflow.mode = .sameFit;startVisit()}.disabled(!workflow.hasModel)
                Button("I’ve reseated · fresh transfer test") {workflow.mode = .reseated;startVisit()}.disabled(!workflow.hasModel)
            }
            Text("Each visit is a separate recorder session with sentence groups kept apart. Speak, then mouth the same text at your own timing. Baselines include comfortable head movement. Only motion is recorded: no microphone, camera or video. Fresh tests keep the chosen model fixed and never enter training automatically.")
                .font(.caption).foregroundStyle(.secondary)
            if !workflow.freshEvaluationSummary.isEmpty {Text(workflow.freshEvaluationSummary).font(.headline).textSelection(.enabled)}
            if let directory=workflow.collectionDirectory {Button("Open latest recording and frozen evaluation") {NSWorkspace.shared.open(directory)}}
        }.disabled(!canCollect)
    }
    private var teachers:some View {
        DisclosureGroup("Second stage · reviewed camera and voice teachers") {
            VStack(alignment:.leading,spacing:10) {
                Text("Run the real-data neural comparisons first. This separate stage adds a camera-shape auxiliary loss on training frames while retaining motion-only inference. Local Whisper proposals, optional acoustic phone alignment and review remain fallible annotations; no word-duration phone boundaries are used.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Collect voiced/camera training pairs") {prepareTeachers(.training)}
                    Button("Collect voiced/camera development pairs") {prepareTeachers(.validation)}
                }
                Text("These collection buttons enable numeric camera points and microphone audio in normal paired sentence training. Video is not recorded. Neural diagnostic visits keep audio/camera off. Review/correct the pair and pronunciation labels before preparing another dataset.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Train separate camera-supervised comparison",action:workflow.trainWithReviewedTeachers)
                    .disabled(workflow.comparison==nil || workflow.datasetURL==nil)
                Text("Unlabelled-motion pretraining is deferred until there is enough data. Anatomy simulation, synthetic anatomy and RLHF are not part of these models.").font(.caption).foregroundStyle(.secondary)
            }.disabled(workflow.isBusy)
        }
    }
    private func prepareTeachers(_ role:DataRole) {
        model.sentenceTraining.captureCameraPoints=true;model.sentenceTraining.collectionRole=role;model.sentenceTraining.presentation = .sentences
        model.showRepresentation=false;model.showSignalCheck=false;model.showSequenceLearning=false
    }
    private var corrections:some View {
        @Bindable var workflow=workflow
        return DisclosureGroup("Review/correct transcript labels") {
            Text("Use an example ID from the dataset manifest to correct its text. Changes are sidecar annotations used only by a new dataset/run; recordings and existing models are unchanged. Pronunciation variants remain editable in Representation pilot.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Example ID",text:$workflow.correctionID)
            TextField("Corrected sentence",text:$workflow.correctionText)
            Button("Save label correction",action:workflow.correctLabel).disabled(workflow.isBusy)
        }
    }
    private func results(_ report:NeuralComparison)->some View {
        VStack(alignment:.leading,spacing:10) {
            Divider();Text("Measured development errors · lower is better").font(.title2.weight(.semibold))
            if report.schema==2 {
                Text("These scores pool held-out predictions from separate fold models. The deployed model was refitted on all eligible recordings after epoch selection, so the table is not an unseen-data score of those refitted weights. Fresh recordings are still needed.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Grid(alignment:.leading,horizontalSpacing:18,verticalSpacing:8) {
                GridRow {Text("Condition / model");Text("Utterances");Text("Phone error");Text("Word error");Text("Word error + LM");Text("Rejected")}.font(.caption.weight(.semibold))
                ForEach(Array(report.comparisons.enumerated()),id:\.offset) {_,row in
                    GridRow {
                        Text(row.condition+" · "+row.experiment);Text("\(row.utterances)")
                        Text(percent(row.phonemeErrorRate));Text(percent(row.wordErrorRateNoLm));Text(percent(row.wordErrorRateLm));Text(percent(row.rejectionRate))
                    }.font(.callout)
                }
            }
            Text("WER includes rejected/missing output as errors. Small repeated phrase sets have weak uncertainty estimates; open the report for sentence-group counts, confusions, exclusions, negative false accepts and coverage. Fresh same-fit/reseated tests remain necessary.")
                .font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("Fold, transition, homophone and rejection details") {
                ForEach(Array(report.comparisons.enumerated()),id:\.offset) {_,row in
                    Text("\(row.condition) · \(row.experiment): \(row.folds ?? 0) folds · refit \(row.refitEpochs ?? 0) epochs · homophone-merged WER \(percent(row.wordErrorRateNoLmHomophonesMerged)) · transition WER \(percent(row.wordErrorRateTransitions)) · \(row.negativeExamples) scored negatives · false accepts \(percent(row.falseAcceptRate))")
                        .font(.caption).textSelection(.enabled)
                }
                Text("Merging homophones measures sound equivalence, not correct spelling. Transition assistance is reported separately from unassisted decoding. Refit weights require new recordings for an independent evaluation.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
    private func percent(_ value:Double?)->String {value.map {String(format:"%.1f%%",$0*100)} ?? "—"}
    private var canPrepareCollection:Bool {!workflow.isBusy && !model.recorder.isRecording && !model.sentenceTraining.isBusy && !model.signalCheck.isBusy && !model.representation.isBusy}
    private var canCollect:Bool {canPrepareCollection && model.motion.status == .streaming}
    private func startVisit() {
        model.representation.camera.stop();model.signalCheck.camera.stop();model.sentenceTraining.camera.stop()
        workflow.startVisit(data:model.sentenceTraining.data,audio:model.audio,recorder:model.recorder)
    }
    private var cueColor:Color {
        if workflow.phase == .relax {return .orange}
        if workflow.phase == .ended {return .green}
        if workflow.recordingNonSpeech {return workflow.phase == .ready ? .cyan:.purple}
        return workflow.condition == .normal ? .green:.blue
    }
    private var captureHelp:String {
        switch workflow.phase {
        case .ready:return workflow.recordingNonSpeech ? "Watch the example, then choose Start take when ready.":"Read the sentence, then choose Start take when ready."
        case .relax:return "Relax your mouth and wait for START."
        case .ended:return "Take saved. Choose Next take to continue, or Redo take to repeat it."
        default:return workflow.recordingNonSpeech ? "Follow the movement example. Choose Done when finished.":"Choose Done when finished. Redo any mistake."
        }
    }
    private var stageName:String {
        switch workflow.phase {case .ready:"Ready";case .relax:"Relax";case .recording:"Recording";case .ended:"Saved";default:"Ready"}
    }
    private var timerText:String {
        if workflow.phase == .ready {return "Preview the movement"}
        if workflow.phase == .ended,let take=workflow.visit?.takes.last {return String(format:"%.1f s recorded",take.recording.cueEnd-take.recording.cueStart)}
        return String(format:"%.1f s",workflow.secondsRemaining)
    }
    private var recordingScreen:some View {
        RecordingViewport {
            VStack(spacing:8) {
                HStack {
                    Text(workflow.recordingNonSpeech ? "Non-speech recording":"Sentence recording").font(.headline)
                    Spacer()
                    Text("\(workflow.displayIndex) of \(workflow.displayTotal)").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                    if let saved=workflow.visit?.carriedCompletedCount,saved>0 {Text("\(saved) saved earlier").font(.caption).foregroundStyle(.secondary)}
                }
                ProgressView(value:Double(workflow.displayIndex-1),total:Double(max(1,workflow.displayTotal))).tint(cueColor)
            }
        } prompt: {
            RecordingPromptPanel(color:cueColor) {
                VStack(spacing:16) {
                    VStack(spacing:8) {
                        Text(timerText).font(.system(size:20,weight:.medium).monospacedDigit()).foregroundStyle(.secondary)
                        HStack(spacing:7) {Circle().fill(cueColor).frame(width:6,height:6);Text(stageName).font(.system(size:13,weight:.semibold))}
                            .padding(.horizontal,13).padding(.vertical,7).background(cueColor.opacity(0.12),in:Capsule())
                    }
                    if let movement=workflow.currentMovement {
                        NonSpeechMovementGuide(movement:movement,color:cueColor,demonstrating:workflow.phase == .ready || workflow.phase == .recording)
                        Label("No speaking or mouthing words",systemImage:"speaker.slash").font(.callout).foregroundStyle(.secondary)
                    } else {
                        Text(workflow.condition == .normal ? "Speak aloud":"Mouth silently").font(.title2.weight(.semibold)).foregroundStyle(cueColor)
                        WordFocusPrompt(text:workflow.currentText.isEmpty ? "Mouth any sentence":workflow.currentText,activeWordIndex:workflow.activeWordIndex,color:.primary)
                    }
                    Text(captureHelp).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
            }
        } actions: {
            VStack(spacing:10) {
                HStack(spacing:12) {
                    Button("Stop and save",action:workflow.stop).buttonStyle(.borderless).foregroundStyle(.secondary)
                    Spacer()
                    Button(action:workflow.redo) {Label("Redo",systemImage:"arrow.counterclockwise")}.disabled(!workflow.canRedo)
                    Button(action:workflow.next) {
                        Label(workflow.phase == .ready ? "Start movement":workflow.phase == .recording ? "Done":"Continue",
                              systemImage:workflow.phase == .recording ? "checkmark":"chevron.right").frame(minWidth:130)
                    }.buttonStyle(.borderedProminent).tint(.accentColor)
                        .disabled(workflow.phase == .relax || (workflow.phase == .ready && model.motion.status != .streaming))
                }.controlSize(.large)
                if let error=workflow.error {Text(error).font(.caption).foregroundStyle(.orange)}
                else if workflow.phase == .ready,model.motion.status != .streaming {Label("Connect your AirPods to begin",systemImage:"headphones").font(.caption).foregroundStyle(.orange)}
                else {Text("← Redo     → Start, finish or continue").font(.caption).foregroundStyle(.tertiary)}
            }
        } motion: {
            LiveMotionPanel(plotHeight:44)
        }
    }

}

extension SpeechCondition {static var allCasesForSequence:[Self] {[.normal,.silent]}}
