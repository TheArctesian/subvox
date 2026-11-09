import SwiftUI

struct RepresentationStudyView:View {
    @Environment(AppModel.self) private var model
    private var workflow:RepresentationWorkflow {model.representation}
    @State private var word=""
    @State private var variants=""
    var body:some View {
        if workflow.isAcquiring { capture } else { mainBody }
    }
    private var mainBody:some View {
        AppPageLayout {
            HStack {
                AppPageHeading(title:"Frozen-model studies",subtitle:"Record A, freeze the model, then reseat and test B.")
                Spacer()
                Button("Back to recording") {model.showRepresentation=false}.disabled(workflow.isBusy)
                Menu("Advanced") {
                    Button("Sentence models") {model.showSequenceLearning=true}
                    Button("Paired sentences with local Whisper") {model.showRepresentation=false}
                    Button("Motion signal checks") {model.showRepresentation=false;model.showSignalCheck=true}
                }.disabled(workflow.isBusy)
            }
        } content: {
            ScrollView {
                    VStack(alignment:.leading,spacing:20) {
                        if workflow.phase == .processing {
                            HStack { ProgressView().controlSize(.small); Text("Evaluating saved takes on this Mac…") }
                                .foregroundStyle(.secondary)
                        }
                        if workflow.phase == .reseat {reseat}
                        else if workflow.phase != .processing {
                            if workflow.canFitSavedTraining {
                                Text("All training takes are saved. Fit them to continue; no re-recording is needed.").font(.headline)
                                Button("Fit saved A and continue to B",action:workflow.fitSavedTraining)
                                    .buttonStyle(.borderedProminent).controlSize(.large)
                            }
                            if workflow.study?.reference != nil,workflow.study?.complete==false {
                                Button("Restart interrupted B as a new session",action:workflow.prepareReplacementB)
                                    .buttonStyle(.borderedProminent)
                            }
                            Text("Keep raw motion; add supervision alongside it").font(.title2.weight(.semibold))
                            Text("A records four training sentences twice, spoken and mouthed. B uses four different validation sentences in a fresh session after reseating. There are 16 takes per block. Follow the highlighted words, then choose Done. Head movement is allowed.")
                            Text("All four arms share the same fixed decoder: raw motion; raw + camera-taught features; raw + camera + linguistic features; and a shifted-camera control. Directional features use a model trained on that reported AirPod side. An unseen side uses a labelled gravity-invariant fallback; it does not silently reuse the other side’s directional calibration.")
                                .foregroundStyle(.secondary)
                            pronunciationEditor
                            Button("Prepare a new guided session") {workflow.prepare(data:model.sentenceTraining.data)}
                                .disabled(!workflow.storageReady || model.sentenceTraining.isBusy || model.signalCheck.isBusy)
                            if let study=workflow.study,!study.complete,study.reference==nil {
                                Text("\(study.takes.count) reviewed historical phone annotations included from these training sentence groups; new A adds a fresh recording session.").font(.caption).foregroundStyle(.secondary)
                                Text("\(study.novelWords.count) word types in B are absent from this pilot’s training sentences. This checks sentence retrieval containing new words; it does not measure isolated unseen-word decoding.")
                                    .font(.callout).foregroundStyle(.secondary)
                                CameraPoseGuide(camera:workflow.camera,allowsMovement:true)
                                startButtons
                                DisclosureGroup("Sentence groups") {
                                    ForEach(study.training) {Text("A · "+$0.text)}
                                    ForEach(study.evaluation) {Text("B · "+$0.text)}
                                }
                                coverage(study)
                            }
                        }
                        if let study=workflow.study,study.complete {results(study)}
                        Text("This guided comparison keeps the existing frozen representation model. Neural CTC training and results are available under Advanced. Diagnostic recording saves numeric camera points and AirPods motion only; microphone audio and video are off. Linguistic supervision here is a weak sentence-level gesture/transition label, with no measured phoneme times. Dictionary variants remain proposals until edited.")
                            .font(.callout).foregroundStyle(.secondary)
                        if let error=workflow.error {Text(error).foregroundStyle(.orange)}
                        if model.sequenceLearning.phase == .working {
                            HStack {ProgressView().controlSize(.small);Text("Neural training is running in the background.");Button("Show progress") {model.showSequenceLearning=true}}
                                .font(.callout).foregroundStyle(.secondary)
                        }
                    }.frame(maxWidth:1000,alignment:.leading).padding(24).frame(maxWidth:.infinity)
            }
        }
    }
    private var pronunciationEditor:some View {
        DisclosureGroup("Your pronunciation variants · IPA") {
            VStack(alignment:.leading,spacing:10) {
                Text("Enter space-separated IPA units, one pronunciation per line. Multiple lines remain alternatives; your family’s accents do not choose a default. Saved changes apply to future pilots, never a frozen model.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    TextField("Word",text:$word).frame(width:180)
                    Button("Load proposals") {
                        variants=workflow.profile.alternatives(word,lexicon:.bundled).map {$0.ipa.joined(separator:" ")}.joined(separator:"\n")
                    }
                }
                TextEditor(text:$variants).font(.title3).frame(height:85).border(.secondary.opacity(0.2))
                Button("Save these alternatives") {workflow.editPronunciation(word:word,variants:variants)}
                Text("Available units: "+ArticulatoryInventory.all.map(\.ipa).joined(separator:"  ")).font(.caption).foregroundStyle(.secondary)
            }.padding(.top,8)
        }
    }
    private var startButtons:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {
                Button(workflow.camera.isRunning ? "Restart camera":"Enable local camera") {
                    Task {workflow.camera.stop();await workflow.camera.start()}
                }
                Button(workflow.block==0 ? "Start training A":"I’ve reseated · start B") {
                    workflow.startBlock(audio:model.audio,recorder:model.recorder)
                }.buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(!workflow.sourceStable || !workflow.cameraReady || model.motion.status != .streaming || model.recorder.isRecording)
            }
            TimelineView(.periodic(from:.now,by:0.1)) {_ in
                Text(workflow.currentSource+" · "+(workflow.sourceStable ? "source stable":"waiting for 1.5 seconds of stable motion"))
                    .font(.caption).foregroundStyle(workflow.sourceStable ? Color.secondary:Color.orange)
            }
        }
    }
    private var reseat:some View {
        VStack(alignment:.leading,spacing:18) {
            Text("A saved · model frozen").font(.title.weight(.semibold))
            Text("Remove and reseat both AirPods. B uses different sentences and a new recording session. It will not update the model.").font(.title2)
            Text("Frozen fingerprint: "+String(workflow.study?.fingerprint?.prefix(12) ?? "unavailable")).font(.caption).foregroundStyle(.secondary)
            CameraPoseGuide(camera:workflow.camera,allowsMovement:true)
            startButtons
            Button("Stop and keep A",action:workflow.stop)
        }
    }
    private var color:Color {
        workflow.phase == .relax ? .orange:workflow.condition == .normal ? .green:.blue
    }
    private var capture:some View {
        RecordingViewport {
            VStack(spacing:8) {
                HStack {
                    Text(workflow.block == 0 ? "A · training" : "B · frozen transfer").font(.headline)
                    Spacer()
                    Text("\(workflow.index+1) of \(workflow.plan.count)").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                }
                ProgressView(value:Double(workflow.index),total:Double(max(1,workflow.plan.count))).tint(color)
            }
        } prompt: {
            VStack(spacing:8) {
                CameraPoseGuide(camera:workflow.camera,compact:true,allowsMovement:true).frame(height:70).padding(.horizontal,12)
                RecordingPromptPanel(color:color) {
                    VStack(spacing:18) {
                        RecordingCueStatus(timer:workflow.phase == .ready ? "When you’re ready" : workflow.phase == .ended ? "Saved on this Mac" : String(format:"%.1f s",workflow.secondsRemaining),
                            title:workflow.phase == .relax ? "Relax" : workflow.phase == .ready ? "Ready" : workflow.phase == .ended ? "Saved" : "Recording", color:color)
                        Label(workflow.condition == .normal ? "Speak aloud":"Mouth silently",systemImage:workflow.condition == .normal ? "speaker.wave.2":"mouth")
                            .font(.title3.weight(.medium)).foregroundStyle(color)
                        WordFocusPrompt(text:workflow.currentText,activeWordIndex:workflow.activeWordIndex,color:color)
                        Text(workflow.phase == .relax ? "Relax your mouth and move your head comfortably. Keep your face visible.":"Follow the highlighted words. Choose Done when finished; redo any mistake.")
                            .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        if let message=workflow.sourceMessage {Text(message).font(.caption).foregroundStyle(.orange)}
                        if let error=workflow.error {Text(error).font(.caption).foregroundStyle(.orange)}
                    }
                }
            }
        } actions: {
            VStack(spacing:10) {
                HStack(spacing:12) {
                    Button("Stop and save",action:workflow.stop).buttonStyle(.borderless).foregroundStyle(.secondary)
                    Spacer()
                    Button(action:workflow.redo) {Label("Redo",systemImage:"arrow.counterclockwise")}.disabled(!workflow.canRedo)
                    Button(action:workflow.next) {
                        Label(workflow.phase == .ready ? "Start take" : workflow.phase == .recording ? "Done" : "Continue",
                              systemImage:workflow.phase == .recording ? "checkmark":"chevron.right").frame(minWidth:130)
                    }.buttonStyle(.borderedProminent).disabled(workflow.phase == .relax)
                }.controlSize(.large)
                Text("← / ↑ Redo     → / ↓ Start, finish or continue · "+workflow.currentSource).font(.caption).foregroundStyle(.tertiary)
            }
        } motion: {LiveMotionPanel(plotHeight:44)}
    }
    private func results(_ study:RepresentationStudy)->some View {
        VStack(alignment:.leading,spacing:12) {
            Divider()
            Text("Fresh-session development results").font(.title2.weight(.semibold))
            Text("All B sentences and its entire session were excluded from fitting. \(study.novelWords.count) new word types. Candidate bank includes A and B; random choice is 1 in \(study.training.count+study.evaluation.count).")
                .font(.callout).foregroundStyle(.secondary)
            ForEach([SpeechCondition.normal,.silent]) {condition in
                Text(condition.rawValue).font(.headline)
                ForEach(RepresentationArm.allCases,id:\.self) {arm in
                    let rows=(study.predictions ?? []).filter {$0.condition==condition && $0.arm==arm}
                    let correct=rows.filter {$0.predicted==$0.expected}.count
                    let scored=rows.filter {!$0.ranked.isEmpty}.count
                    HStack {
                        Text(arm.label);Spacer()
                        Text("\(correct)/\(study.evaluation.count*study.repetitions) planned · \(scored) scored").monospacedDigit()
                    }
                }
            }
            let fallback=(study.predictions ?? []).filter {$0.arm == .raw && $0.invariantFallback}.count
            Text("\(fallback) evaluations used the invariant fallback for an unseen AirPod side. Missing/ambiguous outputs count as incorrect. Score margins are uncalibrated, not probabilities.")
                .font(.caption).foregroundStyle(.secondary)
            Text("\(study.sourceEvents?.count ?? 0) reported AirPod source transitions saved with timestamps and capture gaps.").font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("Ranked alternatives and source identity") {
                ForEach(Array((study.predictions ?? []).enumerated()),id:\.offset) {_,p in
                    Text(p.arm.label+" · source \(p.source ?? 0)"+(p.invariantFallback ? " · invariant fallback":"")+" · "+(p.reason ?? p.ranked.first?.text ?? "unmatched"))
                        .font(.caption)
                }
            }
            DisclosureGroup("Targeted additional examples") {
                ForEach(Array(study.requests(lexicon:.bundled).enumerated()),id:\.offset) {_,request in
                    Text(request.text).font(.headline)
                    Text(request.reason).font(.caption).foregroundStyle(.secondary)
                }
                Text("Suggestions use training coverage and development errors only. They never modify this frozen study or consume the reserved final test.").font(.caption)
            }
            coverage(study)
            if let reference=study.reference {
                Text(reference.heads.contains {$0.frameLinguistic != nil} ? "Linguistic supervision includes reviewed time-local phone estimates." : "No reviewed time-local phone estimates were available: linguistic supervision used sentence-level gesture/transition summaries only.")
                    .font(.caption).foregroundStyle(.secondary)
                DisclosureGroup("Learned sensor overlap · exploratory") {
                    if reference.sensorOverlaps.isEmpty {
                        Text("No phoneme-level overlap estimates yet. Need reviewed phone alignment in at least two sentence contexts with three examples per sound. Dictionary gesture priors are not learned sensor classes.").font(.caption)
                    }
                    ForEach(Array(reference.sensorOverlaps.prefix(20).enumerated()),id:\.offset) {_,overlap in
                        Text("\(overlap.modality): /\(overlap.first)/ ↔ /\(overlap.second)/ · "+overlap.interpretation).font(.caption)
                    }
                }
            }
            Text("This pilot tests retrieval among a small known candidate bank, using weak articulatory priors. It does not establish phoneme recognition, continuous transcription, pure jaw sensing, or a causal reseating effect. A-then-B also confounds time and practice. Linear auxiliary predictions add learned bias, not new sensor information at inference.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
    private func coverage(_ study:RepresentationStudy)->some View {
        let graphs=study.training.map {LinguisticGraph(text:$0.text,profile:study.profile,lexicon:.bundled)}
        let phones=Set(graphs.flatMap {$0.phonemeCoverage.keys})
        let pairs=Set(graphs.flatMap {$0.diphoneCoverage.keys.map(\.key)})
        let words=Dictionary(grouping:graphs.flatMap(\.words),by:\.word)
        return DisclosureGroup("Linguistic annotation plan · \(phones.count) IPA units · \(pairs.count) diphones") {
            Text("Counts describe expected annotations, not sounds the sensors can distinguish. Alternatives have equal weight until better speaker-specific evidence is available.")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(words.keys.sorted(),id:\.self) {word in
                let alternatives=words[word]!.first!.alternatives
                Text(word+" → "+alternatives.map {"/"+$0.ipa.joined(separator:" ")+"/"+($0.speakerConfirmed ? "":" (proposal)")}.joined(separator:" or ")).font(.caption)
            }
            Text("Diphones: "+pairs.sorted().joined(separator:", ")).font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct PhoneAlignmentSettingsView:View {
    @Bindable var workflow:SentenceTrainingWorkflow
    var body:some View {
        @Bindable var aligner=workflow.phoneAligner
        VStack(alignment: .leading, spacing: 10) {
          Toggle("Collect numeric camera points", isOn: $workflow.captureCameraPoints).disabled(workflow.isBusy)
          DisclosureGroup("Local phoneme alignment (optional)") {
            VStack(alignment:.leading,spacing:10) {
                Text("Whisper supplies word estimates. Phoneme boundaries require a separate local acoustic model and an explicit model-phone → IPA mapping. No model is downloaded. Leave these empty to keep phone timings unavailable.")
                    .font(.caption).foregroundStyle(.secondary)
                TextField("MFA 3.x executable path",text:$aligner.executablePath)
                TextField("Local acoustic model file path",text:$aligner.modelPath)
                TextField("Model phone → IPA JSON map path",text:$aligner.phoneMapPath)
                if let role=workflow.collectionRole, !workflow.variedSentences {
                    Text("Teacher collection · \(role.rawValue) sentences only in this recording session.").font(.caption.weight(.semibold))
                    Button("Return to regular paired training") {workflow.collectionRole=nil}
                }
                Text("Silent phone alignment uses its own camera warp, allowing different pacing. Camera frames stay local; video is not saved.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(workflow.isBusy)
          }
        }
    }
}
