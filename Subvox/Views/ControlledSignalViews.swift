import AppKit
import SwiftUI

struct ControlledSignalSetupView: View {
    @Environment(AppModel.self) private var model
    private var workflow: SignalCheckWorkflow { model.signalCheck }
    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { _ in
            let between = workflow.phase == .controlBreak
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(workflow.usesCameraTeacher ? (between ? "2 of 2 · reseat and test B" : "1 of 2 · train and validate A") : between ? "Block B · reseat both AirPods" : "Block A · keep AirPods seated")
                        .font(.system(size: 34, weight: .semibold))
                    Text(workflow.usesCameraTeacher && !between ? "Reseat both AirPods once before training A, so this recording adds a fresh fit to the training history. Enable the camera and keep your face and mouth visible. You can turn, nod and shift comfortably; there is no pose target." : workflow.allowsHeadMovement ? (between ? "Remove both AirPods and put them back in naturally. Keep your face visible to the same camera. Move comfortably; there is no pose target." : "Keep the AirPods seated and enable the camera. You can turn, nod and shift comfortably; keep your face and mouth visible.") : between ? "Remove both AirPods, put them back in naturally, and return to the original camera target. Keep the chair, camera and screen in the same positions."
                         : "Leave the AirPods in place. Sit comfortably facing the camera, keep your chair and screen fixed, and save a neutral head pose.")
                        .font(.title3).fixedSize(horizontal: false, vertical: true)
                    CameraPoseGuide(camera: workflow.camera, target: workflow.comparison?.target, allowsMovement: workflow.allowsHeadMovement)
                    HStack {
                        Button(workflow.camera.isStarting ? "Starting camera…" : workflow.camera.isRunning ? "Restart local camera guide" : "Enable local camera guide") {
                            Task {
                                if workflow.camera.isRunning { workflow.camera.stop() }
                                await workflow.camera.start(cameraID: workflow.comparison?.target?.cameraID)
                            }
                        }.controlSize(.large).disabled(workflow.camera.isStarting)
                        if !between && !workflow.allowsHeadMovement {
                            Button(workflow.comparison?.target == nil ? "Save neutral pose" : "Save new pose before block A") { workflow.savePoseTarget() }
                                .controlSize(.large).disabled(!workflow.canSavePose)
                        }
                        Spacer()
                        Button(between ? "I’ve reseated the AirPods · start block B" : workflow.usesCameraTeacher ? "I’ve reseated for training · start A" : "AirPods stayed in place · start block A") {
                            model.startSignalCheck(againstReference: true, controlled: true)
                        }.buttonStyle(.borderedProminent).controlSize(.large)
                            .disabled(!workflow.canStartControlledBlock || model.motion.status != .streaming)
                    }
                    if let tilt = workflow.imuTiltDifference, !workflow.allowsHeadMovement {
                        Text(String(format: "AirPods gravity direction differs by %.1f° from setup. This is recorded as a fit/pose diagnostic; it is not used to move the camera target or recalibrate the model.", tilt))
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    Text("Recording: lip points, numeric pose and AirPods motion only. Microphone audio and camera video are off.")
                        .font(.callout).foregroundStyle(.secondary)
                    if workflow.usesCameraTeacher {
                        Text(workflow.teacherHistoryCount > 0 ? "Training includes \(workflow.teacherHistoryCount) previously completed recording sessions plus this new A block. Entire sessions are held out to choose features. The new B block is the fresh test." : "No completed session history is available. A uses repetition-based validation; that cannot establish reseating robustness. B is the fresh test.")
                            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                    }
                    Divider()
                    Text(workflow.usesCameraTeacher ? "36 takes total · 18 training + 18 final test" : "Identical visual pacing in both blocks").font(.title2.weight(.medium))
                    Text(workflow.usesCameraTeacher ? "Three phrases, three repetitions each, spoken and mouthed. During each 3-second baseline, relax your mouth and gently turn, nod or shift comfortably. Follow the highlighted words after START. Pacing varies between comfortable speeds and differs between A and B so the model cannot rely on one fixed timeline. No head position is required." : workflow.allowsHeadMovement ? "Three phrases, three repetitions each, spoken and mouthed: 18 takes per block. During each 3-second baseline, relax your mouth and gently move your head naturally. Then follow the highlighted words every 0.65 seconds. Each recording ends after 3.4 seconds. Redo if you miss a word; head movement is allowed." : "Three phrases, three repetitions each, spoken and mouthed: 18 takes per block. Wait through the 3-second still baseline, then follow the highlighted word every 0.65 seconds. Recording ends automatically after 3.4 seconds. Keep your head still; do not nod to the rhythm. Redo a take if you miss a word or the pace.")
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Text(workflow.usesCameraTeacher ? "The camera teaches the AirPods to predict changes in mouth shape. Eye landmarks provide a continuous head-motion reference; relaxed-mouth periods teach what to ignore. A and the saved training history choose and freeze the model before B. Whole sessions are held out when available. B tests AirPods input alone. The primary test compares learned versus raw features with the same linear decoder. Other arms check context, template matching, camera mouth motion and head-motion confounding. Your reserved sentence final test remains untouched." : workflow.allowsHeadMovement ? "Block A learns how camera-measured head movement appears in AirPods rotation, using relaxed-mouth baselines only. The correction and new phrase references are locked before block B. Raw and corrected rotation are compared side by side; the older reference is also scored separately. Only camera tracking and capture timing affect the quality check, not your position. Predictions stay hidden until both blocks finish." : "One pose target, one frozen reference and one shuffled prompt order are used throughout. Face angles must be within 5°, face position within 5% of the frame and apparent face size within 12% before each take. Predictions remain hidden until both blocks finish.")
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if between, workflow.breakRemaining > 0 {
                        Text(String(format: "Rest and reseat · %.0f seconds remaining before block B can start", ceil(workflow.breakRemaining)))
                            .font(.headline).foregroundStyle(.orange)
                    }
                    if between, let note = workflow.comparison?.cameraAdjustmentNote { Text(note).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true) }
                    if between, let correction = workflow.comparison?.cameraReference?.correction {
                        Text(String(format: "Head-motion correction locked · %.0f%% less error on held-out baseline intervals. It will not refit after reseating.", correction.heldOutReduction*100)).font(.callout).foregroundStyle(.secondary)
                    }
                    if between && workflow.usesCameraTeacher {
                        Text("Halfway: all 18 A takes are saved. Reseat the AirPods, then record 18 B takes to finish the test.").font(.headline)
                        if let comparison = workflow.comparison { MouthTeacherResultsView(comparison: comparison) }
                    }
                    if let error = workflow.error { Text(error).foregroundStyle(.orange) }
                    Button(between ? "Stop and keep block A" : "Cancel controlled test", action: workflow.stop)
                    Text("The fixed A-then-B order cannot fully separate reseating from time or fatigue. The report keeps this limitation and all incomplete or off-target takes visible.")
                        .font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
            }
        }
    }
}

struct ControlledSignalResultsView: View {
    let comparison: ControlledSignalComparison
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Controlled comparison").font(.largeTitle.weight(.semibold))
            Text(comparison.blocks.count == 2 && comparison.endedAt != nil ? "Both blocks saved" : "Incomplete comparison · completed takes were saved")
                .foregroundStyle(.secondary)
            Text("New signal checks save points and motion only; no audio or video files are recorded.").foregroundStyle(.secondary)
            if comparison.allowsMovement {
                Text("Natural head movement allowed · quality uses tracking coverage and recording timing. Head position is not a pass/fail condition.").foregroundStyle(.secondary)
                if comparison.motionPolicy == .cameraTeacher { MouthTeacherResultsView(comparison: comparison) }
                else { cameraAdjustedResults }
            }
            Text("Fixed reference " + String(comparison.reference.sha256.prefix(12)) + " · target " + String(comparison.target?.id.uuidString.prefix(8) ?? "not saved"))
                .font(.caption).foregroundStyle(.secondary)
            ForEach(comparison.blocks.indices, id: \.self) { i in
                let run = comparison.blocks[i]
                let quality = ControlledSignalComparison.qualities(run)
                VStack(alignment: .leading, spacing: 12) {
                    Text((run.control?.block.label ?? "Block") + (comparison.allowsMovement ? " · older fixed reference" : "")).font(.title2.weight(.semibold))
                    if let report = run.report {
                        ForEach(report.conditions.indices, id: \.self) { j in
                            let score = report.conditions[j]
                            let takes = run.takes.filter { !$0.discarded && $0.prompt.condition == score.condition }
                            let usable = Set(report.metrics.filter(\.usable).map(\.takeID))
                            let qualified = Set(quality.filter { $0.passes && usable.contains($0.takeID) }.map(\.takeID))
                            let eligible = takes.filter { qualified.contains($0.id) }
                            let correct = report.predictions?.filter { prediction in
                                eligible.contains(where: { $0.id == prediction.takeID }) && prediction.predictedSentence == prediction.expectedSentence
                            }.count ?? 0
                            VStack(alignment: .leading, spacing: 5) {
                                Text(score.condition == .silent ? "Mouthed phrases" : "Spoken phrases").font(.headline)
                                Text("All planned takes: \(score.correct) / 9 matched · \(score.evaluated) scored · \(9 - (score.collectedTakes ?? 0)) unrecorded")
                                Text("\(comparison.allowsMovement ? "Tracking" : "Pose") and capture-timing compliant subset: \(correct) / \(eligible.count) matched")
                                Text("Time-shift control: \(score.shiftedCorrect) / \(score.shiftedEvaluated)").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    } else { Text("No fixed-model score was completed for this block.").foregroundStyle(.orange) }
                    DisclosureGroup("Pose, pacing and recordings") {
                        ForEach(run.takes.filter { !$0.discarded }) { take in
                            let q = quality.first { $0.takeID == take.id }
                            HStack {
                                Text(SignalCheckRun.sentences[take.prompt.sentence] + " · " + (take.prompt.condition == .silent ? "Mouthed" : "Spoken") + " · repeat \(take.prompt.repetition + 1)")
                                Spacer()
                                if let q {
                                    Text(comparison.allowsMovement ? String(format: "Head tracking %.0f%% · timer error %.0f ms", q.coverage*100, q.timingError*1000) : String(format: "Camera %.0f%% · at target %.0f%% · timer error %.0f ms", q.coverage*100, q.withinTarget*100, q.timingError*1000))
                                        .foregroundStyle(q.passes ? Color.secondary : Color.orange)
                                }
                            }.font(.caption).padding(.vertical, 3)
                            let lips = CameraMouthStatistics.inspect(run.control?.readings ?? [], start: take.recording.cueStart, end: take.recording.cueEnd)
                            Text(String(format: "Lip tracking %.0f%% coverage · %.1f samples/s · %d receipt-time fallback frames", lips.coverage*100, lips.rate, lips.receiptTimeFallbacks))
                                .font(.caption).foregroundStyle(.secondary)
                            SignalPlot(title: "Raw rotation", samples: take.recording.cueSamples, channels: SignalPlot.rotation, minScale: 0.05, height: 65)
                            if let prediction = run.report?.predictions?.first(where: { $0.takeID == take.id }) {
                                Text(prediction.predictedSentence.map { "Matched: " + SignalCheckRun.sentences[$0] } ?? prediction.reason ?? "Unmatched")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if let directory = run.session?.directory {
                            Button("Open raw block recording") { NSWorkspace.shared.open(directory) }
                        }
                    }
                }.padding(.vertical, 12)
                Divider()
            }
            if let baseline = comparison.blocks.first?.report?.conditions.first(where: { $0.condition == .silent }), baseline.correct <= 3 {
                Text("The no-reseat mouthed baseline did not exceed the 3/9 chance expectation. A low score after reseating cannot establish an additional reseating effect when the baseline is already this weak.")
                    .foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
            Text(comparison.motionPolicy == .cameraTeacher ? "Teacher-test AirPods arms use motion coverage and the recorded pacing schedule. Missing camera data affects the camera/head benchmarks, not student predictions. All arms retain the nine-planned-take denominator. Older fixed-reference results below are historical six-channel comparisons, separate from the new teacher test." : comparison.allowsMovement ? "Scores retain all nine planned takes per condition. The tracking subset needs at least 90% valid head and lip coverage and recording duration within 100 ms of the schedule. Head movement is allowed. Camera-assisted scores use rotation only and a new reference from block A; older fixed-reference scores use acceleration and rotation and are reported separately." : "Primary scores include every planned take; unmatched valid takes count in the denominator. The compliant subset requires at least 90% camera coverage and time at target, with capture duration within 100 ms of the schedule. Subsets may have unequal phrase counts.")
                .font(.callout).foregroundStyle(.secondary)
            Text("Pacing is visually prescribed, not a measurement of silent word timing. Camera pose is estimated. The fixed A-then-B order also permits time, fatigue and practice effects. This is a small three-phrase diagnostic; the model and the reserved final-test sentences stay unchanged.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var cameraAdjustedResults: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Camera-assisted A → B comparison").font(.title2.weight(.semibold))
            if let reference = comparison.cameraReference {
                if let correction = reference.correction {
                    Text(String(format: "Correction fitted on relaxed-mouth intervals in A; baseline validation error fell %.0f%%. Coefficients, templates and scales are fixed before B.", correction.heldOutReduction*100))
                }
                if let scores = comparison.cameraScores {
                    ForEach(scores.indices, id: \.self) { i in
                        let score = scores[i]
                        Text(score.condition == .silent ? "Mouthed phrases" : "Spoken phrases").font(.headline)
                        Text("Raw rotation: \(score.rawCorrect) / 9 planned · \(score.rawEvaluated) scored · \(9-score.collected) unrecorded")
                        if let correct = score.correctedCorrect, let evaluated = score.correctedEvaluated {
                            Text("Head-corrected rotation: \(correct) / 9 planned · \(evaluated) scored")
                        } else { Text("Head-corrected score unavailable").foregroundStyle(.orange) }
                    }
                } else { Text("A reference is saved. No completed block-B evaluation is available.").foregroundStyle(.secondary) }
            } else { Text("A complete block A is needed to freeze the camera-assisted reference.").foregroundStyle(.secondary) }
            if let note = comparison.cameraAdjustmentNote { Text(note).foregroundStyle(.orange) }
            Text("Correction estimates the component of AirPods rotation associated with head movement. It is experimental and may remove some useful signal or fail after reseating; the raw-versus-corrected comparison measures whether it helps.").font(.caption).foregroundStyle(.secondary)
        }.padding(.vertical, 10)
    }
}
