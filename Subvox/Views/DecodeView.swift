import SwiftUI

/// The "motion → text" screen: each segment is matched against every vocabulary word, built
/// from the trained sounds, and appended to a running transcript, with the mic transcript
/// alongside for comparison.
struct DecodeView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                LiveMotionPanel()
                Text(model.decoderSourceLabel).font(.caption).foregroundStyle(.secondary)
                coverage
                controls
                transcriptPanel
                HStack(alignment: .top, spacing: 16) {
                    lastPredictionPanel
                    micPanel
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private var coverage: some View {
        if let decoder = model.decoder {
            let missing = decoder.missingPhones.compactMap { Phoneme.bySymbol[$0].map { "/\($0.ipa)/" } }
            Label(missing.isEmpty
                  ? "Can spell all \(decoder.wordCount) vocabulary words from your trained sounds."
                  : "Can spell \(decoder.wordCount) words. Words with \(missing.joined(separator: " ")) need those sounds trained first.",
                  systemImage: "character.book.closed")
                .font(.callout)
                .foregroundStyle(missing.isEmpty ? Color.secondary : Color.orange)
            Text("\(model.settings.method.title) · \(model.settings.vocabularyScope.title)")
                .font(.caption).foregroundStyle(.secondary)
            if model.settings.method != .legacyMedoids {
                Text("Experimental jaw units pool several sounds; buildable words do not establish recognition accuracy.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } else {
            Label("Train sounds in the Train tab first.", systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
    }

    // MARK: Controls

    private var controls: some View {
        @Bindable var model = model
        @Bindable var capture = model.capture
        return GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Segmentation", selection: $model.decodeMode) {
                    ForEach(CaptureController.Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)

                if model.decodeMode == .hold {
                    HoldToCaptureButton(
                        title: "Mouth a word",
                        isCapturing: capture.isCapturing,
                        isEnabled: model.decoder != nil && model.motion.status == .streaming,
                        onBegin: capture.begin,
                        onEnd: capture.end)
                } else {
                    autoControls
                }

                if model.settings.method.isSequence {
                    VStack(alignment: .leading, spacing: 8) {
                        LabeledContent("Max sequence score") {
                            Slider(value: $model.hmmMaxScore, in: -10...50).frame(width: 160)
                            Text(String(format: "%.2f", model.hmmMaxScore)).monospacedDigit()
                        }
                        LabeledContent("Min sequence gap") {
                            Slider(value: $model.hmmMinGap, in: 0...5).frame(width: 160)
                            Text(String(format: "%.2f", model.hmmMinGap)).monospacedDigit()
                        }
                        LabeledContent("Common-word bias") {
                            Slider(value: $model.commonBias, in: 0...0.2).frame(width: 160)
                            Text(String(format: "%.2f", model.commonBias)).monospacedDigit()
                        }
                        Text("Sequence score is a normalized negative log-likelihood. These rejection settings need validation; neither gap nor score is a probability.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                } else {
                HStack(spacing: 24) {
                    LabeledContent("Max distance") {
                        Slider(value: $model.maxDistance, in: 0.5...8).frame(width: 140)
                        Text(String(format: "%.2f", model.maxDistance)).monospacedDigit().frame(width: 36)
                    }
                    LabeledContent("Min confidence") {
                        Slider(value: $model.minConfidence, in: 0...0.5).frame(width: 140)
                        Text(String(format: "%.2f", model.minConfidence)).monospacedDigit().frame(width: 36)
                    }
                    LabeledContent("Common-word bias") {
                        Slider(value: $model.commonBias, in: 0...0.2).frame(width: 140)
                        Text(String(format: "%.2f", model.commonBias)).monospacedDigit().frame(width: 36)
                    }
                    .help("Many words move the jaw identically; this tips ties toward the more common word.")
                }
                .font(.callout)
                .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                }
            }
            .padding(6)
        }
    }

    private var autoControls: some View {
        @Bindable var capture = model.capture
        let scaleMax = max((capture.threshold ?? 0.05) * 2, 0.02)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(capture.isCalibrating ? "Calibrating… stay still" : "Calibrate noise floor") {
                    capture.calibrate()
                }
                .disabled(capture.isCalibrating || model.motion.status != .streaming)
                if let floor = capture.noiseFloor {
                    Text(String(format: "floor %.4f", floor)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                Circle()
                    .fill(capture.isCapturing ? Color.red : Color.secondary.opacity(0.3))
                    .frame(width: 10, height: 10)
                Text(capture.isCapturing ? "Capturing" : "Listening")
                    .font(.caption)
            }
            LevelMeter(value: capture.energy / scaleMax,
                       marker: capture.threshold.map { $0 / scaleMax },
                       tint: capture.isCapturing ? .red : .blue)
            LabeledContent("Trigger × floor") {
                Slider(value: $capture.triggerRatio, in: 1.5...10).frame(width: 160)
                Text(String(format: "%.1f×", capture.triggerRatio)).monospacedDigit()
            }
            .font(.callout)
            if capture.noiseFloor == nil {
                Text("Calibrate first: keep still for 2 s so the detector learns your resting motion.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: Transcript

    private var transcriptPanel: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Decoded from motion").font(.headline)
                    Spacer()
                    Button("Clear", action: model.clearDecoded)
                        .disabled(model.decoded.isEmpty)
                }
                if model.decoded.isEmpty {
                    Text(model.isDecoding ? "Decoding…" : "Mouth any word…")
                        .font(.system(size: 28, weight: .regular, design: .rounded))
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, minHeight: 80, alignment: .topLeading)
                } else {
                    decodedText
                        .font(.system(size: 28, weight: .medium, design: .rounded))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, minHeight: 80, alignment: .topLeading)
                }
                Text("Faded words were rejected by the distance/confidence thresholds.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(6)
        }
    }

    private var decodedText: Text {
        model.decoded.reduce(Text("")) { text, word in
            let piece = Text(word.text + " ")
            return text + (word.prediction.accepted ? piece : piece.foregroundColor(.secondary.opacity(0.4)))
        }
    }

    // MARK: Diagnostics

    private var lastPredictionPanel: some View {
        GroupBox("Last segment") {
            VStack(alignment: .leading, spacing: 8) {
                if let prediction = model.decoded.last?.prediction {
                    let worst = prediction.ranked.last?.distance ?? 1
                    ForEach(prediction.ranked) { candidate in
                        HStack {
                            Text(candidate.label).frame(width: 70, alignment: .leading)
                            LevelMeter(value: 1 - candidate.distance / max(worst * 1.2, 1e-6),
                                       tint: candidate == prediction.best ? .green : .gray)
                            Text(String(format: "%.2f", candidate.distance))
                                .font(.caption.monospacedDigit())
                                .frame(width: 40, alignment: .trailing)
                        }
                    }
                    Text(String(format: "Confidence %.2f · %@", prediction.confidence,
                                prediction.accepted ? "accepted" : "rejected"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    SignalPlot(title: "Rotation rate", samples: model.lastSegment,
                               channels: SignalPlot.rotation, minScale: 0.05, height: 60)
                } else {
                    Text("The closest vocabulary words, by DTW distance to their sound-built templates, appear here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private var micPanel: some View {
        GroupBox("Microphone transcript (reference)") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button(model.audio.isRunning ? "Stop Mic" : "Start Mic") {
                        Task { await model.toggleMic() }
                    }
                    LevelMeter(value: Double(model.audio.level))
                    Button("Clear", action: model.transcriber.clear)
                }
                Text(model.transcriber.text.isEmpty ? "Whisper or speak to compare against the motion decoder." : model.transcriber.text)
                    .foregroundStyle(model.transcriber.text.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, minHeight: 60, alignment: .topLeading)
                if model.transcriber.isRunning {
                    Text(model.transcriber.isOnDevice ? "On-device recognition" : "Server recognition")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if let error = model.audio.error ?? model.transcriber.error {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}
