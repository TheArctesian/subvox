import SwiftUI

struct DecodeView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            if model.decoder == nil {
                VStack(spacing: 18) {
                    Image(systemName: "text.bubble").font(.system(size: 40)).foregroundStyle(.secondary)
                    Text(model.isBuildingDecoder ? "Building your model…" : "Start with a training session").font(.title2.weight(.semibold))
                    Text("Record paired passages and review their motion traces. Your model becomes available here automatically.")
                        .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 440)
                    if model.isBuildingDecoder { ProgressView() }
                    Button("Go to Train") { model.screen = .train }.buttonStyle(.borderedProminent)
                }.frame(maxWidth: .infinity, maxHeight: .infinity).padding(32)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        Text("Try a mouthed word").font(.largeTitle.weight(.semibold))
                        Text("This model recognizes individual words. Hold Space while mouthing, then release.")
                            .foregroundStyle(.secondary)
                        controls
                        transcript
                        if !model.decoded.isEmpty { predictionDetails }
                        advanced
                    }.frame(maxWidth: 1000, alignment: .leading).padding(24).frame(maxWidth: .infinity)
                }
            }
            Divider()
            LiveMotionPanel().padding(16)
        }
    }

    private var controls: some View {
        @Bindable var model = model
        @Bindable var capture = model.capture
        return VStack(alignment: .leading, spacing: 12) {
            Picker("Capture", selection: $model.decodeMode) {
                ForEach(CaptureController.Mode.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).frame(maxWidth: 340)
            if model.decodeMode == .hold {
                HoldToCaptureButton(title: "Mouth a word", isCapturing: capture.isCapturing,
                    isEnabled: model.motion.status == .streaming && !model.sentenceTraining.isBusy,
                    onBegin: capture.begin, onEnd: capture.end)
            } else {
                Button(capture.isCalibrating ? "Calibrating… stay still" : "Calibrate resting motion") { capture.calibrate() }
                    .disabled(capture.isCalibrating || model.motion.status != .streaming)
                LevelMeter(value: capture.energy / max((capture.threshold ?? 0.05) * 2, 0.02), tint: capture.isCapturing ? .red : .blue)
                Text(capture.noiseFloor == nil ? "Keep still for two seconds while calibrating." : capture.isCapturing ? "Capturing motion…" : "Waiting for motion…")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(model.sentenceTraining.isBusy)
    }

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Model output").font(.headline)
                Spacer()
                Button("Clear", action: model.clearDecoded).disabled(model.decoded.isEmpty)
            }
            if model.decoded.isEmpty {
                Text(model.isDecoding ? "Decoding…" : "Your words will appear here.").foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
            } else {
                model.decoded.reduce(Text("")) { text, word in
                    text + Text(word.text + " ").foregroundColor(word.prediction.accepted ? .primary : .secondary.opacity(0.4))
                }.font(.system(size: 30, weight: .medium)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, minHeight: 100, alignment: .topLeading)
            }
            Text("Faded words were rejected by the model’s thresholds.").font(.caption).foregroundStyle(.secondary)
        }.padding(18).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    private var predictionDetails: some View {
        DisclosureGroup("Last word details") {
            if let prediction = model.decoded.last?.prediction {
                ForEach(prediction.ranked) { candidate in
                    HStack {
                        Text(candidate.label)
                        Spacer()
                        Text(String(format: "%.3f", candidate.distance)).monospacedDigit()
                    }.font(.callout)
                }
                Text(prediction.accepted ? "Accepted" : "Rejected").font(.caption).foregroundStyle(.secondary)
                SignalPlot(title: "Captured motion", samples: model.lastSegment, channels: SignalPlot.rotation, minScale: 0.05, height: 80)
            }
        }
    }

    private var advanced: some View {
        @Bindable var model = model
        @Bindable var capture = model.capture
        return DisclosureGroup("Recognition settings") {
            VStack(alignment: .leading, spacing: 14) {
                Text(model.decodeSettings.method.title).font(.headline)
                if let decoder = model.decoder {
                    Text("\(decoder.wordCount) candidate words can be built from the collected sounds. This count is not recognition accuracy.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if model.decodeSettings.method.isSequence {
                    LabeledContent("Maximum sequence score") { Slider(value: $model.hmmMaxScore, in: -10...50); Text(model.hmmMaxScore, format: .number.precision(.fractionLength(2))) }
                    LabeledContent("Minimum score gap") { Slider(value: $model.hmmMinGap, in: 0...5); Text(model.hmmMinGap, format: .number.precision(.fractionLength(2))) }
                } else {
                    LabeledContent("Maximum distance") { Slider(value: $model.maxDistance, in: 0.5...8); Text(model.maxDistance, format: .number.precision(.fractionLength(2))) }
                    LabeledContent("Minimum confidence") { Slider(value: $model.minConfidence, in: 0...0.5); Text(model.minConfidence, format: .number.precision(.fractionLength(2))) }
                }
                LabeledContent("Common word preference") { Slider(value: $model.commonBias, in: 0...0.2); Text(model.commonBias, format: .number.precision(.fractionLength(2))) }
                if model.decodeMode == .auto {
                    LabeledContent("Motion trigger") { Slider(value: $capture.triggerRatio, in: 1.5...10); Text(capture.triggerRatio, format: .number.precision(.fractionLength(1))) }
                }
                Text("Tune thresholds with the validation word check in Train. Scores and confidence are not calibrated probabilities.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 12)
        }.disabled(model.decodeSettingsLocked)
    }
}
