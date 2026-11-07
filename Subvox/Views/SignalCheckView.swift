import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// A recording instrument: one timed cue, with the traces kept in view throughout.
struct SignalCheckView: View {
    @Environment(AppModel.self) private var model
    private let spokenColor = Color(red: 0.10, green: 0.46, blue: 0.30)
    private let silentColor = Color(red: 0.19, green: 0.37, blue: 0.78)
    private let thinkColor = Color(red: 0.48, green: 0.24, blue: 0.71)
    private let relaxColor = Color(red: 0.62, green: 0.39, blue: 0.08)

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("Find the signal").font(.largeTitle.weight(.semibold))
                Spacer()
                Text("Paired recording").foregroundStyle(.secondary)
                if model.signalCheck.phase == .setup { beginButton }
            }.padding(24)
            Divider()
            if model.signalCheck.isAcquiring {
                acquisition.padding(24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    switch model.signalCheck.phase {
                    case .setup: setup
                    case .preparing: ProgressView("Starting local recording…")
                    case .baseline, .cue, .recovery: EmptyView()
                    case .processing:
                        VStack(alignment: .leading, spacing: 12) {
                            ProgressView("Checking voiced references with local Whisper…")
                            Text("Motion and audio are saved. This step stays on your Mac.").foregroundStyle(.secondary)
                        }.padding(.vertical, 32)
                    case .finished: finished
                    case .failed(let message):
                        Text(message).foregroundStyle(.red)
                        Button("Back to setup", action: model.signalCheck.reset)
                    }
                }
                .frame(maxWidth: 860, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(24)
            }
            }
            Divider()
            LiveMotionPanel().padding(16)
        }
    }

    private var setup: some View {
        @Bindable var check = model.signalCheck
        @Bindable var audio = model.audio
        let words = model.dataset.experiment.map(SignalCheckPlan.words) ?? []
        let count = words.count * 3 * check.rounds
        return VStack(alignment: .leading, spacing: 22) {
            Text("Same words. Three ways.").font(.title2.weight(.medium))
            Text("We’ll shuffle spoken words, silent mouthing and mouth-still controls in one visit. Comparing repeated whole words tells us whether the motion carries useful information without guessing any sound boundaries.")
                .fixedSize(horizontal: false, vertical: true)
                .foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 26) {
                instruction("Speak", "Say the word once, naturally.", color: spokenColor, symbol: "speaker.wave.2")
                instruction("Mouth", "Move your mouth as if speaking, with no voice.", color: silentColor, symbol: "mouth")
                instruction("Think", "Read the word in your head. Keep your mouth and head still.", color: thinkColor, symbol: "brain")
            }
            Label("Relax between cues: let your jaw rest and breathe normally.", systemImage: "leaf")
                .font(.callout.weight(.medium)).foregroundStyle(relaxColor)
            Text("There are no keypresses between trials. During the blank baseline, relax and keep still. When the word appears, follow the instruction; the recording advances automatically.")
                .font(.callout)
            Divider()
            Text(words.joined(separator: "   ")).font(.title3.weight(.medium))
            Text("Only development training words are used. Diagnostic recordings are labelled separately and kept out of model fitting, validation and the final test.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Pace", selection: $check.pace) {
                ForEach(SignalCheckPace.allCases) { pace in
                    Text("\(pace.rawValue) · \(Int(pace.cue)) s per cue").tag(pace)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: check.pace) {
                check.rounds = min(check.rounds, check.pace.maximumRounds(wordCount: words.count))
            }
            HStack {
                Stepper("\(check.rounds) rounds", value: $check.rounds, in: 2...check.pace.maximumRounds(wordCount: words.count))
                    .frame(maxWidth: 220)
                Text("\(count) trials · about \(max(1, Int(ceil(Double(count) * check.pace.trialSeconds / 60)))) minutes").foregroundStyle(.secondary)
            }
            TextField("Visit notes: fit, posture, accent or anything that changed", text: $check.notes)
                .textFieldStyle(.roundedBorder)
            VStack(alignment: .leading, spacing: 8) {
                Picker("Microphone", selection: $audio.selectedDeviceID) {
                    Text("System default").tag(Optional<UInt32>.none)
                    ForEach(audio.devices) { device in Text(device.name).tag(Optional(device.id)) }
                }
                Text("Prefer your Mac’s built-in mic. AirPods as the mic may change Bluetooth audio mode. Audio is recorded for spoken references; Whisper is never used to label silent motion.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            localModel
            if let error = model.trainingError { Text(error).foregroundStyle(.red).font(.callout) }
            if model.motion.status != .streaming {
                Label("Connect AirPods to start", systemImage: "airpodspro").foregroundStyle(.secondary)
            }
            Text("Repeat this visit later after removing and reseating the AirPods. That gives us an independent session check rather than relying on a single fit.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var beginButton: some View {
        let wordCount = model.dataset.experiment.map { SignalCheckPlan.words(in: $0).count } ?? 0
        return Button("Begin recording", systemImage: "record.circle", action: model.startSignalCheck)
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(model.motion.status != .streaming || !model.localWhisper.isConfigured || model.experimentLocked || model.isEvaluating || !model.storageReady || wordCount < 2 || model.recorder.isRecording || model.reading.isReading)
    }

    private func instruction(_ title: String, _ explanation: String, color: Color, symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbol).font(.headline).foregroundStyle(color)
            Text(explanation).font(.callout).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var localModel: some View {
        @Bindable var whisper = model.localWhisper
        return VStack(alignment: .leading, spacing: 8) {
            Label(whisper.isConfigured ? "Local Whisper ready" : "Choose a local Whisper model", systemImage: "desktopcomputer")
                .font(.headline)
            Text(whisper.modelPath.isEmpty ? "No compatible model selected." : URL(fileURLWithPath: whisper.modelPath).lastPathComponent)
                .font(.callout).foregroundStyle(.secondary)
            DisclosureGroup("Model and runtime") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        TextField("GGML model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
                        Button("Choose model…") { chooseFile(modelFile: true) }
                    }
                    HStack {
                        TextField("Local Whisper server executable", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
                        Button("Choose runtime…") { chooseFile(modelFile: false) }
                    }
                    Text("Uses your existing OpenWhispr model and runtime. No download, cloud API or expected-word prompt. Whisper word times are estimates, not phoneme boundaries.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding(.top, 8)
            }
        }
    }

    private var acquisition: some View {
        let check = model.signalCheck
        let showing = check.phase == .cue
        let purpose = check.trial?.purpose ?? .signalMouthed
        let color = !showing ? relaxColor : purpose == .voicedReference ? spokenColor : purpose == .cueOnly ? thinkColor : silentColor
        let title = !showing ? "Relax" : purpose == .voicedReference ? "Speak" : purpose == .cueOnly ? "Think" : "Mouth"
        let symbol = !showing ? "leaf" : purpose == .voicedReference ? "speaker.wave.2" : purpose == .cueOnly ? "brain" : "mouth"
        return VStack(spacing: 18) {
            HStack {
                Text("Trial \(min(check.position + 1, check.total)) of \(check.total)").font(.headline)
                Spacer()
                Text(String(format: "%.1f s", check.secondsRemaining))
                    .font(.system(size: 30, weight: .medium, design: .rounded).monospacedDigit())
                    .foregroundStyle(color)
            }
            ProgressView(value: check.progress).tint(color)
            VStack(spacing: 24) {
                Label(title, systemImage: symbol)
                    .font(.system(size: 34, weight: .semibold))
                Text(showing ? check.trial?.word ?? "" : "Breathe.")
                    .font(.system(size: 120, weight: .medium, design: .serif))
                    .minimumScaleFactor(0.35).lineLimit(1)
                    .padding(.horizontal, 24)
                Text(!showing ? "Let your jaw rest. The next word is hidden."
                     : purpose == .cueOnly ? "Read it in your head. Keep your mouth and head still."
                     : purpose == .voicedReference ? "Say it aloud once, at your own pace."
                     : "Mouth it once, silently, at your own pace.")
                    .font(.title3)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.vertical, 24)
            .background(color.opacity(0.10), in: RoundedRectangle(cornerRadius: 18))
            HStack {
                Text("No keypresses needed. Follow the word and its instruction.")
                    .font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Stop and keep completed trials", action: check.stop)
                    .keyboardShortcut(.cancelAction)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder private var finished: some View {
        if let summary = model.signalCheck.summary {
            Text("\(summary.count) trials saved").font(.title2.weight(.semibold))
            Text("\(summary.voiced) voiced references. \(summary.verified.map { "\($0) matched the displayed word." } ?? "Voice verification is not available.")")
            if let error = summary.transcriptionError {
                Text("Recordings were kept. Local transcription: \(error)").foregroundStyle(.orange)
            }
            Text("Motion, audio and exact cue times are saved. Silent and mouth-still trials retain their instructions as labels; those labels are not evidence that the sensor recognised the word.")
                .foregroundStyle(.secondary)
            HStack {
                Button("Show recordings", systemImage: "folder") {
                    NSWorkspace.shared.activateFileViewerSelecting([summary.directory])
                }
                Button("Record another visit", action: model.signalCheck.reset)
            }
        }
    }

    private func chooseFile(modelFile: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.message = modelFile ? "Choose a local whisper.cpp GGML model (.bin)." : "Choose a local whisper.cpp server executable."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if modelFile { model.localWhisper.modelPath = url.path } else { model.localWhisper.executablePath = url.path }
    }
}
