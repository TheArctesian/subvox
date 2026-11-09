import AppKit
import CoreAudio
import SwiftUI

struct DevicesView: View {
    @Environment(AppModel.self) private var model

    private var motionHelp: String {
        switch model.motion.status {
        case .unsupported: return "This Mac or headset does not provide headphone motion."
        case .denied: return "Allow Motion & Fitness access in System Settings to use AirPods motion."
        case .error: return "Reconnect your AirPods and check that they are the active audio output."
        case .waiting where !model.motion.isConnected: return "Put in your AirPods and select them as the active audio output."
        case .waiting, .streaming: return "Play audio and choose Head Tracked in the AirPods Spatial Audio menu."
        }
    }

    var body: some View {
        let motion = model.motion
        let span = motion.latest.map { ($0.t - 8)...$0.t }

        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                AppPageHeading(title: "Devices", subtitle: "Check your AirPods signal and microphone before recording.")
                HStack(alignment: .top, spacing: 16) {
                    motionCard
                    MicrophoneCard()
                }

                GroupBox(motion.status == .streaming ? "Live motion (last 8 s)" : "Motion history") {
                  if motion.history.isEmpty {
                    // Empty charts with axis labels only look broken; say what to do instead.
                    ContentUnavailableView("No motion yet", systemImage: "airpods",
                        description: Text(motionHelp))
                        .frame(maxWidth: .infinity, minHeight: 220)
                  } else {
                    VStack(alignment: .leading, spacing: 12) {
                        if motion.status != .streaming {
                            Label("Signal stopped · showing last received samples", systemImage: "pause.circle")
                                .font(.callout).foregroundStyle(.secondary)
                        }
                        SignalPlot(title: "User acceleration (g)", samples: motion.history,
                                   channels: SignalPlot.acceleration, span: span, minScale: 0.01)
                        SignalPlot(title: "Rotation rate (rad/s)", samples: motion.history,
                                   channels: SignalPlot.rotation, span: span, minScale: 0.05)
                        SignalPlot(title: "Attitude (rad)", samples: motion.history,
                                   channels: SignalPlot.attitude, span: span, minScale: 0.1)
                    }
                    .padding(6)
                  }
                }

            }
            .frame(maxWidth: 1100, alignment: .leading).padding(24).frame(maxWidth: .infinity)
        }
    }

    private var motionCard: some View {
        let motion = model.motion
        return GroupBox("AirPods motion") {
            VStack(alignment: .leading, spacing: 8) {
                Label(motion.status == .streaming ? "Receiving motion" : motion.statusText, systemImage: "headphones")
                    .font(.headline).foregroundStyle(motion.status == .streaming ? Color.green : Color.secondary)
                LabeledContent("Sample rate", value: String(format: "%.1f Hz", motion.sampleRate))
                LabeledContent("Samples", value: "\(motion.sampleCount)")
                LabeledContent("Reported sensor", value: motion.latest?.sensorLocation == 1 ? String(localized: "Left AirPod") : motion.latest?.sensorLocation == 2 ? String(localized: "Right AirPod") : String(localized: "Unavailable"))
                if motion.status == .denied {
                    Button("Open Privacy Settings", systemImage: "gearshape") {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Motion") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                }
                if motion.status != .streaming {
                    Text(motionHelp)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .font(.callout.monospacedDigit())
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}

private struct MicrophoneCard: View {
    @State private var starting = false
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var audio = model.audio
        GroupBox("Microphone check") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Input", selection: $audio.selectedDeviceID) {
                    Text("System default").tag(AudioDeviceID?.none)
                    ForEach(audio.devices) { device in
                        Text(device.name).tag(Optional(device.id))
                    }
                }
                HStack {
                    Button(starting ? "Waiting for microphone…" : audio.isRunning ? "Stop test" : "Test microphone", systemImage: audio.isRunning ? "stop.fill" : "mic") {
                        starting = true
                        Task {
                            defer { starting = false }
                            await model.toggleMic()
                        }
                    }.fixedSize().disabled(starting)
                    Button("Refresh", systemImage: "arrow.clockwise") { audio.refreshDevices() }
                        .labelStyle(.iconOnly).frame(minWidth: 28, minHeight: 28)
                        .help("Refresh microphone inputs")
                    LevelMeter(value: Double(audio.level))
                }
                if let name = audio.activeDeviceName {
                    Text("Capturing from \(name)").font(.caption).foregroundStyle(.secondary)
                }
                if let error = audio.error {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                Text("Selecting AirPods as input switches them to the Bluetooth headset profile, which can lower playback quality.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .disabled(model.sentenceTraining.isBusy || model.signalCheck.isBusy || model.representation.isBusy || model.sequenceLearning.isBusy || model.observability.isBusy)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}
