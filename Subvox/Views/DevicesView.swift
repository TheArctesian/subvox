import CoreAudio
import SwiftUI

struct DevicesView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let motion = model.motion
        let span = motion.latest.map { ($0.t - 8)...$0.t }

        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top, spacing: 16) {
                    motionCard
                    MicrophoneCard()
                }

                GroupBox("Live motion (last 8 s)") {
                    VStack(spacing: 12) {
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
            .padding(20)
        }
    }

    private var motionCard: some View {
        let motion = model.motion
        return GroupBox("AirPods motion") {
            VStack(alignment: .leading, spacing: 8) {
                LabeledContent("Status", value: motion.status.label)
                LabeledContent("Sample rate", value: String(format: "%.1f Hz", motion.sampleRate))
                LabeledContent("Samples", value: "\(motion.sampleCount)")
                if motion.status == .waiting {
                    Text("Put in AirPods Pro and make sure they're the active output on this Mac.")
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
                    Button(audio.isRunning ? "Stop Mic" : "Start Mic") {
                        Task { await model.toggleMic() }
                    }
                    Button("Refresh", systemImage: "arrow.clockwise") { audio.refreshDevices() }
                        .labelStyle(.iconOnly)
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
            .disabled(model.sentenceTraining.isBusy || model.signalCheck.isBusy)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}
