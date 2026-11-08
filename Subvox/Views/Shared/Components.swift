import SwiftUI

/// Shared page hierarchy; the window chrome and system colours stay native.
struct AppPageHeading: View {
    var title: LocalizedStringKey
    var subtitle: LocalizedStringKey
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.largeTitle.weight(.semibold)).accessibilityAddTraits(.isHeader)
            Text(subtitle).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct RecordingCueStatus: View {
    var timer: String
    var title: String
    var color: Color
    var body: some View {
        VStack(spacing: 8) {
            Text(timer).font(.system(size: 20, weight: .medium).monospacedDigit()).foregroundStyle(.primary)
            HStack(spacing: 7) {
                Circle().fill(color).frame(width: 9, height: 9)
                Text(title).font(.system(size: 17, weight: .semibold))
            }
            .padding(.horizontal, 13).padding(.vertical, 7)
            .background(color.opacity(0.12), in: Capsule())
        }
    }
}

/// A bounded content region keeps page titles and the live signal visible at
/// the minimum window size, even for intrinsically tall empty states or forms.
struct AppPageLayout<Header: View, Content: View>: View {
    /// Practice pages leave the live plot out: the toolbar badge already shows the AirPods state,
    /// and the room goes to the text being practised.
    var showsMotion = true
    @ViewBuilder var header: () -> Header
    @ViewBuilder var content: () -> Content
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                header().padding(.horizontal, 24).frame(height: 108)
                Divider()
                content().frame(width: geometry.size.width, height: max(1, geometry.size.height - (showsMotion ? 286 : 109))).clipped()
                if showsMotion {
                    Divider()
                    LiveMotionPanel(plotHeight: 56).padding(.horizontal, 16).frame(height: 176).clipped()
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .top).clipped()
        }
    }
}

/// Multi-channel line plot of motion samples, drawn with Canvas so it keeps up at sensor rate.
struct SignalPlot: View {
    struct Channel {
        var name: String
        var color: Color
        var value: KeyPath<MotionSample, Double>
    }

    var title: String
    var samples: [MotionSample]
    var channels: [Channel]
    /// Visible time span; defaults to the extent of `samples`.
    var span: ClosedRange<TimeInterval>?
    var minScale: Double = 0.02
    var height: CGFloat = 90
    var showsHeader = true

    static let acceleration = [
        Channel(name: "x", color: .red, value: \.ax),
        Channel(name: "y", color: .green, value: \.ay),
        Channel(name: "z", color: .blue, value: \.az),
    ]
    static let rotation = [
        Channel(name: "x", color: .red, value: \.rx),
        Channel(name: "y", color: .green, value: \.ry),
        Channel(name: "z", color: .blue, value: \.rz),
    ]
    static let attitude = [
        Channel(name: "pitch", color: .orange, value: \.pitch),
        Channel(name: "roll", color: .purple, value: \.roll),
        Channel(name: "yaw", color: .teal, value: \.yaw),
    ]

    var body: some View {
        let range = span ?? ((samples.first?.t ?? 0)...(samples.last?.t ?? 1))
        let visible = samples.filter { range.contains($0.t) }
        let scale = max(minScale, channels.flatMap { c in visible.map { abs($0[keyPath: c.value]) } }.max() ?? 0)

        VStack(alignment: .leading, spacing: 4) {
            if showsHeader {
                HStack(spacing: 10) {
                    Text(title).font(.caption.weight(.semibold))
                    ForEach(channels, id: \.name) { channel in
                        Label(channel.name, systemImage: "circle.fill")
                            .labelStyle(.titleAndIcon)
                            .font(.caption2)
                            .foregroundStyle(channel.color)
                    }
                    Spacer()
                    Text("±\(scale, specifier: "%.3f")")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            Canvas { context, size in
                let mid = size.height / 2
                var axis = Path()
                axis.move(to: CGPoint(x: 0, y: mid))
                axis.addLine(to: CGPoint(x: size.width, y: mid))
                context.stroke(axis, with: .color(.secondary.opacity(0.3)), lineWidth: 0.5)

                let width = max(range.upperBound - range.lowerBound, 1e-3)
                for channel in channels {
                    var path = Path()
                    for (i, s) in visible.enumerated() {
                        let point = CGPoint(
                            x: (s.t - range.lowerBound) / width * size.width,
                            y: mid - s[keyPath: channel.value] / scale * (mid - 2))
                        if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
                    }
                    context.stroke(path, with: .color(channel.color), lineWidth: 1.2)
                }
            }
            .frame(height: height)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}

struct LevelMeter: View {
    var value: Double
    var marker: Double?
    var tint: Color = .green

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(tint.gradient)
                    .frame(width: geo.size.width * min(max(value, 0), 1))
                if let marker {
                    Rectangle().fill(.primary)
                        .frame(width: 2)
                        .offset(x: geo.size.width * min(max(marker, 0), 1) - 1)
                }
            }
        }
        .frame(height: 8)
        .animation(.linear(duration: 0.08), value: value)
    }
}

/// Mouse alternative to holding Space.
struct HoldToCaptureButton: View {
    var title: String
    var isCapturing: Bool
    var isEnabled = true
    var onBegin: () -> Void
    var onEnd: () -> Void

    @State private var pressed = false

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: isCapturing ? "waveform" : "mouth")
                .font(.system(size: 30, weight: .medium))
                .symbolEffect(.variableColor.iterative, isActive: isCapturing)
            Text(isCapturing ? "Capturing…" : title)
                .font(.headline)
            Text("Hold Space or press and hold here")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 120)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(isCapturing ? Color.red.opacity(0.18) : Color.accentColor.opacity(0.10)))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(isCapturing ? Color.red : Color.accentColor.opacity(0.4), lineWidth: isCapturing ? 2 : 1))
        .contentShape(Rectangle())
        .opacity(isEnabled ? 1 : 0.4)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard isEnabled, !pressed else { return }
                    pressed = true
                    onBegin()
                }
                .onEnded { _ in
                    guard pressed else { return }
                    pressed = false
                    onEnd()
                })
    }
}

struct MotionStatusBadge: View {
    var status: HeadphoneMotionService.Status
    var rate: Double
    var text: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(status == .streaming ? "AirPods · \(Int(rate.rounded())) Hz" : text ?? status.label)
                .font(.callout.monospacedDigit())
        }
        .padding(.horizontal, 8)
    }

    private var color: Color {
        switch status {
        case .streaming: .green
        case .waiting: .orange
        default: .red
        }
    }
}

/// Shared channel choices for inspecting the original raw motion traces.
enum MotionPlotChannel: String, CaseIterable, Identifiable {
    case rotation = "Rotation (rad/s)"
    case acceleration = "Acceleration (g)"
    case attitude = "Attitude (rad)"
    var id: String { rawValue }
    var channels: [SignalPlot.Channel] {
        switch self {
        case .rotation: SignalPlot.rotation
        case .acceleration: SignalPlot.acceleration
        case .attitude: SignalPlot.attitude
        }
    }
    var minScale: Double {
        switch self {
        case .rotation: 0.05
        case .acceleration: 0.01
        case .attitude: 0.1
        }
    }
}

struct LiveMotionPanel: View {
    var plotHeight: CGFloat = 75
    @Environment(AppModel.self) private var model
    @State private var channel = MotionPlotChannel.rotation

    var body: some View {
        let motion = model.motion
        GroupBox("Live motion") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Picker("Channels", selection: $channel) {
                        ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .labelsHidden()
                    Text(String(format: "%.1f Hz", motion.sampleRate)).font(.caption.monospacedDigit())
                }
                SignalPlot(title: channel.rawValue, samples: motion.history, channels: channel.channels,
                           span: motion.latest.map { ($0.t - 8)...$0.t }, minScale: channel.minScale, height: plotHeight)
            }.padding(6)
        }
    }
}
