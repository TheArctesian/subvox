import AVFoundation
import SwiftUI

/// The waveform and word intervals use one audio-file clock. Playback lets the user audit
/// estimates directly rather than judging alignment from a motion trace alone.
struct AudioTimingView: View {
    var transcript: LocalWhisperTranscript
    var audioURL: URL
    @State private var player: AVAudioPlayer?
    @State private var playback: Task<Void, Never>?
    @State private var error: String?
    @State private var selected: Int?

    var body: some View {
        if let review = transcript.timingReview {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Audio timing").font(.headline)
                    Spacer()
                    Button("Play reference") { play(start: transcript.sourceAudioStartSeconds ?? 0, duration: duration) }
                    Button("Stop") { stop() }.disabled(player?.isPlaying != true)
                }
                Canvas { context, size in
                    let peak = max(0.0001, review.envelope.map(\.rms).max() ?? 0)
                    let offset = transcript.sourceAudioStartSeconds ?? 0
                    for (i, word) in transcript.words.enumerated() {
                        let x = (word.start - offset) / duration * size.width
                        let width = (word.end - word.start) / duration * size.width
                        context.fill(Path(CGRect(x: x, y: 0, width: max(1, width), height: size.height)),
                                     with: .color(.blue.opacity(selected == i ? 0.3 : 0.08)))
                        var line = Path(); line.move(to: CGPoint(x: x, y: 0)); line.addLine(to: CGPoint(x: x, y: size.height))
                        context.stroke(line, with: .color(.blue.opacity(0.4)), lineWidth: 1)
                    }
                    var wave = Path()
                    for frame in review.envelope {
                        let x = frame.start / duration * size.width
                        let amplitude = min(1, frame.rms / peak) * size.height * 0.46
                        wave.move(to: CGPoint(x: x, y: size.height / 2 - amplitude))
                        wave.addLine(to: CGPoint(x: x, y: size.height / 2 + amplitude))
                    }
                    context.stroke(wave, with: .color(.secondary), lineWidth: 1)
                }.frame(height: 64).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                    .accessibilityLabel("Recorded audio waveform with estimated word boundaries")
                HStack { Text("0 s"); Spacer(); Text(String(format: "%.2f s", duration)) }.font(.caption).foregroundStyle(.secondary)
                ScrollView(.horizontal) {
                    HStack {
                        ForEach(transcript.words.indices, id: \.self) { i in
                            let word = transcript.words[i]
                            Button {
                                selected = i; play(start: word.start, duration: word.end - word.start)
                            } label: {
                                VStack(spacing: 3) {
                                    Text(word.text)
                                    Text(String(format: "%.2f–%.2f", word.start - (transcript.sourceAudioStartSeconds ?? 0),
                                                word.end - (transcript.sourceAudioStartSeconds ?? 0))).font(.caption2)
                                }
                            }.help("Listen to this estimated word interval")
                        }
                    }
                }
                ForEach(review.warnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                if let error { Text(error).font(.caption).foregroundStyle(.orange) }
            }.onDisappear { stop() }
        }
    }

    private var duration: Double {
        guard let review = transcript.timingReview, let last = review.envelope.last else { return 1 }
        return max(0.02, last.start + review.frameSeconds)
    }
    private func stop() { playback?.cancel(); player?.stop() }
    private func play(start: Double, duration: Double) {
        stop(); error = nil
        do {
            let next = try AVAudioPlayer(contentsOf: audioURL)
            next.currentTime = max(0, start); next.play(); player = next
            playback = Task { @MainActor in
                do { try await Task.sleep(for: .seconds(max(0.02, duration))); next.stop() }
                catch { next.stop() }
            }
        } catch { self.error = error.localizedDescription }
    }
}
