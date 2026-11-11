import AVFoundation
import SwiftUI

@MainActor @Observable
final class SpokenTakePlayer {
    struct Clip {
        let url: URL
        let range: ClosedRange<Double>
    }

    struct Take {
        let id: UUID
        let clip: Clip
        let createdAt: Date
    }

    private(set) var player: AVPlayer?
    private(set) var playing: UUID?
    private(set) var isComparing = false
    fileprivate(set) var error: String?
    private var queue: [Take] = []
    private var position = 0
    private var loops = false

    static func latestPoetryTake(for lines: [PracticeCorpus.Line], language: AppLanguage,
                                in pairs: [SentencePair]) -> Take? {
        latestTake(for: lines, mode: .poetry, language: language, in: pairs)
    }

    static func latestTake(for lines: [PracticeCorpus.Line], mode: SentencePromptKind,
                           language: AppLanguage, pieceID: String? = nil,
                           in pairs: [SentencePair]) -> Take? {
        recentTakes(for: lines, mode: mode, language: language, pieceID: pieceID, in: pairs, limit: 1).first
    }

    static func recentTakes(for lines: [PracticeCorpus.Line], mode: SentencePromptKind,
                            language: AppLanguage, pieceID: String? = nil, in pairs: [SentencePair],
                            limit: Int = 2) -> [Take] {
        let text = lines.map { $0.text(language) }
        let ids = text.map(SentencePrompt.identity)
        let exactText = text.joined(separator: " ")
        var takes: [Take] = []
        guard limit > 0 else { return takes }
        for pair in pairs.reversed()
        where pair.practiceMode == mode && (pair.language ?? "en") == language.tag
            && pair.promptIDs == ids && pair.text == exactText
            && (pieceID == nil || pair.literaturePieceID == pieceID || (mode != .freeText && pair.literaturePieceID == nil)) {
            if let clip = clip(for: pair) {
                takes.append(Take(id: pair.id, clip: clip, createdAt: pair.createdAt))
                if takes.count == limit { break }
            }
        }
        return takes
    }

    static func clip(for pair: SentencePair) -> Clip? {
        guard let spoken = pair.spoken, let directory = pair.directory,
              let sessionStart = pair.sessionStartUptime, let audioOffset = pair.audioOffset else { return nil }
        let origin = sessionStart + audioOffset
        let start = spoken.cueStart - origin
        let end = spoken.cueEnd - origin
        let url = URL(fileURLWithPath: directory).appending(path: "audio.caf")
        guard start.isFinite, end.isFinite, start >= 0, end > start,
              FileManager.default.isReadableFile(atPath: url.path) else { return nil }
        return Clip(url: url, range: start...end)
    }

    func toggle(_ clip: Clip, id: UUID) {
        let wasPlaying = playing == id
        if wasPlaying { stop() }
        else { play([Take(id: id, clip: clip, createdAt: .distantPast)], loops: false) }
    }

    func play(_ takes: [Take], loops: Bool) {
        stop()
        error = nil
        guard !takes.isEmpty else { return }
        queue = takes
        self.loops = loops
        isComparing = takes.count > 1
        playCurrentTake()
    }

    func setLooping(_ enabled: Bool) { loops = enabled }

    func finished(_ item: AVPlayerItem) {
        guard item === player?.currentItem else { return }
        let next = position + 1
        guard next < queue.count || loops else { stop(); return }
        position = next % queue.count
        playCurrentTake()
    }

    private func playCurrentTake() {
        player?.pause()
        let take = queue[position]
        let (url, range) = (take.clip.url, take.clip.range)
        let item = AVPlayerItem(url: url)
        item.forwardPlaybackEndTime = CMTime(seconds: range.upperBound, preferredTimescale: 600)
        let next = AVPlayer(playerItem: item)
        player = next
        playing = take.id
        next.seek(to: CMTime(seconds: range.lowerBound, preferredTimescale: 600),
                  toleranceBefore: .zero, toleranceAfter: .zero) { finished in
            Task { @MainActor in
                guard self.player === next else { return }
                if finished { next.play() }
                else { self.stop(); self.error = String(localized: "The saved audio could not be played.") }
            }
        }
    }

    func stop() {
        player?.pause()
        player = nil
        playing = nil
        isComparing = false
        queue.removeAll()
        position = 0
        loops = false
    }
}

struct SpokenPlaybackLifecycle: ViewModifier {
    @Environment(AppModel.self) private var model
    let playback: SpokenTakePlayer

    func body(content: Content) -> some View {
        content
            .onChange(of: model.sentenceTraining.canStartRecording) {
                if !model.sentenceTraining.canStartRecording { playback.stop() }
            }
            .onChange(of: model.voicePractice.isBusy) {
                if model.voicePractice.isBusy { playback.stop() }
            }
            .onChange(of: model.voicePractice.playingRunID) {
                if model.voicePractice.playingRunID != nil { playback.stop() }
            }
            .onChange(of: playback.playing) {
                if playback.playing != nil { model.voicePractice.stopPlayback() }
            }
            .onChange(of: model.referenceSpeaker.playing) {
                if model.referenceSpeaker.playing != nil { playback.stop() }
            }
            .onDisappear(perform: playback.stop)
            .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { notification in
                if let item = notification.object as? AVPlayerItem { playback.finished(item) }
            }
            .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemFailedToPlayToEndTime)) { notification in
                if notification.object as? AVPlayerItem === playback.player?.currentItem {
                    playback.stop()
                    playback.error = String(localized: "The saved audio could not be played.")
                }
            }
    }
}

/// Keeps the example, local microphone takes and spoken halves of paired takes on one section.
struct SectionRehearsalControls: View {
    @Environment(AppModel.self) private var model
    @State private var repeats = false
    @State private var referenceTask: Task<Void, Never>?
    let target: VoicePracticeTarget
    let title: String
    let pairedTakes: [SpokenTakePlayer.Take]
    let playback: SpokenTakePlayer

    var body: some View {
        let micTakes = model.voicePractice.recentRuns(for: target).prefix(2).map { run in
            SpokenTakePlayer.Take(id: run.id, clip: .init(url: model.voicePractice.audioURL(for: run), range: 0...run.duration),
                                  createdAt: run.createdAt)
        }
        let takes = Array((pairedTakes + micTakes).sorted { $0.createdAt > $1.createdAt }.prefix(2))
        let referenceKey = "\(target.mode)/\(target.language.tag)/\(target.itemID)"
        let referencePlaying = model.referenceSpeaker.playing == referenceKey
        let playingHere = takes.contains { $0.id == playback.playing }
        return VStack(alignment: .leading, spacing: 7) {
            ViewThatFits(in: .horizontal) {
                HStack { listenActions(takes, referenceKey: referenceKey, referencePlaying: referencePlaying, playingHere: playingHere) }
                VStack(alignment: .leading, spacing: 6) {
                    listenActions(takes, referenceKey: referenceKey, referencePlaying: referencePlaying, playingHere: playingHere)
                }
            }
            HStack {
                Toggle("Loop", isOn: $repeats)
                    .help("Repeat the example, the take, or both takes in a comparison.")
                    .accessibilityLabel("Loop \(title)")
                Button("Record with mic", systemImage: "mic") {
                    stopReference(referenceKey)
                    playback.stop()
                    Task { await model.startVoicePractice(target) }
                }
                .accessibilityLabel("Record \(title) with microphone")
                .disabled(!model.canStartVoicePractice)
            }
            if playingHere && playback.isComparing {
                Text(playback.playing == takes.first?.id ? "Playing latest take" : "Playing previous take")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if model.voicePractice.errorTarget == target, let error = model.voicePractice.error {
                Label(error, systemImage: "exclamationmark.circle").font(.callout).foregroundStyle(.orange)
            }
        }
        .disabled(model.navigationLocked)
        .onChange(of: repeats) {
            if playingHere { playback.setLooping(repeats) }
            if referencePlaying { startReference(referenceKey) }
        }
        .onChange(of: model.voicePractice.isBusy) {
            if model.voicePractice.isBusy { stopReference(referenceKey) }
        }
        .onChange(of: target) { stopReference(referenceKey) }
        .onDisappear { stopReference(referenceKey) }
    }

    @ViewBuilder private func listenActions(_ takes: [SpokenTakePlayer.Take], referenceKey: String,
                                           referencePlaying: Bool, playingHere: Bool) -> some View {
        Button(referencePlaying ? "Stop example" : "Hear example", systemImage: referencePlaying ? "stop.fill" : "speaker.wave.2") {
            if referencePlaying { stopReference(referenceKey) }
            else { startReference(referenceKey) }
        }
        .accessibilityLabel(referencePlaying ? "Stop example for \(title)" : "Hear example for \(title)")
        if let latest = takes.first {
            takeButton(latest, label: takes.count == 1 ? "Play my take" : "Play latest")
        }
        if takes.count == 2 {
            takeButton(takes[1], label: "Play previous")
            Button(playingHere && playback.isComparing ? "Stop comparison" : "Compare takes",
                   systemImage: playingHere && playback.isComparing ? "stop.fill" : "arrow.left.arrow.right") {
                if playingHere && playback.isComparing { playback.stop() }
                else {
                    model.referenceSpeaker.stop()
                    playback.play(takes, loops: repeats)
                }
            }
            .help("Hear the latest take, then the previous take.")
        }
    }

    private func takeButton(_ take: SpokenTakePlayer.Take, label: LocalizedStringKey) -> some View {
        let playing = !playback.isComparing && playback.playing == take.id
        return Button(playing ? "Stop take" : label, systemImage: playing ? "stop.fill" : "play.fill") {
            if playing { playback.stop() }
            else {
                model.referenceSpeaker.stop()
                playback.play([take], loops: repeats)
            }
        }
    }

    private func startReference(_ key: String) {
        referenceTask?.cancel()
        playback.stop()
        referenceTask = Task {
            await model.referenceSpeaker.rehearse(target.text, language: target.language.tag, slower: false,
                                                  repeats: repeats, key: key)
        }
    }

    private func stopReference(_ key: String) {
        referenceTask?.cancel()
        if model.referenceSpeaker.playing == key { model.referenceSpeaker.stop() }
    }
}
