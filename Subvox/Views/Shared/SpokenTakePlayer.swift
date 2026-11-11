import AVFoundation
import SwiftUI

@MainActor @Observable
final class SpokenTakePlayer {
    struct Clip {
        let url: URL
        let range: ClosedRange<Double>
    }

    private(set) var player: AVPlayer?
    private(set) var playing: UUID?
    fileprivate(set) var error: String?

    static func latestPoetryTake(for lines: [PracticeCorpus.Line], language: AppLanguage,
                                in pairs: [SentencePair]) -> (id: UUID, clip: Clip)? {
        latestTake(for: lines, mode: .poetry, language: language, in: pairs)
    }

    static func latestTake(for lines: [PracticeCorpus.Line], mode: SentencePromptKind,
                           language: AppLanguage, pieceID: String? = nil,
                           in pairs: [SentencePair]) -> (id: UUID, clip: Clip)? {
        recentTakes(for: lines, mode: mode, language: language, pieceID: pieceID, in: pairs, limit: 1).first
    }

    static func recentTakes(for lines: [PracticeCorpus.Line], mode: SentencePromptKind,
                            language: AppLanguage, pieceID: String? = nil, in pairs: [SentencePair],
                            limit: Int = 2) -> [(id: UUID, clip: Clip)] {
        let text = lines.map { $0.text(language) }
        let ids = text.map(SentencePrompt.identity)
        let exactText = text.joined(separator: " ")
        var takes: [(id: UUID, clip: Clip)] = []
        guard limit > 0 else { return takes }
        for pair in pairs.reversed()
        where pair.practiceMode == mode && (pair.language ?? "en") == language.tag
            && pair.promptIDs == ids && pair.text == exactText
            && (pieceID == nil || pair.literaturePieceID == pieceID || (mode != .freeText && pair.literaturePieceID == nil)) {
            if let clip = clip(for: pair) {
                takes.append((pair.id, clip))
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
        stop()
        error = nil
        guard !wasPlaying else { return }
        let (url, range) = (clip.url, clip.range)
        let item = AVPlayerItem(url: url)
        item.forwardPlaybackEndTime = CMTime(seconds: range.upperBound, preferredTimescale: 600)
        let next = AVPlayer(playerItem: item)
        player = next
        playing = id
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
            .onChange(of: model.referenceSpeaker.playing) {
                if model.referenceSpeaker.playing != nil { playback.stop() }
            }
            .onDisappear(perform: playback.stop)
            .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { notification in
                if notification.object as? AVPlayerItem === playback.player?.currentItem { playback.stop() }
            }
            .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemFailedToPlayToEndTime)) { notification in
                if notification.object as? AVPlayerItem === playback.player?.currentItem {
                    playback.stop()
                    playback.error = String(localized: "The saved audio could not be played.")
                }
            }
    }
}
