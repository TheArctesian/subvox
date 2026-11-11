import AVFoundation
import Foundation
import Observation

struct VoicePracticeTarget: Codable, Hashable {
    let mode: String
    let language: AppLanguage
    let itemID: String
    let text: String
}

struct VoicePracticeRun: Codable, Identifiable, Equatable {
    let id: UUID
    let target: VoicePracticeTarget
    let createdAt: Date
    let duration: TimeInterval
}

/// Audio and its metadata become visible together when the finished run directory is renamed.
struct VoicePracticeStore {
    let root: URL

    func begin(id: UUID) throws -> URL {
        let staging = root.appending(path: ".recording-\(id.uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        return staging
    }

    func audioURL(in directory: URL) -> URL {
        directory.appending(path: "audio.m4a", directoryHint: .notDirectory)
    }

    func audioURL(for run: VoicePracticeRun) -> URL {
        audioURL(in: directory(for: run.id))
    }

    func finish(_ run: VoicePracticeRun, in staging: URL) throws {
        let metadata = try JSONEncoder().encode(run)
        try metadata.write(to: staging.appending(path: "run.json"), options: .atomic)
        try FileManager.default.moveItem(at: staging, to: directory(for: run.id))
    }

    func savedRuns() -> [VoicePracticeRun] {
        let directories = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil, options: .skipsHiddenFiles
        )) ?? []
        let decoder = JSONDecoder()
        return directories.compactMap { directory -> VoicePracticeRun? in
            guard let id = UUID(uuidString: directory.lastPathComponent),
                  let data = try? Data(contentsOf: directory.appending(path: "run.json")),
                  let run = try? decoder.decode(VoicePracticeRun.self, from: data),
                  run.id == id, run.duration.isFinite, run.duration > 0,
                  FileManager.default.isReadableFile(atPath: audioURL(in: directory).path) else { return nil }
            return run
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    private func directory(for id: UUID) -> URL {
        root.appending(path: id.uuidString, directoryHint: .isDirectory)
    }
}

protocol VoicePracticeRecorder: AnyObject {
    var delegate: AVAudioRecorderDelegate? { get set }
    func record() -> Bool
    func stop()
}

extension AVAudioRecorder: VoicePracticeRecorder {}

/// A voice run uses the system microphone input and stays in the private practice store.
@MainActor @Observable
final class VoicePractice: NSObject, AVAudioRecorderDelegate, AVAudioPlayerDelegate {
    private struct Recording {
        let id: UUID
        let target: VoicePracticeTarget
        let createdAt: Date
        let directory: URL
        let recorder: any VoicePracticeRecorder
    }

    private let store: VoicePracticeStore
    private let requestAccess: @MainActor () async -> Bool
    private let makeRecorder: @MainActor (URL) throws -> any VoicePracticeRecorder
    private var requestID: UUID?
    private var recording: Recording?
    @ObservationIgnored private var player: AVAudioPlayer?

    private(set) var runs: [VoicePracticeRun]
    private(set) var playingRunID: UUID?
    private(set) var error: String?
    var recordingTarget: VoicePracticeTarget? { recording?.target }
    var starting: Bool { requestID != nil }
    var isBusy: Bool { starting || recording != nil }

    init(root: URL = SentenceStore.directory.appending(path: "VoicePractice", directoryHint: .isDirectory),
         requestAccess: @escaping @MainActor () async -> Bool = { await AVCaptureDevice.requestAccess(for: .audio) },
         makeRecorder: @escaping @MainActor (URL) throws -> any VoicePracticeRecorder = { url in
             try AVAudioRecorder(url: url, settings: [
                 AVFormatIDKey: kAudioFormatMPEG4AAC,
                 AVSampleRateKey: 44_100,
                 AVNumberOfChannelsKey: 1,
                 AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
             ])
         }) {
        let store = VoicePracticeStore(root: root)
        self.store = store
        self.requestAccess = requestAccess
        self.makeRecorder = makeRecorder
        runs = store.savedRuns()
        super.init()
    }

    func recentRuns(for target: VoicePracticeTarget) -> [VoicePracticeRun] {
        runs.filter { $0.target == target }
    }

    func start(_ target: VoicePracticeTarget) async {
        guard !isBusy, !target.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        stopPlayback()
        error = nil
        let id = UUID()
        requestID = id
        let authorized = await requestAccess()
        guard requestID == id else { return }
        defer { requestID = nil }
        guard !Task.isCancelled else { return }
        guard authorized else {
            error = String(localized: "Allow microphone access for Subvox in System Settings under Privacy & Security.")
            return
        }

        do {
            let directory = try store.begin(id: id)
            do {
                let recorder = try makeRecorder(store.audioURL(in: directory))
                recorder.delegate = self
                guard recorder.record() else {
                    try? FileManager.default.removeItem(at: directory)
                    error = String(localized: "The microphone could not start recording. Check your Mac’s sound input.")
                    return
                }
                recording = Recording(id: id, target: target, createdAt: .now, directory: directory, recorder: recorder)
            } catch {
                try? FileManager.default.removeItem(at: directory)
                throw error
            }
        } catch {
            self.error = String(localized: "Subvox couldn’t start the recording. \(error.localizedDescription)")
        }
    }

    func finish() {
        guard let current = takeRecording() else { return }
        current.recorder.stop()
        do {
            let audio = try AVAudioFile(forReading: store.audioURL(in: current.directory))
            let duration = Double(audio.length) / audio.processingFormat.sampleRate
            guard duration.isFinite, duration > 0 else {
                try? FileManager.default.removeItem(at: current.directory)
                error = String(localized: "No audio was recorded. Try another take.")
                return
            }
            let run = VoicePracticeRun(id: current.id, target: current.target, createdAt: current.createdAt, duration: duration)
            try store.finish(run, in: current.directory)
            runs.insert(run, at: 0)
        } catch {
            try? FileManager.default.removeItem(at: current.directory)
            self.error = String(localized: "Subvox couldn’t save this recording. \(error.localizedDescription)")
        }
    }

    func cancel() {
        requestID = nil
        guard let current = takeRecording() else { return }
        current.recorder.stop()
        try? FileManager.default.removeItem(at: current.directory)
    }

    private func takeRecording() -> Recording? {
        let current = recording
        recording = nil
        current?.recorder.delegate = nil
        return current
    }

    func prepareForTermination() {
        requestID = nil
        finish()
        stopPlayback()
    }

    func play(_ run: VoicePracticeRun) {
        guard !isBusy, runs.contains(run) else { return }
        stopPlayback()
        error = nil
        do {
            let player = try AVAudioPlayer(contentsOf: store.audioURL(for: run))
            player.delegate = self
            if player.play() {
                self.player = player
                playingRunID = run.id
            } else {
                error = String(localized: "This recording couldn’t be played.")
            }
        } catch {
            self.error = String(localized: "This recording couldn’t be played. \(error.localizedDescription)")
        }
    }

    func stopPlayback() {
        player?.delegate = nil
        player?.stop()
        player = nil
        playingRunID = nil
    }

    func clearError() { error = nil }

    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let id = ObjectIdentifier(recorder)
        Task { @MainActor in
            guard let active = recording, ObjectIdentifier(active.recorder) == id else { return }
            if flag {
                finish()
            } else {
                cancel()
                error = String(localized: "The microphone stopped recording. Check your Mac’s sound input and try again.")
            }
        }
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        let id = ObjectIdentifier(recorder)
        Task { @MainActor in
            guard let active = recording, ObjectIdentifier(active.recorder) == id else { return }
            cancel()
            self.error = String(localized: "The recording stopped before it could be saved.")
        }
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let id = ObjectIdentifier(player)
        Task { @MainActor in
            guard let active = self.player, ObjectIdentifier(active) == id else { return }
            stopPlayback()
            if !flag { error = String(localized: "Playback stopped early.") }
        }
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        let id = ObjectIdentifier(player)
        Task { @MainActor in
            guard let active = self.player, ObjectIdentifier(active) == id else { return }
            stopPlayback()
            self.error = String(localized: "This recording couldn’t be played.")
        }
    }
}
