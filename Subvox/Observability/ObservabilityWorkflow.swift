import Foundation

/// Records the observability protocol (motion only) and scores it locally. Sessions live in their
/// own folder, so they never enter sentence training data or neural dataset snapshots.
@MainActor @Observable final class ObservabilityWorkflow {
    enum Phase { case idle, ready, relax, recording, ended, analysing }
    static let relaxSeconds = 2.0
    static let cueSeconds = 3.0

    private(set) var phase: Phase = .idle
    private(set) var error: String?
    private(set) var session: ObservabilitySession?
    private(set) var history: [ObservabilitySession] = []
    private(set) var index = 0
    private(set) var secondsRemaining = 0.0
    var blocks = 12
    var permutations = 499

    private let root: URL
    private let clock: () -> Double
    private var recorder: SessionRecorder?
    private var task: Task<Void, Never>?
    private var buffer: [MotionSample] = []
    private var latest: MotionSample?
    private var cueStart: Double?
    private var source: Int?

    var isAcquiring: Bool { [.ready, .relax, .recording, .ended].contains(phase) }
    var isBusy: Bool { phase != .idle }
    var current: ObservabilitySession.Planned? { session.flatMap { $0.planned.indices.contains(index) ? $0.planned[index] : nil } }
    var total: Int { session?.planned.count ?? 0 }
    var lastTake: ObservabilityTake? { session?.takes.last }
    var takeIssue: String? { phase == .ended ? lastTake?.discardedReason : nil }
    var canRedo: Bool { phase == .recording || phase == .ended }
    var canAdvance: Bool { phase == .ready || phase == .ended }
    var latestReport: ObservabilitySession? { history.first { $0.report != nil } }
    var estimatedMinutes: Int { Int(ceil(Double(blocks * ObservabilityGesture.allCases.count) * (Self.relaxSeconds + Self.cueSeconds + 1.5) / 60)) }
    var hasFreshMotion: Bool { latest.map { clock() - $0.t >= 0 && clock() - $0.t < 0.3 } ?? false }

    init(root: URL? = nil, clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.root = root ?? SentenceStore.directory.appending(path: "Observability")
        self.clock = clock
        loadHistory()
    }

    func start(audio: AudioService, recorder: SessionRecorder, seed: UInt64 = .random(in: 1...UInt64.max)) {
        guard !isBusy, !recorder.isRecording else { return }
        guard hasFreshMotion else { error = "Waiting for fresh AirPods motion."; return }
        audio.stop()
        recorder.condition = .observability
        recorder.prompt = "Mouth observability check; motion only, no camera/audio/video"
        guard let recording = recorder.start(audioSink: nil, cameraPoints: false) else { error = recorder.error; return }
        self.recorder = recorder
        session = ObservabilitySession(session: recording, seed: seed, planned: ObservabilitySession.plan(blocks: blocks, seed: seed))
        index = 0; error = nil; phase = .ready; save()
    }

    func ingest(_ sample: MotionSample) {
        latest = sample
        guard phase == .relax || phase == .recording else { return }
        let changed = source.map { $0 != sample.sensorLocation } ?? false
        let gap = buffer.last.map { sample.t - $0.t > 0.12 || sample.t <= $0.t } ?? false
        buffer.append(sample)
        guard changed || gap else { return }
        if phase == .recording { finishTake(reason: "Motion source changed or had a gap; redo this take.") }
        else { task?.cancel(); buffer = []; phase = .ready; error = "Motion source changed; start the take again." }
    }

    /// Ready → relax → timed cue → saved. From a saved take, continuing starts the next take directly.
    func next() {
        switch phase {
        case .ready: startTake()
        case .ended:
            guard takeIssue == nil else { redo(); return }
            if index + 1 < total { index += 1; phase = .ready; startTake() } else { finish() }
        default: break
        }
    }

    func redo() {
        guard canRedo else { return }
        if phase == .recording { finishTake(reason: "Discarded by Redo") }
        else if let i = session?.takes.indices.last, session?.takes[i].discardedReason == nil { session?.takes[i].discardedReason = "Discarded by Redo" }
        phase = .ready; error = nil; save()
    }

    func stop() {
        task?.cancel()
        if phase == .recording { finishTake(reason: "Stopped during capture") }
        if isAcquiring { finish() } else if phase != .analysing { phase = .idle }
    }

    func reanalyse(_ id: UUID) {
        guard !isBusy, let stored = history.first(where: { $0.id == id }) else { return }
        session = stored; analyse()
    }

    private func startTake() {
        guard phase == .ready, current != nil else { return }
        guard hasFreshMotion, let latest else { error = "Waiting for fresh AirPods motion."; return }
        buffer = []; source = latest.sensorLocation; error = nil; phase = .relax
        task = Task {
            do {
                let relaxEnd = clock() + Self.relaxSeconds
                while clock() < relaxEnd { try Task.checkCancellation(); secondsRemaining = max(0, relaxEnd - clock()); try await Task.sleep(for: .milliseconds(40)) }
                beginCue(at: clock())
                let end = clock() + Self.cueSeconds
                while phase == .recording, clock() < end {
                    try Task.checkCancellation(); secondsRemaining = max(0, end - clock())
                    if !hasFreshMotion { finishTake(reason: "Motion stopped; redo this take."); return }
                    try await Task.sleep(for: .milliseconds(40))
                }
                if phase == .recording { finishTake() }
            } catch {}
        }
    }

    func beginCue(at time: Double) {
        guard phase == .relax, let current else { return }
        cueStart = time; phase = .recording
        recorder?.logMarker(uptime: time, kind: "observability-start", label: current.gesture.rawValue)
    }

    func finishTake(reason: String? = nil) {
        guard phase == .recording, let cueStart, let current else { return }
        task?.cancel()
        let end = clock()
        session?.takes.append(.init(gesture: current.gesture, block: current.block, samples: buffer.filter { $0.t <= end },
                                    cueStart: cueStart, cueEnd: end, discardedReason: reason))
        recorder?.logMarker(uptime: end, kind: reason == nil ? "observability-end" : "observability-interrupted", label: current.gesture.rawValue)
        self.cueStart = nil; buffer = []; secondsRemaining = 0; error = reason; phase = .ended; save()
    }

    private func finish() {
        guard let session else { phase = .idle; return }
        _ = recorder?.stop(sessionID: session.session.id); recorder = nil
        self.session?.endedUnix = Date().timeIntervalSince1970
        save(); analyse()
    }

    private func analyse() {
        guard let session else { phase = .idle; return }
        let takes = session.validTakes, permutations = permutations, seed = session.seed
        phase = .analysing
        task = Task {
            let report = await Task.detached(priority: .userInitiated) {
                ObservabilityAnalysis.analyse(takes, permutations: permutations, seed: seed)
            }.value
            self.session?.report = report
            save(); loadHistory(); phase = .idle
        }
    }

    private func save() {
        guard let session else { return }
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try JSONEncoder().encode(session).write(to: root.appending(path: "\(session.id).json"), options: .atomic)
        } catch { self.error = error.localizedDescription }
    }

    private func loadHistory() {
        let files = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        history = files.filter { $0.pathExtension == "json" }
            .compactMap { try? JSONDecoder().decode(ObservabilitySession.self, from: Data(contentsOf: $0)) }
            .sorted { $0.startedUnix > $1.startedUnix }
    }
}
