import Foundation
import Observation

/// A separate acquisition and analysis path. Diagnostic takes never enter SentenceTrainingData.
@MainActor @Observable
final class SignalCheckWorkflow {
    enum Phase { case setup, reseat, controlSetup, controlBreak, preparing, ready, relax, recording, ended, processing, finished }
    private(set) var phase: Phase = .setup
    private(set) var run: SignalCheckRun?
    let camera = CameraPoseService()
    private(set) var comparison: ControlledSignalComparison?
    private(set) var controlledActive = false
    private(set) var teacherHistoryCount = 0
    @ObservationIgnored private var teacherHistory: [SignalCheckRun] = []
    private var controlBreakDeadline = 0.0
    @ObservationIgnored private var latestMotion: MotionSample?
    private(set) var reference: FrozenSignalReference?
    private(set) var referenceLink: SignalReferenceLink?
    private(set) var referenceError: String?
    private(set) var error: String?
    private(set) var storageReady = true
    private(set) var index = 0
    private(set) var secondsRemaining = 0.0
    private(set) var processingText = ""
    private(set) var plan: [SignalCheckPrompt] = []
    var isAcquiring: Bool { [.reseat, .controlSetup, .controlBreak, .ready, .relax, .recording, .ended].contains(phase) }
    var showsCaptureCue: Bool { [.ready, .relax, .recording, .ended].contains(phase) }
    var isBusy: Bool { isAcquiring || phase == .preparing || phase == .processing }
    var current: SignalCheckPrompt? { plan.indices.contains(index) ? plan[index] : nil }
    var currentText: String { current.map { SignalCheckRun.sentences[$0.sentence] } ?? "" }
    var latestTake: SignalCheckTake? { run?.takes.last(where: { !$0.discarded }) }
    var canRedo: Bool { phase == .recording || phase == .ended }
    var canPrepareReseat: Bool {
        !isBusy && storageReady && referenceError == nil && (reference != nil || (run?.endedAt != nil && run?.reference == nil))
    }
    @ObservationIgnored private let storeURL: URL
    @ObservationIgnored private let referenceURL: URL
    @ObservationIgnored private let comparisonURL: URL
    @ObservationIgnored private let clock: () -> Double
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var buffer: [MotionSample] = []
    @ObservationIgnored private var cueStart: Double?
    @ObservationIgnored private var recorder: SessionRecorder?
    @ObservationIgnored private var canRecord: () -> Bool = { false }

    init(storeURL: URL? = nil, clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.clock = clock
        self.storeURL = storeURL ?? SentenceStore.directory.appending(path: "signal-check.json")
        comparisonURL = self.storeURL.deletingLastPathComponent().appending(path: "controlled-signal-test.json")
        referenceURL = self.storeURL.deletingLastPathComponent().appending(path: "frozen-signal-reference.json")
        if FileManager.default.fileExists(atPath: referenceURL.path) {
            do {
                let frozen = try FrozenSignalReference.load(from: referenceURL)
                reference = frozen; referenceLink = try frozen.link()
            } catch { referenceError = error.localizedDescription }
        }
        if FileManager.default.fileExists(atPath: comparisonURL.path) {
            do { comparison = try JSONDecoder().decode(ControlledSignalComparison.self, from: Data(contentsOf: comparisonURL)) }
            catch { self.error = "Could not load the controlled comparison: " + error.localizedDescription }
        }
        if FileManager.default.fileExists(atPath: self.storeURL.path) {
            do { run = try JSONDecoder().decode(SignalCheckRun.self, from: Data(contentsOf: self.storeURL)) }
            catch { self.error = "Could not load the previous signal check: " + error.localizedDescription; storageReady = false }
        }
        if let run, comparison?.activeRunID == run.id, run.control?.testID == comparison?.id,
           comparison?.blocks.contains(where: { $0.id == run.id }) == false {
            comparison?.blocks.append(run); comparison?.activeRunID = nil
        }
        camera.onReading = { [weak self] reading in self?.ingestPose(reading) }
    }

    var breakRemaining: Double { max(0, controlBreakDeadline - clock()) }
    var isPaced: Bool { run?.control != nil && controlledActive }
    var activeWordIndex: Int? {
        guard phase == .recording, let start = cueStart, run?.control != nil else { return nil }
        return currentPacing?.wordIndex(elapsed: ProcessInfo.processInfo.systemUptime - start,
            words: Lexicon.words(in: currentText).count)
    }
    var currentPacing: SignalPacing? {
        guard let control=run?.control else { return nil }
        if control.motionPolicy == .cameraTeacher, let current { return .teacher(prompt:current,block:control.block) }
        return control.pacing
    }
    var mouthReady: Bool {
        guard comparison?.target?.estimator == "apple-vision-face-landmarks-v1" else { return true }
        return camera.mouthReady(now: clock())
    }
    var allowsHeadMovement: Bool { comparison?.allowsMovement == true }
    var usesCameraTeacher: Bool { comparison?.motionPolicy == .cameraTeacher }
    var poseReady: Bool {
        guard let target = comparison?.target else { return false }
        if allowsHeadMovement {
            guard let latest = camera.latest, latest.cameraID == target.cameraID,
                  latest.estimator == target.estimator,
                  usesCameraTeacher ? latest.mouth != nil : latest.pose?.isFinite == true else { return false }
            return clock() - latest.t >= 0 && clock() - latest.t <= 0.3
        }
        return target.isStable(camera.recent, now: clock())
    }
    var motionFresh: Bool {
        latestMotion.map { clock() - $0.t >= 0 && clock() - $0.t < 0.3 } == true
    }
    var canSavePose: Bool {
        phase == .controlSetup && comparison?.blocks.isEmpty == true &&
            CameraPoseTarget.capture(camera.recent, now: clock()) != nil &&
            motionFresh
    }
    var canStartControlledBlock: Bool {
        controlledActive && referenceLink == comparison?.reference && poseReady && mouthReady && motionFresh &&
            (phase == .controlSetup || (phase == .controlBreak && breakRemaining == 0))
    }
    var imuTiltDifference: Double? {
        guard let baseline = comparison?.gravityTarget, let sample = latestMotion else { return nil }
        return Self.gravityAngle(baseline, [sample.gx, sample.gy, sample.gz])
    }
    static func gravityAngle(_ a: [Double], _ b: [Double]) -> Double? {
        guard a.count == 3, b.count == 3, (a + b).allSatisfy(\.isFinite) else { return nil }
        let na = sqrt(a.reduce(0) { $0 + $1 * $1 }), nb = sqrt(b.reduce(0) { $0 + $1 * $1 })
        guard na > 0.1, nb > 0.1 else { return nil }
        return acos(max(-1, min(1, zip(a,b).reduce(0) { $0 + $1.0 * $1.1 } / (na * nb)))) * 180 / .pi
    }
    func prepareControlled(policy: CameraMotionPolicy = .cameraTeacher) {
        guard !isBusy, storageReady, referenceError == nil, let referenceLink else { return }
        comparison = ControlledSignalComparison(reference: referenceLink, seed: UInt64.random(in: .min ... .max))
        comparison?.motionPolicy = policy
        teacherHistory = policy == .cameraTeacher ? completedTeacherHistory() : []
        teacherHistoryCount = teacherHistory.count
        if policy == .cameraTeacher { comparison?.teacherHistoryRunIDs = teacherHistory.map(\.id) }
        controlledActive = true; error = nil; phase = .controlSetup
        camera.onReading = { [weak self] reading in self?.ingestPose(reading) }
        _ = saveComparison()
    }
    private func completedTeacherHistory() -> [SignalCheckRun] {
        let archive=comparisonURL.deletingLastPathComponent().appending(path:"ControlledSignalTests")
        guard let files=try? FileManager.default.contentsOfDirectory(at:archive,includingPropertiesForKeys:nil) else { return [] }
        let completed=files.filter { $0.pathExtension=="json" }.compactMap { file -> ControlledSignalComparison? in
            guard let data=try? Data(contentsOf:file),let value=try? JSONDecoder().decode(ControlledSignalComparison.self,from:data),
                  value.endedAt != nil,value.reference==referenceLink,value.blocks.count==2,
                  value.blocks.allSatisfy({$0.takes.filter({!$0.discarded}).count==18}) else {return nil}
            return value
        }.sorted { ($0.endedAt ?? .distantPast)>($1.endedAt ?? .distantPast) }
        return completed.first?.blocks ?? []
    }
    func savePoseTarget() {
        guard canSavePose, let target = CameraPoseTarget.capture(camera.recent, now: clock()),
              let sample = latestMotion else { return }
        comparison?.target = target
        comparison?.gravityTarget = [sample.gx, sample.gy, sample.gz]
        _ = saveComparison()
    }
    func ingestPose(_ reading: CameraPoseReading) {
        if controlledActive, phase == .controlSetup, allowsHeadMovement,
           comparison?.target == nil, let pose = reading.pose, pose.isFinite,
           reading.mouth != nil, clock() - reading.t >= 0, clock() - reading.t <= 0.3 {
            comparison?.target = .init(id: UUID(), cameraID: reading.cameraID, pose: pose,
                angleTolerance: 5, positionTolerance: 0.05, sizeTolerance: 0.12, estimator: reading.estimator)
            if let sample = latestMotion { comparison?.gravityTarget = [sample.gx, sample.gy, sample.gz] }
            _ = saveComparison()
        }
        if let id = run?.session?.id, recorder?.activeSessionID == id {
            _ = recorder?.recordCamera(reading, sessionID: id)
        }
        if controlledActive {
            let tail = run?.control?.allowsMovement == true && run?.takes.last.map { reading.t <= $0.recording.cueEnd + 0.25 } == true
            if phase == .relax || phase == .recording || (tail && recorder?.isRecording == true) {
                run?.control?.readings.append(reading)
            }
        }
    }
    func exitControlled() {
        guard !isBusy else { return }
        controlledActive = false; camera.stop()
    }

    /// Freeze before asking the user to reseat. An existing reference is reused unchanged.
    func prepareReseat() {
        guard canPrepareReseat else { return }
        error = nil
        do {
            if reference == nil {
                guard let run else { throw FrozenSignalReference.Failure.incomplete }
                let frozen = try FrozenSignalReference.freeze(run)
                try frozen.saveNew(to: referenceURL)
                reference = frozen; referenceLink = try frozen.link()
            }
            phase = .reseat
        } catch { self.error = error.localizedDescription }
    }

    func ingest(_ sample: MotionSample) {
        latestMotion = sample
        if phase == .relax || phase == .recording { buffer.append(sample) }
    }

    func start(audio: AudioService, recorder: SessionRecorder,
               againstReference: Bool = false, controlled: Bool = false, canRecord: @escaping () -> Bool) {
        let controlledReady = controlled && canStartControlledBlock
        let failurePhase = phase
        let reseatReady = againstReference && phase == .reseat && reference != nil && referenceLink != nil
        guard (!isBusy || reseatReady || controlledReady), (!againstReference || reseatReady || controlledReady), (!controlled || controlledReady), storageReady, !recorder.isRecording, canRecord() else { return }
        self.recorder = recorder; self.canRecord = canRecord
        let fixedLink = againstReference || controlled ? referenceLink : nil
        let noReseatConfirmed = Date()
        let block: ControlledSignalBlock = comparison?.blocks.isEmpty == true ? .unchanged : .reseated
        let control: ControlledBlockCapture? = controlled ? comparison?.target.map {
            ControlledBlockCapture(testID: comparison!.id, block: block, target: $0,
                pacing: comparison!.pacing, blockConfirmedAt: noReseatConfirmed, motionPolicy: comparison!.motionPolicy)
        } : nil
        let reseatedAt = controlled ? (block == .reseated ? Date() : nil) : (againstReference ? Date() : nil)
        phase = .preparing; error = nil
        task = Task {
            guard !Task.isCancelled, canRecord() else {
                error = "Signal check cancelled before recording started."; phase = controlled ? failurePhase : .setup; return
            }
            // Signal checks are geometry/motion only, even if a prior microphone check was on.
            audio.stop()
            recorder.condition = .signalCheck
            recorder.prompt = controlled && usesCameraTeacher ? "Camera-teacher block \(block.rawValue); fresh-fit confirmation; variable pacing; natural head movement; \(teacherHistoryCount) historical training sessions; frozen source reference \(fixedLink!.id)." : controlled ? "Controlled block \(block.rawValue); fixed visual pacing; locked camera pose target; frozen reference \(fixedLink!.id)." : againstReference
                ? "Second-session reseat test; frozen reference \(fixedLink!.id); manual START/END; no model adaptation."
                : "Three short sentences, three repetitions, spoken and mouthed; manual START/END; excluded from model training."
            guard let session = recorder.start(audioSink: nil, cameraPoints: true) else {
                error = recorder.error ?? "Could not start the signal check."; phase = controlled ? failurePhase : .setup; return
            }
            begin(session: session, audioDevice: nil, seed: controlled ? comparison!.seed : UInt64.random(in: .min ... .max),
                  reference: fixedLink, reseatedAt: reseatedAt, control: control)
        }
    }

    /// Also usable with a fixture recorder/sink for lifecycle tests, without microphone access.
    func begin(session: RecordingSession, audioDevice: String?, seed: UInt64, recorder: SessionRecorder? = nil,
               reference link: SignalReferenceLink? = nil, reseatedAt: Date? = nil, control: ControlledBlockCapture? = nil) {
        let controlledEntry = controlledActive && control?.testID == comparison?.id && canStartControlledBlock
        guard !isAcquiring || controlledEntry else { return }
        if let recorder { self.recorder = recorder }
        if let link {
            do {
                guard let reference, referenceLink == link else { throw FrozenSignalReference.Failure.missingReference }
                try reference.saveNew(to: session.directory.appending(path: "frozen-signal-reference.json"))
            } catch {
                self.error = error.localizedDescription
                _ = self.recorder?.stop(sessionID: session.id)
                phase = .finished; return
            }
        }
        run = SignalCheckRun(seed: seed, audioDevice: audioDevice, session: session, reference: link, reseatedAt: reseatedAt, control: control, recordsAudio: false)
        if control != nil { comparison?.activeRunID = run?.id; _ = saveComparison() }
        plan = SignalCheckRun.plan(seed: seed); index = 0
        if save() { phase = .ready }
        else { closeRecording(); phase = .finished }
    }

    func startTake() {
        guard phase == .ready, current != nil else { return }
        if isPaced, !motionFresh { error = "Waiting for fresh AirPods motion. Reconnect before starting a take."; return }
        if isPaced, !mouthReady { error = "Keep the lip markers visible before starting a take."; return }
        if isPaced, !poseReady { error = allowsHeadMovement ? "Keep your face visible so the camera can track your head." : "Return to the saved camera pose and hold still for one second before starting."; return }
        error = nil
        guard canRecord() else { error = "Motion stopped. Reconnect your AirPods and start a new check."; stop(); return }
        buffer = []; cueStart = nil; phase = .relax; secondsRemaining = 3
        task = Task {
            do {
                let end = ProcessInfo.processInfo.systemUptime + 3
                while ProcessInfo.processInfo.systemUptime < end {
                    try Task.checkCancellation()
                    secondsRemaining = max(0, end - ProcessInfo.processInfo.systemUptime)
                    try await Task.sleep(for: .milliseconds(50))
                }
                guard canRecord() else { error = "Motion stopped before START."; stop(); return }
                if isPaced, !motionFresh { error = "AirPods motion stopped before START."; stop(); return }
                if isPaced, !mouthReady { error = "Lip tracking was lost before START. Keep your mouth visible and start again."; buffer = []; phase = .ready; return }
                if isPaced, !poseReady { error = allowsHeadMovement ? "Head tracking was lost before START. Keep your face visible and start again." : "Pose moved during the still baseline. Return to the target and start again."; buffer = []; phase = .ready; return }
                beginCapture(at: ProcessInfo.processInfo.systemUptime)
                let duration = currentPacing?.duration(words: Lexicon.words(in: currentText).count) ?? 12
                let deadline = cueStart! + duration
                while phase == .recording && ProcessInfo.processInfo.systemUptime < deadline {
                    try Task.checkCancellation()
                    if !canRecord() || (isPaced && !motionFresh) { error = "Motion stopped during the take. This take was not completed."; stop(); return }
                    secondsRemaining = max(0, deadline - ProcessInfo.processInfo.systemUptime)
                    try await Task.sleep(for: .milliseconds(50))
                }
                if phase == .recording { endCapture(at: ProcessInfo.processInfo.systemUptime, reason: isPaced ? .scheduled : .timeout) }
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription; stop() }
        }
    }

    func beginCapture(at time: Double) {
        guard (phase == .relax || phase == .ready), time.isFinite, let current else { return }
        if isPaced, !poseReady || !mouthReady { return }
        cueStart = time; phase = .recording; secondsRemaining = 12
        recorder?.logMarker(uptime: time, kind: "signal-start", label: label(current))
        if let pacing = currentPacing {
            for (i, word) in currentText.split(separator: " ").enumerated() {
                recorder?.logMarker(uptime: time + pacing.leadIn + Double(i) * pacing.wordSeconds,
                    kind: "signal-pace-target", label: String(word))
            }
        }
    }

    func finishTake() {
        guard !isPaced else { return }
        endCapture(at: ProcessInfo.processInfo.systemUptime, reason: .done)
    }

    func endCapture(at time: Double, reason: SignalCheckTake.EndReason) {
        guard phase == .recording, let start = cueStart, let current, time.isFinite, time > start else { return }
        task?.cancel()
        let recording = SentenceRecording(condition: current.condition, samples: buffer.filter { $0.t <= time }, cueStart: start, cueEnd: time)
        let pacing = currentPacing
        run?.takes.append(.init(prompt: current, recording: recording, endReason: reason,pacing:pacing))
        recorder?.logMarker(uptime: time, kind: reason == .timeout ? "signal-timeout" : "signal-end", label: label(current))
        checkpoint()
        if let recordingError = recorder?.error { error = recordingError }
        cueStart = nil; buffer = []; secondsRemaining = 0
        phase = .ended
        if !save() { closeRecording(); phase = .finished }
    }

    func redo() {
        guard canRedo, let current else { return }
        task?.cancel()
        if phase == .ended, let i = run?.takes.indices.last { run?.takes[i].discarded = true }
        recorder?.logMarker(uptime: ProcessInfo.processInfo.systemUptime, kind: "signal-redo", label: label(current))
        cueStart = nil; buffer = []; phase = .ready
        if !save() { closeRecording(); phase = .finished }
    }

    func next() {
        if phase == .ready { startTake() }
        else if phase == .recording { finishTake() }
        else if phase == .ended {
            if index + 1 < plan.count { index += 1; phase = .ready }
            else { stop() }
        }
    }

    func stop() {
        guard isAcquiring || phase == .preparing else { return }
        task?.cancel()
        if phase == .controlSetup || phase == .controlBreak {
            comparison?.endedAt = Date(); _ = saveComparison(); controlledActive = false; camera.stop(); phase = .finished; return
        }
        if phase == .reseat { phase = .finished; return }
        if phase == .preparing { return }
        let captureTail = run?.control?.allowsMovement == true && phase == .ended && recorder?.isRecording == true
        if !captureTail {
            closeRecording()
            guard run != nil, save() else { phase = .finished; return }
        }
        phase = .processing
        if captureTail { processingText = "Finishing camera capture…" }
        task = Task {
            if captureTail {
                // Vision arrives after its capture timestamp. Retain camera-only tail context
                // for the centred derivative without changing the take's END or duration.
                do { try await Task.sleep(for: .milliseconds(500)); try Task.checkCancellation() }
                catch { return }
                closeRecording()
                guard run != nil, save() else { phase = .finished; return }
            }
            await process(); phase = .finished; processingText = ""; finishControlledBlock()
        }
    }

    func prepareForTermination() {
        guard isBusy else { return }
        camera.stop()
        if phase == .controlSetup || phase == .controlBreak {
            comparison?.endedAt = Date(); _ = saveComparison(); controlledActive = false; phase = .finished; return
        }
        if phase == .reseat { phase = .finished; return }
        task?.cancel()
        closeRecording()
        _ = save(); phase = .finished
        if controlledActive, let run, run.control != nil {
            comparison?.blocks.append(run); comparison?.activeRunID = nil; comparison?.endedAt = Date()
            _ = saveComparison(); controlledActive = false
        }
    }

    private func closeRecording() {
        if phase == .recording, let current {
            recorder?.logMarker(uptime: ProcessInfo.processInfo.systemUptime, kind: "signal-aborted", label: label(current))
        }
        if let id = run?.session?.id, let closed = recorder?.stop(sessionID: id) { run?.session = closed }
        if run?.endedAt == nil { run?.endedAt = Date() }
        if run?.control?.block == .unchanged { controlBreakDeadline = clock() + (comparison?.minimumBreakSeconds ?? 30) }
        buffer = []; cueStart = nil
    }

    private func checkpoint() {
        guard let id = run?.session?.id else { return }
        do {
            if let current = try recorder?.checkpoint(sessionID: id) { run?.session = current }
        } catch { self.error = error.localizedDescription }
    }

    private func process() async {
        guard let snapshot = run else { return }
        do {
            processingText = snapshot.reference == nil ? "Comparing repeated sentences and timing controls…" : "Scoring only against the frozen first session…"
            let frozen = reference
            let analysis = Task.detached(priority: .userInitiated) {
                if snapshot.reference != nil {
                    guard let frozen else { throw FrozenSignalReference.Failure.missingReference }
                    return try frozen.evaluate(snapshot)
                }
                return try SignalCheckAnalysis.run(snapshot)
            }
            let report = try await withTaskCancellationHandler(operation: { try await analysis.value }, onCancel: { analysis.cancel() })
            try Task.checkCancellation()
            run?.report = report
            if snapshot.control?.motionPolicy == .cameraTeacher {
                processingText = snapshot.control?.block == .unchanged ? "Training AirPods features from the camera mouth teacher…" : "Testing frozen AirPods features and independent camera controls…"
                let savedTeacher = comparison?.teacherReference
                let history = teacherHistory
                let teacherTask = Task.detached(priority: .userInitiated) { () -> (CameraTeacherReference?,[TeacherConditionScores]?,String?) in
                    do {
                        if snapshot.control?.block == .unchanged { return (try CameraTeacherReference.freeze(snapshot,history:history),nil,nil) }
                        guard let savedTeacher else { return (nil,nil,"A complete source block is needed before the teacher model can be tested.") }
                        return (nil,try savedTeacher.evaluate(snapshot),nil)
                    } catch { return (nil,nil,error.localizedDescription) }
                }
                let result = await withTaskCancellationHandler(operation:{ await teacherTask.value },onCancel:{teacherTask.cancel()})
                try Task.checkCancellation()
                if let teacher=result.0 {
                    let fingerprint=try teacher.fingerprint()
                    comparison?.teacherReference=teacher;comparison?.teacherFingerprint=fingerprint
                }
                if let scores=result.1 { comparison?.teacherScores=scores }
                if let note=result.2 { comparison?.teacherNote=note }
            } else if snapshot.control?.allowsMovement == true {
                processingText = snapshot.control?.block == .unchanged ? "Learning head-motion correction from relaxed-mouth baselines…" : "Comparing raw and head-corrected rotation against locked block A…"
                let savedReference = comparison?.cameraReference
                let cameraAnalysis = Task.detached(priority: .userInitiated) { () -> (CameraMotionReference?, [CameraMotionScore]?, String?) in
                    do {
                        if snapshot.control?.block == .unchanged {
                            let reference = try CameraMotionReference.freeze(snapshot)
                            return (reference, nil, reference.correctionNote)
                        }
                        guard let savedReference else { return (nil, nil, "No complete block-A camera reference is available.") }
                        return (nil, try savedReference.evaluate(snapshot), nil)
                    } catch { return (nil, nil, error.localizedDescription) }
                }
                let result = await withTaskCancellationHandler(operation: { await cameraAnalysis.value }, onCancel: { cameraAnalysis.cancel() })
                try Task.checkCancellation()
                if let reference = result.0 { comparison?.cameraReference = reference }
                if let scores = result.1 { comparison?.cameraScores = scores }
                if let note = result.2 { comparison?.cameraAdjustmentNote = note }
            }
            _ = save()
        } catch is CancellationError { }
        catch { self.error = error.localizedDescription }
    }

    private func finishControlledBlock() {
        guard controlledActive, let run, run.control != nil else { return }
        comparison?.blocks.append(run); comparison?.activeRunID = nil
        let complete = run.takes.filter { !$0.discarded }.count == 18 && run.report != nil
        if run.control?.block == .unchanged && complete {
            phase = .controlBreak
        } else {
            comparison?.endedAt = Date(); camera.stop(); phase = .finished
        }
        _ = saveComparison()
    }

    @discardableResult private func saveComparison() -> Bool {
        guard let comparison else { return true }
        do {
            try comparison.validate()
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            let bytes = try encoder.encode(comparison)
            let archive = comparisonURL.deletingLastPathComponent().appending(path: "ControlledSignalTests")
            try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: true)
            try bytes.write(to: archive.appending(path: "\(comparison.id).json"), options: .atomic)
            try bytes.write(to: comparisonURL, options: .atomic)
            for block in comparison.blocks {
                if let directory = block.session?.directory { try bytes.write(to: directory.appending(path: "controlled-comparison.json"), options: .atomic) }
            }
            return true
        } catch { self.error = error.localizedDescription; storageReady = false; return false }
    }

    private func label(_ prompt: SignalCheckPrompt) -> String {
        "sentence=\(prompt.sentence) repetition=\(prompt.repetition) condition=\(prompt.condition.rawValue) \(SignalCheckRun.sentences[prompt.sentence])"
    }

    @discardableResult private func save() -> Bool {
        guard let run else { return true }
        do {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            let bytes = try encoder.encode(run)
            try FileManager.default.createDirectory(at: storeURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Save a permanent per-visit copy as well as the most recent check for the UI.
            if let directory = run.session?.directory {
                try bytes.write(to: directory.appending(path: "signal-check.json"), options: .atomic)
            }
            try bytes.write(to: storeURL, options: .atomic)
            return true
        } catch { self.error = error.localizedDescription; storageReady = false; return false }
    }
}
