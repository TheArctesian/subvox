import Foundation

struct SignalPacing: Codable, Equatable, Sendable {
    let leadIn: Double
    let wordSeconds: Double
    let tail: Double
    static let controlled = SignalPacing(leadIn: 0.4, wordSeconds: 0.65, tail: 0.4)
    static func teacher(prompt:SignalCheckPrompt,block:ControlledSignalBlock)->Self {
        let i=(prompt.repetition+prompt.sentence+(block == .reseated ? 1 : 0))%3
        return .init(leadIn:[0.25,0.55,0.85][(i+prompt.repetition)%3],wordSeconds:[0.5,0.65,0.8][i],tail:0.4)
    }
    func duration(words: Int) -> Double { leadIn + Double(words) * wordSeconds + tail }
    func wordIndex(elapsed: Double, words: Int) -> Int? {
        guard elapsed >= leadIn, elapsed < leadIn + Double(words) * wordSeconds else { return nil }
        return min(words - 1, Int((elapsed - leadIn + 1e-9) / wordSeconds))
    }
}

enum ControlledSignalBlock: String, Codable, Sendable {
    case unchanged, reseated
    var label: String { self == .unchanged ? "A · keep AirPods seated" : "B · after reseating" }
}
enum CameraMotionPolicy: String, Codable, Sendable {
    case strictPose, naturalMovement, cameraTeacher
}
struct ControlledBlockCapture: Codable, Sendable {
    let testID: UUID
    let block: ControlledSignalBlock
    let target: CameraPoseTarget
    let pacing: SignalPacing
    let blockConfirmedAt: Date
    var readings: [CameraPoseReading] = []
    var motionPolicy: CameraMotionPolicy?
    var allowsMovement: Bool { motionPolicy == .naturalMovement || motionPolicy == .cameraTeacher }
}
struct ControlledPoseQuality: Codable, Sendable {
    var takeID: UUID
    var coverage: Double
    var withinTarget: Double
    var timingError: Double
    var passes: Bool
}
struct ControlledSignalComparison: Codable, Identifiable, Sendable {
    var id = UUID()
    var startedAt = Date()
    var endedAt: Date?
    let reference: SignalReferenceLink
    let seed: UInt64
    var target: CameraPoseTarget?
    var gravityTarget: [Double]?
    var activeRunID: UUID?
    var pacing = SignalPacing.controlled
    var minimumBreakSeconds = 30.0
    var blocks: [SignalCheckRun] = []
    // Missing means the original strict-pose protocol, preserving historical scores.
    var motionPolicy: CameraMotionPolicy?
    var cameraReference: CameraMotionReference?
    var cameraScores: [CameraMotionScore]?
    var cameraAdjustmentNote: String?
    var teacherReference: CameraTeacherReference?
    var teacherScores: [TeacherConditionScores]?
    var teacherNote: String?
    var teacherHistoryRunIDs: [UUID]?
    var teacherFingerprint: String?
    var allowsMovement: Bool { motionPolicy == .naturalMovement || motionPolicy == .cameraTeacher }

    static func qualities(_ run: SignalCheckRun) -> [ControlledPoseQuality] {
        guard let control = run.control else { return [] }
        return run.takes.filter { !$0.discarded }.map { take in
            let start = take.recording.cueStart, end = take.recording.cueEnd, duration = end - start
            guard duration > 0 else { return .init(takeID: take.id, coverage: 0, withinTarget: 0, timingError: 0, passes: false) }
            let readings = control.readings.filter { $0.t >= start - 0.4 && $0.t <= end }.sorted { $0.t < $1.t }
            var covered = 0.0, matched = 0.0
            for (i, reading) in readings.enumerated() {
                // At most 0.25 s of support per processed frame. Gaps stay missing.
                let next = i + 1 < readings.count ? readings[i + 1].t : end
                let span = max(0, min(end, next, reading.t + 0.25) - max(start, reading.t))
                if reading.cameraID == control.target.cameraID, reading.pose?.isFinite == true,
                   !control.allowsMovement || reading.estimator == control.target.estimator { covered += span }
                if control.target.accepts(reading) { matched += span }
            }
            let prescribed = control.motionPolicy == .cameraTeacher ? SignalPacing.teacher(prompt:take.prompt,block:control.block) : control.pacing
            let expected = prescribed.duration(words: Lexicon.words(in: SignalCheckRun.sentences[take.prompt.sentence]).count)
            let pacingValid = control.motionPolicy != .cameraTeacher || take.pacing == prescribed
            let timing = abs(duration - expected)
            let lips = CameraMouthStatistics.inspect(control.readings, start: start, end: end)
            let trackingPasses = covered / duration >= 0.9 &&
                (control.allowsMovement ? lips.coverage >= 0.9 : matched / duration >= 0.9)
            return .init(takeID: take.id, coverage: covered / duration, withinTarget: matched / duration,
                         timingError: timing, passes: trackingPasses &&
                            timing <= 0.1 && take.endReason == .scheduled && pacingValid)
        }
    }

    func validate() throws {
        guard pacing == .controlled, minimumBreakSeconds == 30, blocks.count <= 2 else { throw FrozenSignalReference.Failure.contaminated }
        if let target {
            guard target.pose.isFinite, !target.cameraID.isEmpty, target.angleTolerance == 5,
                  target.positionTolerance == 0.05, target.sizeTolerance == 0.12 else { throw FrozenSignalReference.Failure.contaminated }
        }
        var ids = Set<UUID>(), sessions = Set<UUID>()
        for (i, run) in blocks.enumerated() {
            guard run.reference == reference, let control = run.control, control.testID == id,
                  control.target.id == target?.id, control.target == target,
                  (control.motionPolicy ?? .strictPose) == (motionPolicy ?? .strictPose),
                  control.block == (i == 0 ? .unchanged : .reseated), control.pacing == pacing,
                  run.seed == seed, ids.insert(run.id).inserted,
                  let session = run.session, sessions.insert(session.id).inserted else { throw FrozenSignalReference.Failure.contaminated }
        }
        if let cameraReference {
            guard allowsMovement, let first = blocks.first, cameraReference.sourceRunID == first.id,
                  cameraReference.sourceSessionID == first.session?.id,
                  cameraReference.testID == id, cameraReference.targetID == target?.id,cameraReference.seed == seed,
                  cameraReference.cameraID == target?.cameraID else { throw FrozenSignalReference.Failure.contaminated }
        }
        if let teacherReference {
            try teacherReference.validate()
            if let teacherFingerprint {guard try teacherReference.fingerprint()==teacherFingerprint else {throw FrozenSignalReference.Failure.fingerprint}}
            guard motionPolicy == .cameraTeacher,let first=blocks.first,teacherReference.sourceRunID==first.id,
                  teacherReference.sourceSessionID==first.session?.id,teacherReference.testID==id,
                  teacherReference.targetID==target?.id,teacherReference.cameraID==target?.cameraID,
                  teacherReference.seed==seed else { throw FrozenSignalReference.Failure.contaminated }
        }
    }
}
