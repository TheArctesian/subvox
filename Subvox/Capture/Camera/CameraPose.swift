import Foundation

struct CameraHeadPose: Codable, Hashable, Sendable {
    var yaw: Double
    var pitch: Double
    var roll: Double
    var x: Double
    var y: Double
    var size: Double
    var isFinite: Bool { [yaw, pitch, roll, x, y, size].allSatisfy(\.isFinite) && size > 0 }
}
struct CameraPoseReading: Codable, Sendable {
    var t: Double
    var cameraID: String
    var pose: CameraHeadPose?
    var usedReceiptTime: Bool?
    var mouth: CameraMouthObservation?
    var imageWidth: Int?
    var imageHeight: Int?
    var receivedAt: Double?
    var processedAt: Double?
    var estimator: String?
}
struct CameraPoseTarget: Codable, Identifiable, Sendable {
    let id: UUID
    let cameraID: String
    let pose: CameraHeadPose
    // Fixed before block A. Camera estimates guide pose; they do not establish exact head position.
    let angleTolerance: Double
    let positionTolerance: Double
    let sizeTolerance: Double
    var estimator: String?

    func accepts(_ reading: CameraPoseReading) -> Bool {
        guard cameraID == reading.cameraID, estimator == reading.estimator, let value = reading.pose, value.isFinite else { return false }
        func angle(_ a: Double, _ b: Double) -> Double {
            abs(atan2(sin((a - b) * .pi / 180), cos((a - b) * .pi / 180))) * 180 / .pi
        }
        return angle(value.yaw, pose.yaw) <= angleTolerance && angle(value.pitch, pose.pitch) <= angleTolerance &&
            angle(value.roll, pose.roll) <= angleTolerance && hypot(value.x - pose.x, value.y - pose.y) <= positionTolerance &&
            abs(value.size / pose.size - 1) <= sizeTolerance
    }

    static func capture(_ readings: [CameraPoseReading], now: Double) -> CameraPoseTarget? {
        let window = readings.filter { $0.t >= now - 1.6 && $0.t <= now }
        guard window.count >= 9, let first = window.first, let last = window.last, last.t - first.t >= 1.2,
              now - last.t <= 0.4, window.allSatisfy({ $0.cameraID == first.cameraID && $0.pose?.isFinite == true }),
              zip(window, window.dropFirst()).allSatisfy({ $1.t > $0.t && $1.t - $0.t <= 0.4 }) else { return nil }
        let values = window.compactMap(\.pose)
        func mean(_ key: KeyPath<CameraHeadPose, Double>) -> Double { values.map { $0[keyPath: key] }.reduce(0, +) / Double(values.count) }
        let pose = CameraHeadPose(yaw: mean(\.yaw), pitch: mean(\.pitch), roll: mean(\.roll), x: mean(\.x), y: mean(\.y), size: mean(\.size))
        let stable = CameraPoseTarget(id: UUID(), cameraID: first.cameraID, pose: pose,
            angleTolerance: 3, positionTolerance: 0.025, sizeTolerance: 0.08, estimator: first.estimator)
        guard window.allSatisfy(stable.accepts) else { return nil }
        return CameraPoseTarget(id: UUID(), cameraID: first.cameraID, pose: pose,
            angleTolerance: 5, positionTolerance: 0.05, sizeTolerance: 0.12, estimator: first.estimator)
    }

    func isStable(_ readings: [CameraPoseReading], now: Double) -> Bool {
        let window = readings.filter { $0.t >= now - 1 && $0.t <= now }
        return window.count >= 5 && (window.last.map { now - $0.t <= 0.4 } ?? false) &&
            (window.first.map { now - $0.t >= 0.75 } ?? false) && window.allSatisfy(accepts) &&
            zip(window, window.dropFirst()).allSatisfy { $1.t > $0.t && $1.t - $0.t <= 0.4 }
    }
}

extension CameraPoseTarget: Equatable {}
