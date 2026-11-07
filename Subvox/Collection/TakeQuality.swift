import Foundation

struct TakeQuality: Sendable {
    var issues: [String]
    var invalid: Bool
    var sampleRate: Double?

    static func inspect(_ samples: [MotionSample], maximumDuration: TimeInterval = 6) -> TakeQuality {
        guard samples.count >= 2 else {
            return TakeQuality(issues: ["Too few samples"], invalid: true)
        }
        let finite = samples.allSatisfy { s in
            [s.t, s.ax, s.ay, s.az, s.rx, s.ry, s.rz, s.pitch, s.roll, s.yaw].allSatisfy(\.isFinite)
        }
        guard finite else { return TakeQuality(issues: ["Invalid sensor values"], invalid: true) }
        let intervals = zip(samples, samples.dropFirst()).map { $1.t - $0.t }
        guard intervals.allSatisfy({ $0 > 0 }) else {
            return TakeQuality(issues: ["Sensor timestamps are not increasing"], invalid: true)
        }
        let sorted = intervals.sorted()
        let step = sorted[sorted.count / 2]
        let duration = samples.last!.t - samples.first!.t
        var issues: [String] = []
        if samples.count < 12 || duration < 0.35 { issues.append("Short recording") }
        if duration > maximumDuration { issues.append("Long recording") }
        if intervals.contains(where: { $0 > max(0.12, step * 3) }) { issues.append("Gap in sensor samples") }
        if samples.map(MotionFeatures.energy).max()! < 0.005 { issues.append("Very little motion") }
        if samples.contains(where: {
            sqrt($0.rx * $0.rx + $0.ry * $0.ry + $0.rz * $0.rz) > 2 ||
            sqrt($0.ax * $0.ax + $0.ay * $0.ay + $0.az * $0.az) > 0.3
        }) { issues.append("Large motion — check for head movement") }
        return TakeQuality(issues: issues, invalid: false, sampleRate: 1 / step)
    }
}
