import CoreGraphics
import Foundation

struct CameraLandmarkPoint: Codable, Hashable, Sendable {
    var x: Double
    var y: Double
    var valid: Bool { x.isFinite && y.isFinite && (0...1).contains(x) && (0...1).contains(y) }
}

/// Raw points use full-image normalized coordinates, lower-left origin. Shape measurements
/// use a roll-aligned eye frame and inter-eye-distance units; yaw/pitch remain uncorrected.
/// Eye/corner left and right refer to the unmirrored image, not anatomical left/right.
struct CameraMouthObservation: Codable, Sendable {
    var outerLips: [CameraLandmarkPoint]
    var innerLips: [CameraLandmarkPoint]
    var leftEye: CameraLandmarkPoint
    var rightEye: CameraLandmarkPoint
    var leftCorner: CameraLandmarkPoint
    var rightCorner: CameraLandmarkPoint
    var openingTop: CameraLandmarkPoint
    var openingBottom: CameraLandmarkPoint
    var centre: CameraLandmarkPoint
    var width: Double
    var opening: Double
    var innerArea: Double
    var outerAspect: Double
    var centreX: Double
    var centreY: Double
    var confidence: Double
    var openingSpeed: Double?
    var movementSpeed: Double?

    static func measure(outer: [CameraLandmarkPoint], inner: [CameraLandmarkPoint], leftEye: CameraLandmarkPoint,
                        rightEye: CameraLandmarkPoint, imageWidth: Int, imageHeight: Int, confidence: Double) -> Self? {
        guard outer.count >= 3, inner.count >= 3, outer.count <= 100, inner.count <= 100,
              (outer + inner + [leftEye, rightEye]).allSatisfy(\.valid),
              imageWidth > 0, imageHeight > 0, confidence.isFinite, (0.5...1).contains(confidence) else { return nil }
        // Pixel coordinates avoid aspect-ratio distortion when measuring eye distance/angles.
        let imageLeft = leftEye.x <= rightEye.x ? leftEye : rightEye
        let imageRight = leftEye.x <= rightEye.x ? rightEye : leftEye
        let dx = (imageRight.x - imageLeft.x) * Double(imageWidth)
        let dy = (imageRight.y - imageLeft.y) * Double(imageHeight)
        let distance = hypot(dx, dy)
        guard distance >= 10 else { return nil }
        let ux = dx / distance, uy = dy / distance
        let eyeMid = CameraLandmarkPoint(x: (leftEye.x + rightEye.x)/2, y: (leftEye.y + rightEye.y)/2)
        func aligned(_ p: CameraLandmarkPoint) -> (x: Double, y: Double) {
            let x = (p.x - eyeMid.x) * Double(imageWidth), y = (p.y - eyeMid.y) * Double(imageHeight)
            return ((x * ux + y * uy) / distance, (-x * uy + y * ux) / distance)
        }
        let o = outer.map(aligned), i = inner.map(aligned)
        guard let left = o.indices.min(by: { o[$0].x < o[$1].x }),
              let right = o.indices.max(by: { o[$0].x < o[$1].x }),
              let top = i.indices.max(by: { i[$0].y < i[$1].y }),
              let bottom = i.indices.min(by: { i[$0].y < i[$1].y }) else { return nil }
        let width = o[right].x - o[left].x
        guard width > 1e-6 else { return nil }
        let centre = CameraLandmarkPoint(x: outer.map(\.x).reduce(0,+)/Double(outer.count),
                                         y: outer.map(\.y).reduce(0,+)/Double(outer.count))
        let c = aligned(centre)
        let outerHeight = (o.map(\.y).max() ?? 0) - (o.map(\.y).min() ?? 0)
        var twiceArea = 0.0
        for index in i.indices { let next = i[(index + 1) % i.count]; twiceArea += i[index].x * next.y - next.x * i[index].y }
        return Self(outerLips: outer, innerLips: inner, leftEye: imageLeft, rightEye: imageRight,
            leftCorner: outer[left], rightCorner: outer[right], openingTop: inner[top], openingBottom: inner[bottom],
            centre: centre, width: width, opening: max(0, i[top].y - i[bottom].y), innerArea: abs(twiceArea)/2,
            outerAspect: outerHeight/width, centreX: c.x, centreY: c.y, confidence: confidence)
    }

    func withVelocity(from previous: CameraMouthObservation, seconds: Double) -> Self {
        var result = self
        result.openingSpeed = nil; result.movementSpeed = nil
        guard seconds.isFinite, seconds >= 0.01, seconds <= 0.2 else { return result }
        result.openingSpeed = (opening - previous.opening) / seconds
        result.movementSpeed = sqrt(pow(width - previous.width, 2) + pow(opening - previous.opening, 2) +
            pow(centreX - previous.centreX, 2) + pow(centreY - previous.centreY, 2)) / seconds
        return result
    }
}

enum CameraOverlayGeometry {
    static func aspectFit(imageWidth: Int, imageHeight: Int, width: Double, height: Double) -> CGRect {
        guard imageWidth > 0, imageHeight > 0, width > 0, height > 0 else { return .zero }
        let scale = min(width / Double(imageWidth), height / Double(imageHeight))
        let w = Double(imageWidth) * scale, h = Double(imageHeight) * scale
        return CGRect(x: (width-w)/2, y: (height-h)/2, width: w, height: h)
    }
    static func point(_ p: CameraLandmarkPoint, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + p.x * rect.width, y: rect.maxY - p.y * rect.height)
    }
}

struct CameraMouthStatistics: Sendable {
    var samples: Int
    var rate: Double
    var coverage: Double
    var meanConfidence: Double?
    var receiptTimeFallbacks: Int
    static func inspect(_ readings: [CameraPoseReading], start: Double, end: Double) -> Self {
        guard start.isFinite, end.isFinite, end > start else { return .init(samples: 0, rate: 0, coverage: 0, receiptTimeFallbacks: 0) }
        let values = readings.filter { $0.t >= start - 0.1 && $0.t <= end }.sorted { $0.t < $1.t }
        let frames = values.filter { $0.t >= start }
        var supported = 0.0
        for (i, frame) in values.enumerated() where frame.mouth != nil {
            let next = i + 1 < values.count ? values[i + 1].t : end
            supported += max(0, min(end, next, frame.t + 0.1) - max(start, frame.t))
        }
        let confidence = frames.compactMap { $0.mouth?.confidence }
        let rate = frames.count >= 2 && frames.last!.t > frames.first!.t ? Double(frames.count-1)/(frames.last!.t-frames.first!.t) : 0
        return .init(samples: frames.count, rate: rate, coverage: min(1, supported/(end-start)),
            meanConfidence: confidence.isEmpty ? nil : confidence.reduce(0,+)/Double(confidence.count),
            receiptTimeFallbacks: frames.filter { $0.usedReceiptTime == true }.count)
    }
}
