import AVFoundation
import CoreMedia
import Foundation
import Observation
import Vision

/// Local geometry only. Pixel buffers are used by Vision and the preview, never written to disk.
@MainActor @Observable
final class CameraPoseService {
    private(set) var isRunning = false
    private(set) var isStarting = false
    private(set) var error: String?
    private(set) var recent: [CameraPoseReading] = []
    var latest: CameraPoseReading? { recent.last }
    var sampleRate: Double {
        guard let first = recent.first, let last = recent.last, last.t > first.t else { return 0 }
        return Double(recent.count - 1) / (last.t - first.t)
    }
    func mouthReady(now: Double) -> Bool {
        let window = recent.filter { $0.t >= now - 0.5 && $0.t <= now }
        return window.count >= 3 && (window.last.map { now - $0.t <= 0.2 } ?? false) && window.suffix(3).allSatisfy { $0.mouth != nil }
    }
    @ObservationIgnored var onReading: ((CameraPoseReading) -> Void)?
    @ObservationIgnored private let capture = CameraPoseCapture()
    @ObservationIgnored private var generation = UUID()
    var session: AVCaptureSession { capture.session }

    func start(cameraID: String? = nil) async {
        guard !isRunning, !isStarting else { return }
        isStarting = true; error = nil; recent = []
        let token = UUID(); generation = token
        defer { if generation == token { isStarting = false } }
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            if generation == token { error = "Camera access is needed for the independent pose guide. Enable it in System Settings › Privacy & Security › Camera." }
            return
        }
        guard generation == token, !Task.isCancelled else { return }
        do {
            try await capture.start(cameraID: cameraID) { [weak self] reading in
                Task { @MainActor in
                    guard let self, self.generation == token, self.isRunning else { return }
                    self.ingest(reading)
                }
            }
            guard generation == token else { return }
            guard !Task.isCancelled else { stop(); return }
            isRunning = true
        } catch { if generation == token { self.error = error.localizedDescription } }
    }
    func stop() {
        generation = UUID(); capture.stop(); isRunning = false; isStarting = false; recent = []
    }
    func ingest(_ reading: CameraPoseReading) {
        guard reading.t.isFinite, recent.last.map({ reading.t > $0.t }) ?? true else { return }
        recent.append(reading); recent.removeAll { $0.t < reading.t - 2 }
        onReading?(reading)
    }
}

private final class CameraPoseCapture: @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "subvox.camera-pose", qos: .userInitiated)
    private var processor: CameraPoseProcessor?
    func start(cameraID: String?, onReading: @escaping @Sendable (CameraPoseReading) -> Void) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    let builtIn = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified).devices.first
                    guard let device = builtIn ?? AVCaptureDevice.default(for: .video), cameraID == nil || cameraID == device.uniqueID else {
                        throw NSError(domain: "Subvox.Camera", code: 1, userInfo: [NSLocalizedDescriptionKey: "The same Mac camera must be available for both blocks."])
                    }
                    let input = try AVCaptureDeviceInput(device: device)
                    self.session.beginConfiguration()
                    for input in self.session.inputs { self.session.removeInput(input) }
                    for output in self.session.outputs { self.session.removeOutput(output) }
                    if self.session.canSetSessionPreset(.hd1280x720) { self.session.sessionPreset = .hd1280x720 }
                    let output = AVCaptureVideoDataOutput()
                    output.alwaysDiscardsLateVideoFrames = true
                    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                    guard self.session.canAddInput(input), self.session.canAddOutput(output) else {
                        self.session.commitConfiguration()
                        throw NSError(domain: "Subvox.Camera", code: 2, userInfo: [NSLocalizedDescriptionKey: "The camera could not provide a video stream."])
                    }
                    self.session.addInput(input); self.session.addOutput(output)
                    let processor = CameraPoseProcessor(cameraID: device.uniqueID, onReading: onReading)
                    self.processor = processor; output.setSampleBufferDelegate(processor, queue: self.queue)
                    if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
                        connection.automaticallyAdjustsVideoMirroring = false; connection.isVideoMirrored = false
                    }
                    self.session.commitConfiguration(); self.session.startRunning()
                    guard self.session.isRunning else { throw NSError(domain: "Subvox.Camera", code: 3, userInfo: [NSLocalizedDescriptionKey: "Camera did not start. Close another app using it and try again."]) }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    func stop() { queue.async { self.session.stopRunning() } }
}

private final class CameraPoseProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let cameraID: String
    private let onReading: @Sendable (CameraPoseReading) -> Void
    private var lastTime = 0.0
    private var previous: CameraPoseReading?
    init(cameraID: String, onReading: @escaping @Sendable (CameraPoseReading) -> Void) {
        self.cameraID = cameraID; self.onReading = onReading
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput buffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let received = ProcessInfo.processInfo.systemUptime
        let timestamp = CMSampleBufferGetPresentationTimeStamp(buffer).seconds
        let hostTime = timestamp.isFinite && abs(timestamp - received) <= 1
        let time = hostTime ? timestamp : received
        // Target up to 30 processed frames/s; log actual timestamps/rate and leave dropped frames missing.
        guard time - lastTime >= 1.0/30 - 0.001 else { return }
        lastTime = time
        var reading = CameraPoseReading(t: time, cameraID: cameraID, usedReceiptTime: !hostTime,
            receivedAt: received, estimator: "apple-vision-face-landmarks-v1")
        if let pixel = CMSampleBufferGetImageBuffer(buffer) {
            let width = CVPixelBufferGetWidth(pixel), height = CVPixelBufferGetHeight(pixel)
            reading.imageWidth = width; reading.imageHeight = height
            let handler = VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .up)
            // Landmark observations can omit pitch. Keep the rectangle detector's
            // head angles, then refine that same face for lip landmarks.
            let detection = VNDetectFaceRectanglesRequest()
            if (try? handler.perform([detection])) != nil,
               let faces = detection.results, faces.count == 1, let face = faces.first, face.confidence >= 0.6 {
                let box = face.boundingBox
                if let yaw = face.yaw?.doubleValue, let pitch = face.pitch?.doubleValue, let roll = face.roll?.doubleValue {
                    reading.pose = CameraHeadPose(yaw: yaw * 180 / .pi, pitch: pitch * 180 / .pi, roll: roll * 180 / .pi,
                        x: box.midX, y: box.midY, size: sqrt(box.width * box.height))
                }
                let request = VNDetectFaceLandmarksRequest()
                request.inputFaceObservations = [face]
                if (try? handler.perform([request])) != nil,
                   let landmarkFace = request.results?.first, let landmarks = landmarkFace.landmarks {
                    let box = landmarkFace.boundingBox
                    func points(_ region: VNFaceLandmarkRegion2D?) -> [CameraLandmarkPoint] {
                        guard let region else { return [] }
                        return region.normalizedPoints.map {
                            CameraLandmarkPoint(x: box.minX + $0.x * box.width, y: box.minY + $0.y * box.height)
                        }
                    }
                    func centre(_ values: [CameraLandmarkPoint]) -> CameraLandmarkPoint? {
                        guard !values.isEmpty else { return nil }
                        return .init(x: values.map(\.x).reduce(0,+)/Double(values.count), y: values.map(\.y).reduce(0,+)/Double(values.count))
                    }
                    let contour = CameraFaceLandmarks(faceContour: points(landmarks.faceContour), noseCrest: points(landmarks.noseCrest))
                    reading.face = contour.valid ? contour : nil
                    if let left = centre(points(landmarks.leftEye)), let right = centre(points(landmarks.rightEye)) {
                        reading.mouth = CameraMouthObservation.measure(outer: points(landmarks.outerLips), inner: points(landmarks.innerLips),
                            leftEye: left, rightEye: right, imageWidth: width, imageHeight: height,
                            confidence: Double(min(face.confidence, landmarks.confidence)))
                    }
                }
            }
        }
        if let prior = previous, prior.usedReceiptTime == reading.usedReceiptTime,
           let before = prior.mouth, let current = reading.mouth {
            reading.mouth = current.withVelocity(from: before, seconds: reading.t - prior.t)
        }
        reading.processedAt = ProcessInfo.processInfo.systemUptime
        previous = reading.mouth == nil ? nil : reading
        onReading(reading)
    }
}
