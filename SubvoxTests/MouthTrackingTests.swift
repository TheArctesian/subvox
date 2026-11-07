import XCTest
@testable import Subvox

final class MouthTrackingTests: XCTestCase {
    private let eyes = [CameraLandmarkPoint(x: 0.4, y: 0.6), .init(x: 0.6, y: 0.6)]
    private let outer = [CameraLandmarkPoint(x: 0.38, y: 0.4), .init(x: 0.5, y: 0.45), .init(x: 0.62, y: 0.4), .init(x: 0.5, y: 0.35)]
    private let inner = [CameraLandmarkPoint(x: 0.42, y: 0.4), .init(x: 0.5, y: 0.42), .init(x: 0.58, y: 0.4), .init(x: 0.5, y: 0.38)]
    private func mouth() throws -> CameraMouthObservation {
        try XCTUnwrap(CameraMouthObservation.measure(outer: outer, inner: inner, leftEye: eyes[0], rightEye: eyes[1], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
    }
    func testMouthMeasurementsHaveKnownEyeSpacingUnitsAndVisiblePointLocations() throws {
        let m = try mouth()
        XCTAssertEqual(m.width, 1.2, accuracy: 1e-9)
        XCTAssertEqual(m.opening, 0.2, accuracy: 1e-9)
        XCTAssertEqual(m.innerArea, 0.08, accuracy: 1e-9)
        XCTAssertEqual(m.centreX, 0, accuracy: 1e-9)
        XCTAssertEqual(m.centreY, -1, accuracy: 1e-9)
        XCTAssertEqual(m.leftCorner, outer[0]); XCTAssertEqual(m.rightCorner, outer[2])
        XCTAssertEqual(m.openingTop, inner[1]); XCTAssertEqual(m.openingBottom, inner[3])
        XCTAssertNil(m.movementSpeed)
    }
    func testMeasurementsRemoveTranslationScaleAndRollUsingPixelAspectRatio() throws {
        let original = try mouth(), angle = 20 * Double.pi / 180
        func transform(_ p: CameraLandmarkPoint) -> CameraLandmarkPoint {
            let x = p.x * 1000 - 500, y = p.y * 1000 - 500
            return .init(x: ((x*cos(angle)-y*sin(angle)+500)*1.2+200)/2000,
                         y: ((x*sin(angle)+y*cos(angle)+500)*1.2+300)/1500)
        }
        let changed = try XCTUnwrap(CameraMouthObservation.measure(outer: outer.map(transform), inner: inner.map(transform),
            leftEye: transform(eyes[0]), rightEye: transform(eyes[1]), imageWidth: 2000, imageHeight: 1500, confidence: 0.9))
        for key in [\CameraMouthObservation.width, \.opening, \.innerArea, \.outerAspect, \.centreX, \.centreY] {
            XCTAssertEqual(original[keyPath: key], changed[keyPath: key], accuracy: 1e-8)
        }
        let reversed = try XCTUnwrap(CameraMouthObservation.measure(outer: outer, inner: inner, leftEye: eyes[1], rightEye: eyes[0], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
        XCTAssertEqual(reversed.centreY, original.centreY, accuracy: 1e-9)
    }
    func testInvalidTrackingStaysMissingAndClosedMouthIsARealZeroOpening() throws {
        XCTAssertNil(CameraMouthObservation.measure(outer: [], inner: inner, leftEye: eyes[0], rightEye: eyes[1], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
        XCTAssertNil(CameraMouthObservation.measure(outer: outer, inner: inner, leftEye: eyes[0], rightEye: eyes[0], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
        XCTAssertNil(CameraMouthObservation.measure(outer: outer, inner: inner, leftEye: eyes[0], rightEye: eyes[1], imageWidth: 1000, imageHeight: 1000, confidence: 0.1))
        var invalid = outer; invalid[0].x = .nan
        XCTAssertNil(CameraMouthObservation.measure(outer: invalid, inner: inner, leftEye: eyes[0], rightEye: eyes[1], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
        let closed = try XCTUnwrap(CameraMouthObservation.measure(outer: outer, inner: inner.map { .init(x: $0.x, y: 0.4) }, leftEye: eyes[0], rightEye: eyes[1], imageWidth: 1000, imageHeight: 1000, confidence: 0.9))
        XCTAssertEqual(closed.opening, 0, accuracy: 1e-10); XCTAssertEqual(closed.innerArea, 0, accuracy: 1e-10)
    }
    func testVelocityUsesTimestampsAndNeverBridgesLongOrInvalidGaps() throws {
        let old = try mouth(); var current = old; current.opening += 0.1
        let velocity = current.withVelocity(from: old, seconds: 0.05)
        XCTAssertEqual(try XCTUnwrap(velocity.openingSpeed), 2, accuracy: 1e-8)
        XCTAssertEqual(try XCTUnwrap(velocity.movementSpeed), 2, accuracy: 1e-8)
        for gap in [0, -1, 1, Double.nan] {
            XCTAssertNil(velocity.withVelocity(from: old, seconds: gap).movementSpeed)
        }
    }
    func testOverlayCoordinatesFollowUnmirroredAspectFitAndBottomLeftOrigin() {
        let fit = CameraOverlayGeometry.aspectFit(imageWidth: 640, imageHeight: 480, width: 320, height: 180)
        XCTAssertEqual(fit, CGRect(x: 40, y: 0, width: 240, height: 180))
        XCTAssertEqual(CameraOverlayGeometry.point(.init(x: 0, y: 0), in: fit), CGPoint(x: 40, y: 180))
        XCTAssertEqual(CameraOverlayGeometry.point(.init(x: 1, y: 1), in: fit), CGPoint(x: 280, y: 0))
    }
    func testTrackingCoverageDoesNotFillDroppedFramesAndLegacyReadingsStillDecode() throws {
        let old = try JSONDecoder().decode(CameraPoseReading.self, from: Data(#"{"t":1,"cameraID":"old"}"#.utf8))
        XCTAssertNil(old.mouth); XCTAssertNil(old.estimator)
        let m = try mouth()
        let frames = (0...30).map { CameraPoseReading(t: Double($0)/30, cameraID: "fixture", mouth: m) }
        let complete = CameraMouthStatistics.inspect(frames, start: 0, end: 1)
        XCTAssertEqual(complete.rate, 30, accuracy: 1e-8)
        XCTAssertEqual(complete.coverage, 1, accuracy: 1e-8)
        let gap = frames.filter { $0.t < 0.3 || $0.t > 0.7 }
        XCTAssertLessThan(CameraMouthStatistics.inspect(gap, start: 0, end: 1).coverage, 0.8)
    }
    @MainActor func testMouthReadinessAndEstimatorVersionDoNotTreatMissingPointsAsTracked() throws {
        let camera = CameraPoseService(), m = try mouth()
        for i in 0..<4 { camera.ingest(.init(t: 100 + Double(i)/30, cameraID: "fixture")) }
        XCTAssertFalse(camera.mouthReady(now: 100.1))
        for i in 4..<8 { camera.ingest(.init(t: 100 + Double(i)/30, cameraID: "fixture", mouth: m)) }
        XCTAssertTrue(camera.mouthReady(now: 100 + 7.0/30))
        XCTAssertFalse(camera.mouthReady(now: 101))
        camera.ingest(.init(t: 100.3, cameraID: "fixture"))
        XCTAssertFalse(camera.mouthReady(now: 100.3))
        let pose = CameraHeadPose(yaw: 0, pitch: 0, roll: 0, x: 0.5, y: 0.5, size: 0.3)
        let target = CameraPoseTarget(id: UUID(), cameraID: "fixture", pose: pose, angleTolerance: 5, positionTolerance: 0.05, sizeTolerance: 0.12)
        XCTAssertTrue(target.accepts(.init(t: 100, cameraID: "fixture", pose: pose)))
        XCTAssertFalse(target.accepts(.init(t: 100, cameraID: "fixture", pose: pose, estimator: "apple-vision-face-landmarks-v1")))
    }

    @MainActor func testSignalWorkflowStreamsNumericPointsWithoutStartingMicrophoneOrWhisper() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-points-only-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let workflow = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"))
        let audio = AudioService(), recorder = SessionRecorder(root: root.appending(path: "raw"))
        workflow.start(audio: audio, recorder: recorder, canRecord: { true })
        for _ in 0..<200 {
            if workflow.phase == .ready { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertEqual(workflow.phase, .ready)
        XCTAssertFalse(audio.isRunning)
        XCTAssertEqual(workflow.run?.recordsAudio, false)
        let session = try XCTUnwrap(recorder.currentSession)
        let reading = CameraPoseReading(t: session.startUptime + 0.1, cameraID: "fixture", mouth: try mouth(), imageWidth: 1000, imageHeight: 1000, estimator: "apple-vision-face-landmarks-v1")
        XCTAssertFalse(recorder.recordCamera(reading, sessionID: UUID()))
        workflow.camera.ingest(reading)
        XCTAssertEqual(recorder.cameraPointCount, 1)
        workflow.prepareForTermination()
        let lines = try String(contentsOf: session.directory.appending(path: "camera-points.ndjson"), encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(lines.count, 1)
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(row["sessionSeconds"] as? Double), 0.1, accuracy: 1e-8)
        let saved = try XCTUnwrap(row["reading"] as? [String: Any])
        let lips = try XCTUnwrap(saved["mouth"] as? [String: Any])
        XCTAssertEqual((lips["outerLips"] as? [[String: Any]])?.count, 4)
        let meta = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: session.directory.appending(path: "meta.json"))) as? [String: Any])
        XCTAssertEqual(meta["audioRecordingEnabled"] as? Bool, false)
        XCTAssertEqual(meta["videoRecordingEnabled"] as? Bool, false)
        XCTAssertEqual(meta["cameraPointSamples"] as? Int, 1)
        let names = try FileManager.default.contentsOfDirectory(atPath: session.directory.path)
        XCTAssertFalse(names.contains { ["caf", "wav", "mp4", "mov", "m4a"].contains(URL(fileURLWithPath: $0).pathExtension) })
        workflow.camera.ingest(.init(t: reading.t + 0.05, cameraID: "fixture", mouth: try mouth()))
        XCTAssertEqual(recorder.cameraPointCount, 1) // Late callbacks cannot append after stop.
    }
}
