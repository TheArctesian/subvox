import AVFoundation
import SwiftUI

private final class PosePreviewNSView: NSView {
    let preview = AVCaptureVideoPreviewLayer()
    override init(frame: NSRect) { super.init(frame: frame); wantsLayer = true; layer?.addSublayer(preview); preview.videoGravity = .resizeAspect }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layout() { super.layout(); preview.frame = bounds }
}
private struct PoseCameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> PosePreviewNSView { let view = PosePreviewNSView(); view.preview.session = session; return view }
    func updateNSView(_ view: PosePreviewNSView, context: Context) {
        if view.preview.session !== session { view.preview.session = session }
        if let connection = view.preview.connection, connection.isVideoMirroringSupported {
            if connection.automaticallyAdjustsVideoMirroring { connection.automaticallyAdjustsVideoMirroring = false }
            if connection.isVideoMirrored { connection.isVideoMirrored = false }
        }
    }
}

struct CameraPoseGuide: View {
    let camera: CameraPoseService
    var target: CameraPoseTarget?
    var compact = false
    var allowsMovement = false
    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { _ in
            let now = ProcessInfo.processInfo.systemUptime
            let fresh = camera.latest.flatMap { now - $0.t <= 0.4 ? $0 : nil }
            let matching = fresh.map { target?.accepts($0) == true } ?? false
            let stable = target?.isStable(camera.recent, now: now) == true
            HStack(spacing: 18) {
                ZStack {
                    PoseCameraPreview(session: camera.session)
                    Canvas { context, size in
                        let content = CameraOverlayGeometry.aspectFit(imageWidth: fresh?.imageWidth ?? 1280,
                            imageHeight: fresh?.imageHeight ?? 720, width: size.width, height: size.height)
                        func mapped(_ point: CameraLandmarkPoint) -> CGPoint { CameraOverlayGeometry.point(point, in: content) }
                        if let mouth = fresh?.mouth {
                            func contour(_ points: [CameraLandmarkPoint], color: Color) {
                                guard let first = points.first else { return }
                                var path = Path(); path.move(to: mapped(first))
                                for point in points.dropFirst() { path.addLine(to: mapped(point)) }
                                path.closeSubpath()
                                context.stroke(path, with: .color(color), lineWidth: compact ? 1 : 1.5)
                                for point in points {
                                    let p = mapped(point), radius = compact ? 0.8 : 1.6
                                    context.fill(Path(ellipseIn: CGRect(x: p.x-radius, y: p.y-radius, width: 2*radius, height: 2*radius)), with: .color(color))
                                }
                            }
                            contour(mouth.outerLips, color: .cyan); contour(mouth.innerLips, color: .pink)
                            var opening = Path(); opening.move(to: mapped(mouth.openingTop)); opening.addLine(to: mapped(mouth.openingBottom))
                            context.stroke(opening, with: .color(.pink), lineWidth: 2)
                            for point in [mouth.leftCorner, mouth.rightCorner] {
                                let p = mapped(point), radius = compact ? 2.0 : 3.0
                                context.fill(Path(ellipseIn: CGRect(x: p.x-radius, y: p.y-radius, width: 2*radius, height: 2*radius)), with: .color(.yellow))
                            }
                            let centre = mapped(mouth.centre)
                            context.fill(Path(ellipseIn: CGRect(x: centre.x-2, y: centre.y-2, width: 4, height: 4)), with: .color(.white))
                            if allowsMovement {
                                var eyes=Path();eyes.move(to:mapped(mouth.leftEye));eyes.addLine(to:mapped(mouth.rightEye))
                                context.stroke(eyes,with:.color(.green.opacity(0.7)),lineWidth:1)
                                for point in [mouth.leftEye,mouth.rightEye] {
                                    let p=mapped(point),radius=compact ? 1.5 : 2.5
                                    context.fill(Path(ellipseIn:CGRect(x:p.x-radius,y:p.y-radius,width:radius*2,height:radius*2)),with:.color(.green))
                                }
                            }
                        }
                        if let target, !allowsMovement {
                            let point = mapped(.init(x: target.pose.x, y: target.pose.y))
                            let rect = CGRect(x: point.x - 16, y: point.y - 16, width: 32, height: 32)
                            context.stroke(Path(ellipseIn: rect), with: .color(.white), style: .init(lineWidth: 2, dash: [4, 3]))
                        }
                        if let pose = fresh?.pose {
                            let point = mapped(.init(x: pose.x, y: pose.y))
                            context.fill(Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)), with: .color(allowsMovement ? .cyan : matching ? .green : .orange))
                        }
                    }
                }.frame(width: compact ? 124 : 320, height: compact ? 70 : 180)
                    .background(.black, in: RoundedRectangle(cornerRadius: 8)).clipShape(RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel("Local camera preview with head-position target, lip contours, mouth opening, corners and centre")
                VStack(alignment: .leading, spacing: compact ? 3 : 7) {
                    Text(allowsMovement ? (fresh?.pose != nil && fresh?.mouth != nil ? "Tracking head and mouth · move naturally" : "Keep your face and mouth visible") : target == nil ? "Choose a comfortable neutral pose" : stable ? "Pose matched · ready" : matching ? "Hold still for one second" : "Return to the saved camera pose")
                        .font(compact ? .caption.weight(.semibold) : .title3.weight(.semibold)).foregroundStyle(stable ? Color.green : Color.primary)
                    if let pose = fresh?.pose {
                        let reference = target?.pose ?? pose
                        Text(target == nil || allowsMovement
                             ? String(format: "Yaw %+.1f°   Pitch %+.1f°   Roll %+.1f°", pose.yaw, pose.pitch, pose.roll)
                             : String(format: "Turn %+.1f°   Nod %+.1f°   Tilt %+.1f°", pose.yaw - reference.yaw, pose.pitch - reference.pitch, pose.roll - reference.roll))
                            .font(compact ? .caption.monospacedDigit() : .callout.monospacedDigit())
                        if let target, !compact, !allowsMovement {
                            Text(String(format: "Position offset %.1f%% · size change %+.1f%%", hypot(pose.x-target.pose.x, pose.y-target.pose.y)*100, (pose.size/target.pose.size-1)*100))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    } else {
                        Text(camera.isRunning ? "Keep one face visible to the camera." : "Enable the Mac camera to begin.").foregroundStyle(.secondary)
                    }
                    if let mouth = fresh?.mouth {
                        Text(String(format: "Opening %.3f · width %.3f · movement %@/s", mouth.opening, mouth.width,
                            mouth.movementSpeed.map { String(format: "%.2f", $0) } ?? "—"))
                            .font(.caption.monospacedDigit())
                    } else if camera.isRunning {
                        Text("Lip tracking unavailable · keep your mouth visible").font(.caption).foregroundStyle(.orange)
                    }
                    if !compact {
                        Text(String(format: "%.1f camera samples/s · lip confidence %.0f%%", camera.sampleRate, (fresh?.mouth?.confidence ?? 0)*100))
                            .font(.caption).foregroundStyle(.secondary)
                        Text("Cyan: lips · pink: opening · yellow: corners · green: rigid eye reference. Measurements use eye-spacing units.")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(allowsMovement ? "You can turn, nod and shift comfortably. The camera measures head movement; you do not need to return to a target." : "Camera geometry is an estimate. Match it before each take; keep your chair, screen and camera fixed.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        Text("Numeric lip points, pose and motion measurements are saved during the test. Audio and video are not recorded.").font(.caption).foregroundStyle(.secondary)
                    }
                    if let error = camera.error { Text(error).foregroundStyle(.orange).font(.caption) }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}
