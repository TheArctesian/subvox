import AppKit
import SwiftUI

/// A side view of the mouth at one sound's target shape: lips on the right, throat on the left.
/// Drawn as a classic midsagittal section: soft tissue filled, the airway left open, and the parts
/// a learner moves (tongue, lips, soft palate) in stronger colour than the fixed anatomy.
struct MouthDiagram: View {
    let diagrams: ArticulationDiagrams
    let posture: ArticulationDiagrams.Posture
    var note: ArticulationDiagrams.Note?
    let soundName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Canvas { context, size in draw(in: &context, size: size) }
                .aspectRatio(aspect, contentMode: .fit)
                .frame(maxWidth: 320)
                .background(Self.airway)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.separator))
                .accessibilityElement()
                .accessibilityLabel(String(localized: "Side view of the mouth for \(soundName)"))
                .accessibilityAddTraits(.isImage)
            Group {
                switch note {
                case .start: Text("Side view, lips on the right. This sound moves; the drawing shows where it starts.")
                case .approximate: Text("Side view, lips on the right. Approximate: drawn from closely related mouth shapes.")
                case nil: Text("Side view, lips on the right.")
                }
            }
            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private static func dynamic(_ light: NSColor, _ dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { $0.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light })
    }
    private static func rgb(_ hex: UInt32) -> NSColor {
        NSColor(srgbRed: CGFloat(hex >> 16 & 0xff) / 255, green: CGFloat(hex >> 8 & 0xff) / 255, blue: CGFloat(hex & 0xff) / 255, alpha: 1)
    }
    static let airway = dynamic(rgb(0xFBF8F6), rgb(0x1E1B1A))
    static let tissue = dynamic(rgb(0xF2DCD3), rgb(0x4A3A35))
    static let tissueEdge = dynamic(rgb(0xC9A196), rgb(0x8A6A61))
    static let tongue = dynamic(rgb(0xE88A8F), rgb(0xC0626A))
    static let tongueEdge = dynamic(rgb(0xB8505A), rgb(0xE0959B))
    static let lip = dynamic(rgb(0xD0606E), rgb(0xE88F99))
    static let tooth = dynamic(rgb(0xFFFFFF), rgb(0xD9D4CF))
    static let toothEdge = dynamic(rgb(0xA89C95), rgb(0x8E8580))
    static let velum = dynamic(rgb(0xB57FAF), rgb(0xC99BC4))

    private var aspect: CGFloat {
        let v = diagrams.view
        return CGFloat((v[2] - v[0]) / (v[3] - v[1]))
    }

    /// Chaikin corner cutting: rounds the model's polyline joints without moving the shape.
    static func smoothed(_ points: [CGPoint], passes: Int = 2) -> [CGPoint] {
        var p = points
        for _ in 0..<passes where p.count > 2 {
            var out = [p[0]]
            for (a, b) in zip(p, p.dropFirst()) {
                out.append(CGPoint(x: 0.75 * a.x + 0.25 * b.x, y: 0.75 * a.y + 0.25 * b.y))
                out.append(CGPoint(x: 0.25 * a.x + 0.75 * b.x, y: 0.25 * a.y + 0.75 * b.y))
            }
            out.append(p[p.count - 1])
            p = out
        }
        return p
    }

    /// `lip` continued past the point nearest `end`, so a wall runs on round the lip to the face.
    static func continuation(of lip: [CGPoint], after end: CGPoint) -> [CGPoint] {
        guard let i = lip.indices.min(by: { hypot(lip[$0].x - end.x, lip[$0].y - end.y) < hypot(lip[$1].x - end.x, lip[$1].y - end.y) }) else { return [] }
        return Array(lip[(i + 1)...])
    }

    private func draw(in context: inout GraphicsContext, size: CGSize) {
        let v = diagrams.view
        let scale = min(size.width / (v[2] - v[0]), size.height / (v[3] - v[1]))
        func points(_ part: String) -> [CGPoint] {
            (posture[part] ?? []).map { CGPoint(x: ($0.x - v[0]) * scale, y: (v[3] - $0.y) * scale) }
        }
        func path(_ pts: [CGPoint], closed: Bool) -> Path {
            var p = Path()
            guard let first = pts.first else { return p }
            p.move(to: first)
            for q in pts.dropFirst() { p.addLine(to: q) }
            if closed { p.closeSubpath() }
            return p
        }
        let pad: CGFloat = 40
        func line(_ width: CGFloat) -> StrokeStyle { StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round) }

        // Roof: back of the throat, soft and hard palate, upper teeth ridge, then round the upper lip.
        let outer = points("outer"), upperLip = points("upperLip")
        if let first = outer.first, let last = outer.last {
            let wall = Self.smoothed(outer + Self.continuation(of: upperLip, after: last))
            let end = wall.last ?? last
            // Straight up from the lip: everything in front of the face stays open air.
            let roof = path(wall + [CGPoint(x: end.x, y: -pad), CGPoint(x: -pad, y: -pad), CGPoint(x: -pad, y: first.y)], closed: true)
            context.fill(roof, with: .color(Self.tissue))
            context.stroke(path(wall, closed: false), with: .color(Self.tissueEdge), style: line(1.2))
        }
        // Floor: tongue root, floor of the mouth, lower teeth ridge, then round the lower lip.
        let inner = points("inner"), lowerLip = points("lowerLip")
        if let first = inner.first, let last = inner.last {
            let wall = Self.smoothed(inner + Self.continuation(of: lowerLip, after: last))
            let end = wall.last ?? last
            let floor = path(wall + [CGPoint(x: end.x, y: size.height + pad), CGPoint(x: first.x, y: size.height + pad)], closed: true)
            context.fill(floor, with: .color(Self.tissue))
            context.stroke(path(wall, closed: false), with: .color(Self.tissueEdge), style: line(1.2))
        }
        // The nasal passage above the palate, faintly.
        context.stroke(path(Self.smoothed(points("nasalCavity")), closed: false), with: .color(Self.tissueEdge.opacity(0.6)),
                       style: StrokeStyle(lineWidth: 1, lineCap: .round, dash: [3, 3]))
        // Soft palate: hangs from the roof; lowered for nasal sounds.
        let velum = path(Self.smoothed(points("velum")), closed: true)
        context.fill(velum, with: .color(Self.tissue))
        context.stroke(velum, with: .color(Self.velum), style: line(2))
        // Teeth.
        for part in ["upperTeeth", "lowerTeeth"] {
            let tooth = path(points(part), closed: true)
            context.fill(tooth, with: .color(Self.tooth))
            context.stroke(tooth, with: .color(Self.toothEdge), style: line(1))
        }
        // Tongue, the main mover.
        let tongue = path(Self.smoothed(points("tongue")), closed: true)
        context.fill(tongue, with: .color(Self.tongue))
        context.stroke(tongue, with: .color(Self.tongueEdge), style: line(1.8))
        // Lips.
        for part in ["upperLip", "lowerLip"] {
            context.stroke(path(Self.smoothed(points(part)), closed: false), with: .color(Self.lip), style: line(2.6))
        }
    }
}
