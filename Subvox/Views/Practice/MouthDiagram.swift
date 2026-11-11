import SwiftUI

/// A side view of the mouth at one sound's target shape: lips on the right, throat on the left.
/// The tongue and lips carry colour; the fixed anatomy stays quiet so the moving parts read first.
struct MouthDiagram: View {
    let diagrams: ArticulationDiagrams
    let posture: ArticulationDiagrams.Posture
    var note: ArticulationDiagrams.Note?
    let soundName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Canvas { context, size in draw(in: &context, size: size) }
                .aspectRatio(aspect, contentMode: .fit)
                .frame(maxWidth: 340)
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.35)))
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

    private var aspect: CGFloat {
        let v = diagrams.view
        return CGFloat((v[2] - v[0]) / (v[3] - v[1]))
    }

    private func style(_ part: String) -> (GraphicsContext.Shading, CGFloat, Bool) {
        switch part {
        case "tongue": (.color(.pink), 2.2, false)
        case "upperLip", "lowerLip": (.color(.orange), 2.2, false)
        case "velum": (.color(.purple), 1.8, false)
        case "upperTeeth", "lowerTeeth": (.style(.secondary), 1.1, true)
        case "nasalCavity": (.style(.tertiary), 1, false)
        default: (.style(.secondary), 1.3, false)
        }
    }

    private func draw(in context: inout GraphicsContext, size: CGSize) {
        let v = diagrams.view
        let scale = min(size.width / (v[2] - v[0]), size.height / (v[3] - v[1]))
        func point(_ p: ArticulationDiagrams.Point) -> CGPoint {
            CGPoint(x: (p.x - v[0]) * scale, y: (v[3] - p.y) * scale)
        }
        context.clip(to: Path(CGRect(origin: .zero, size: size)))
        for part in diagrams.parts {
            guard let points = posture[part], let first = points.first else { continue }
            var path = Path()
            path.move(to: point(first))
            for p in points.dropFirst() { path.addLine(to: point(p)) }
            let (shading, width, closed) = style(part)
            if closed { path.closeSubpath() }
            context.stroke(path, with: shading, style: StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round))
        }
    }
}
