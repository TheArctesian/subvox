import SwiftUI

/// The highlight changes color without changing the positions of words or readings.
struct WordFocusPrompt: View {
    var text: String
    var activeWordIndex: Int?
    var color: Color
    var pinyin: [String?] = []
    var verseLines: [String] = []
    var viewportHeight: CGFloat = 0

    private var displayText: String { verseLines.isEmpty ? text : verseLines.joined(separator: "\n") }

    private var highlightedText: AttributedString {
        var result = AttributedString()
        var index = 0
        for word in WordFocusGuide.fragments(in: displayText) {
            if word.allSatisfy(\.isWhitespace) {
                result.append(AttributedString(word))
                continue
            }
            var fragment = AttributedString(word)
            fragment.foregroundColor = activeWordIndex == nil || activeWordIndex == index ? color : Color.primary
            if activeWordIndex == index { fragment.backgroundColor = color.opacity(0.22) }
            result.append(fragment)
            index += 1
        }
        return result
    }

    var body: some View {
        let words = WordFocusGuide.words(in: text)
        let hasPinyin = pinyin.count == words.count && pinyin.contains(where: { $0 != nil })
        let compactSpacing = words.contains { $0.unicodeScalars.contains { $0.properties.isIdeographic } }
        Group {
            if hasPinyin || !verseLines.isEmpty {
                PromptRows(lineLengths: verseLines.map { WordFocusGuide.words(in: $0).count },
                           lineIndents: verseLines.map { CGFloat($0.prefix(while: { $0 == " " }).count) * 12 },
                           viewportHeight: viewportHeight, spacing: compactSpacing ? 4 : 12) {
                    ForEach(words.indices, id: \.self) { index in
                        VStack(spacing: 0) {
                            if hasPinyin {
                                Text(pinyin[index] ?? " ").font(.title3).foregroundStyle(.secondary)
                            }
                            Text(words[index]).font(.system(size: 52, weight: .medium))
                                .foregroundStyle(activeWordIndex == nil || activeWordIndex == index ? color : Color.primary)
                                .background(activeWordIndex == index ? color.opacity(0.22) : .clear)
                        }.fixedSize(horizontal: false, vertical: true)
                    }
                }
            } else {
                Text(highlightedText)
                    .font(.system(size: 52, weight: .medium))
                    .multilineTextAlignment(.center)
                    .lineSpacing(8)
            }
        }
        .frame(maxWidth: .infinity)
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(displayText)
    }
}

/// Preserve source lines and keep pronunciation labels with their characters.
private struct PromptRows: Layout {
    var lineLengths: [Int]
    var lineIndents: [CGFloat]
    var viewportHeight: CGFloat
    var spacing: CGFloat
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = rows(for: subviews, width: proposal.width ?? .infinity)
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0,
                      height: rows.map(\.height).reduce(0, +) + 8 * CGFloat(max(0, rows.count - 1)))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = rows(for: subviews, width: bounds.width)
        let stanzaWidth = rows.map(\.width).max() ?? 0
        var y = bounds.minY
        for row in rows {
            var x = bounds.midX - (lineLengths.isEmpty ? row.width : stanzaWidth) / 2 + row.indent
            for index in row.indices {
                let proposal = ProposedViewSize(width: max(1, bounds.width - (lineIndents.max() ?? 0)), height: nil)
                let size = subviews[index].sizeThatFits(proposal)
                subviews[index].place(at: CGPoint(x: x, y: y), anchor: .topLeading, proposal: proposal)
                x += size.width + (spacing + (row.coupletBreak == index + 1 ? row.coupletGap : 0))
            }
            y += row.height + 8
        }
    }

    private func rows(for subviews: Subviews, width: CGFloat) -> [Row] {
        let maximumIndent = lineIndents.max() ?? 0
        let proposal = width.isFinite ? ProposedViewSize(width: max(1, width - maximumIndent), height: nil) : .unspecified
        let sizes = subviews.map { $0.sizeThatFits(proposal) }
        let lineHeight = sizes.map(\.height).max() ?? 0
        // Reserve space for the cue, translation and instructions in the scrollable panel.
        let compact = viewportHeight > 0 && CGFloat(lineLengths.count) * (lineHeight + 8) > viewportHeight - 185
        var lines: [(range: Range<Int>, indent: CGFloat)] = []
        var end = 0
        for (length, indent) in zip(lineLengths, lineIndents) where length > 0 {
            lines.append((end..<(end + length), indent))
            end += length
        }
        if end != sizes.count || lines.isEmpty { lines = [(sizes.indices, 0)] }

        var result: [Row] = []
        var index = 0
        while index < lines.count {
            let (line, indent) = lines[index]
            if compact, index + 1 < lines.count {
                let next = lines[index + 1]
                let pair = line.lowerBound..<next.range.upperBound
                let gap = 8 + next.indent
                let pairWidth = rowWidth(pair, sizes: sizes) + gap + indent
                if pairWidth <= width {
                    result.append(Row(indices: pair, width: pairWidth, height: lineHeight, indent: indent,
                                      coupletBreak: line.upperBound, coupletGap: gap))
                    index += 2
                    continue
                }
            }
            // A verse line wraps within itself only when the full line is wider than the window.
            var start = line.lowerBound
            var used: CGFloat = 0
            for cell in line {
                if cell > start, indent + used + spacing + sizes[cell].width > width {
                    result.append(Row(indices: start..<cell, width: used + indent, height: lineHeight, indent: indent))
                    start = cell
                    used = 0
                }
                used += sizes[cell].width + (cell == start ? 0 : spacing)
            }
            if start < line.upperBound {
                result.append(Row(indices: start..<line.upperBound, width: used + indent, height: lineHeight, indent: indent))
            }
            index += 1
        }
        return result
    }

    private func rowWidth(_ range: Range<Int>, sizes: [CGSize]) -> CGFloat {
        sizes[range].reduce(0) { $0 + $1.width } + spacing * CGFloat(max(0, range.count - 1))
    }

    private struct Row {
        let indices: Range<Int>
        let width: CGFloat
        let height: CGFloat
        var indent: CGFloat = 0
        var coupletBreak: Int? = nil
        var coupletGap: CGFloat = 0
    }
}
