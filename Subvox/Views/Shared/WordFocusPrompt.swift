import SwiftUI

/// A single attributed text block wraps naturally; the highlight never changes word widths
/// or moves the sentence as it advances.
struct WordFocusPrompt: View {
    var text: String
    var activeWordIndex: Int?
    var color: Color

    private var highlightedText: AttributedString {
        var result = AttributedString()
        for (index, word) in WordFocusGuide.words(in: text).enumerated() {
            if index > 0 { result.append(AttributedString(" ")) }
            var fragment = AttributedString(word)
            fragment.foregroundColor = activeWordIndex == nil || activeWordIndex == index ? color : color.opacity(0.35)
            if activeWordIndex == index { fragment.backgroundColor = color.opacity(0.18) }
            result.append(fragment)
        }
        return result
    }

    var body: some View {
        Text(highlightedText)
            .font(.system(size: 52, weight: .medium))
            .multilineTextAlignment(.center)
            .lineSpacing(8)
            .frame(maxWidth: .infinity)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityLabel(text)
    }
}
