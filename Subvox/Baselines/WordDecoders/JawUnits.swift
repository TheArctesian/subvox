import Foundation

/// A conservative experimental prior, not sensor-verified articulatory classes.
/// Five vowel jaw classes share evidence; all known consonants share one model.
enum JawUnits {
    static let version = "jaw6-v1"

    static func unit(for phone: String) -> String? {
        if let jaw = Lexicon.jawClass(phone) { return jaw }
        return Phoneme.bySymbol[phone] == nil ? nil : "consonant"
    }
}
