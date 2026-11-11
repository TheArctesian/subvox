import Foundation
import Observation

struct LiteraturePin: Codable, Equatable, Identifiable {
    let pieceID: String
    let language: AppLanguage
    private let modeID: String

    init(pieceID: String, language: AppLanguage, mode: TrainMode) {
        self.pieceID = pieceID
        self.language = language
        modeID = mode.rawValue
    }

    var mode: TrainMode { TrainMode(rawValue: modeID) ?? .speech }
    var id: String { "\(modeID):\(language.tag):\(pieceID)" }
}

@MainActor @Observable
final class LiteraturePins {
    private static let key = "practice.literaturePins"
    @ObservationIgnored private let defaults: UserDefaults
    private(set) var items: [LiteraturePin]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        items = defaults.data(forKey: Self.key).flatMap {
            try? JSONDecoder().decode([LiteraturePin].self, from: $0)
        } ?? []
    }

    func contains(_ pin: LiteraturePin) -> Bool { items.contains(pin) }

    func toggle(_ pin: LiteraturePin) {
        var next = items
        if let index = next.firstIndex(of: pin) {
            next.remove(at: index)
        } else {
            next.insert(pin, at: 0)
        }
        guard let data = try? JSONEncoder().encode(next) else { return }
        defaults.set(data, forKey: Self.key)
        items = next
    }
}
