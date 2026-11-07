import Foundation

struct Dataset: Codable, Sendable {
    /// Each take is one mouthed word, labelled with that word.
    var templates: [Template]
    /// Raised per phoneme when its test fails and more examples are requested.
    var phoneTargets: [String: Int]?
    var visits: [RecordingVisit]?
    var experiment: TrainingExperiment?

    static let starter = Dataset(templates: [])
}

/// Persists the training set as JSON in Application Support.
enum TemplateStore {
    static var url: URL {
        let dir = URL.applicationSupportDirectory.appending(path: "Subvox", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appending(path: "dataset.json")
    }

    static func load() -> Dataset {
        guard let data = try? Data(contentsOf: url),
              let dataset = try? JSONDecoder().decode(Dataset.self, from: data) else { return .starter }
        return dataset
    }

    @discardableResult
    static func save(_ dataset: Dataset) -> Bool {
        do {
            let data = try JSONEncoder().encode(dataset)
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            print("Subvox: failed to save dataset: \(error)")
            return false
        }
    }
}
