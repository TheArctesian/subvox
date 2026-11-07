import Foundation

/// A guided recording run through a list of prompt words, with undo and skip.
struct TrainingSession: Equatable {
    let prompts: [String]
    let purposes: [RecordingPurpose]?
    private(set) var position = 0
    private var takes: [Int: Template.ID] = [:]

    init(prompts: [String], purposes: [RecordingPurpose]? = nil) {
        precondition(purposes == nil || purposes?.count == prompts.count)
        self.prompts = prompts
        self.purposes = purposes
    }

    var currentPurpose: RecordingPurpose {
        guard let purposes, purposes.indices.contains(position) else { return .training }
        return purposes[position]
    }

    var current: String? { prompts.indices.contains(position) ? prompts[position] : nil }
    var upNext: String? { prompts.indices.contains(position + 1) ? prompts[position + 1] : nil }
    var isFinished: Bool { position >= prompts.count }
    var recordedCount: Int { takes.count }
    var progress: Double { prompts.isEmpty ? 1 : Double(position) / Double(prompts.count) }

    mutating func record(_ id: Template.ID) {
        guard !isFinished else { return }
        takes[position] = id
        position += 1
    }

    mutating func skip() {
        guard !isFinished else { return }
        position += 1
    }

    /// Steps back one prompt. Returns the take recorded there, if any, for the caller to delete.
    mutating func undo() -> Template.ID? {
        guard position > 0 else { return nil }
        position -= 1
        return takes.removeValue(forKey: position)
    }
}
