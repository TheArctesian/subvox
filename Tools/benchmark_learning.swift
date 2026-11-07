import Foundation

/// Offline validation only. There is deliberately no final-test option.
@main struct LearningBenchmark {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print("Usage: benchmark_learning.sh DATASET_JSON OUTPUT_JSON")
            return
        }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
        guard let experiment = dataset.experiment, experiment.frozenAt == nil else {
            throw NSError(domain: "Benchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Requires an unfrozen experiment; only validation will be scored."])
        }
        let lexicon = Lexicon(tsv: try String(contentsOf: root.appending(path: "Corpus/lexicon.tsv"), encoding: .utf8) +
                              String(contentsOf: root.appending(path: "Corpus/words.tsv"), encoding: .utf8))
        let vocabulary: [(word: String, count: Int)] = try String(contentsOf: root.appending(path: "Corpus/words.tsv"), encoding: .utf8)
            .split(separator: "\n").compactMap { line in
                let fields = line.split(separator: "\t")
                guard fields.count >= 3, let count = Int(fields[2]) else { return nil }
                return (String(fields[0]), count)
            }
        let training = experiment.fittingTakes(from: dataset.templates)
        let testing = dataset.templates.filter {
            $0.isUsable && experiment.role(of: $0) == .validation && experiment.matchesCondition($0)
        }
        let reserved = Set(dataset.templates.filter { experiment.role(of: $0) == .finalTest }.map(\.id))
        precondition(Set(training.map(\.id)).isDisjoint(with: reserved))
        precondition(Set(testing.map(\.id)).isDisjoint(with: reserved))
        print("\(training.count) training, \(testing.count) validation; reserved final-test data will not be fitted or scored.")
        var reports: [DecoderReport] = []
        for scope in VocabularyScope.allCases {
            for method in LearningMethod.allCases {
                var settings = experiment.settings
                settings.method = method
                settings.vocabularyScope = scope
                settings.commonBias = 0
                let start = Date()
                let report = try await DecoderEvaluation.run(training: training, testing: testing, role: .validation,
                    lexicon: lexicon, vocabulary: vocabulary, settings: settings)
                reports.append(report)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(reports).write(to: URL(fileURLWithPath: args[2]), options: .atomic)
                print(String(format: "%@ | %@: top1 %.1f%% top5 %.1f%% in-vocab %d/%d accepted %.1f%%, %.2f s", scope.rawValue,
                    method.rawValue, 100 * report.top1, 100 * report.top5,
                    report.rows.filter { $0.inVocabulary == true }.count, report.rows.count,
                    100 * report.acceptance, Date().timeIntervalSince(start)))
                fflush(stdout)
            }
        }
    }
}
