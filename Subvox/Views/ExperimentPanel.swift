import SwiftUI

struct ExperimentPanel: View {
    @Environment(AppModel.self) private var model
    @State private var confirmFinal = false

    var body: some View {
        @Bindable var model = model
        GroupBox("Training experiment") {
            VStack(alignment: .leading, spacing: 12) {
                Text("25% of word groups are reserved for the final test. The remaining 75% contains training and validation groups. Repeated words stay together; assignments never move.")
                    .font(.callout)
                splitCounts
                if let experiment = model.dataset.experiment, experiment.historicalTakeCount > 0 {
                    Text("\(experiment.historicalTakeCount) recordings predate this experiment and were previously available for training. Their holdout is a historical check, not a fresh test. Older takes have unknown session and speaking condition.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Text("\(model.pendingReviewCount) takes need review. Flagged and excluded takes do not enter fitting or scoring. Checks are heuristic; accepting a flag does not establish sensor quality.")
                    .font(.caption).foregroundStyle(.secondary)
                Picker("Evaluate condition", selection: $model.trainingCondition) {
                    ForEach(SpeechCondition.allCases) { Text($0.rawValue).tag($0) }
                }
                .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                Picker("Learning method", selection: $model.learningMethod) {
                    ForEach(LearningMethod.allCases) { Text($0.title).tag($0) }
                }
                .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                Picker("Candidate vocabulary", selection: $model.vocabularyScope) {
                    ForEach(VocabularyScope.allCases) { Text($0.title).tag($0) }
                }
                .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                Text("The sequence methods use six hypothesised jaw units, not verified sound equivalences. Learned alignment is experimental. Reduced vocabularies are fixed common-word lists; expected words outside them still count as misses.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    feature("Acceleration", .acceleration)
                    feature("Rotation", .rotation)
                    feature("Attitude", .attitude)
                }
                .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
                Text("Validation uses the same word decoder, vocabulary and rejection settings as Decode. Tune its thresholds in Decode before validating. Sound counts describe fitting coverage, not phoneme accuracy.")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = model.trainingError {
                    Text(error).font(.callout).foregroundStyle(.red)
                }
                if model.isEvaluating {
                    ProgressView().controlSize(.small)
                    Text(model.evaluationProgress).font(.caption)
                }
                HStack {
                    Button("Run validation", action: model.evaluate)
                        .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false || !model.storageReady)
                    Button("Compare learning methods", action: model.compareLearningMethods)
                        .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false || !model.storageReady)
                    Button(model.experimentLocked ? "Resume final test" : "Freeze and run final test…") { confirmFinal = true }
                        .disabled(model.finalReport != nil || model.isEvaluating || model.pendingReviewCount > 0 || model.session?.isFinished == false || !model.storageReady)
                }
                if !model.comparisonReports.isEmpty { comparisons }
                if let report = model.validationReport { reportView(report, title: "Validation") }
                if let report = model.finalReport { reportView(report, title: "Final test — frozen experiment") }
                if model.experimentLocked {
                    Text("The experiment is frozen: fitting data, settings and review decisions are locked. The final report is saved and cannot be rerun after completion.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }.padding(6)
        }
        .confirmationDialog("Freeze this experiment and reveal the final test?", isPresented: $confirmFinal) {
            Button("Freeze and test", action: model.runFinalTest)
        } message: {
            Text("All accepted development takes (training + validation) will fit the final decoder. Reserved takes will only be scored. Training, review decisions and settings will be locked, and the report will be saved.")
        }
    }

    private var splitCounts: some View {
        let takes = model.dataset.templates.filter { $0.purpose == nil || $0.purpose == .training }
        return HStack(spacing: 18) {
            ForEach([DataRole.training, .validation, .finalTest], id: \.rawValue) { role in
                let group = takes.filter { model.role(of: $0) == role }
                VStack(alignment: .leading) {
                    Text(role.rawValue).font(.caption).foregroundStyle(.secondary)
                    Text("\(group.filter(\.isUsable).count)/\(group.count) accepted takes").font(.callout)
                    Text("\(Set(group.map { TrainingExperiment.key($0.label) }).count) words").font(.caption)
                }
            }
        }
    }

    private func feature(_ name: String, _ flag: FeatureSet) -> some View {
        Toggle(name, isOn: Binding(get: { model.featureSet.contains(flag) }, set: { value in
            var set = model.featureSet
            if value { set.insert(flag) } else { set.remove(flag) }
            model.featureSet = set
        })).toggleStyle(.checkbox)
    }

    private func reportView(_ report: DecoderReport, title: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            Text(title).font(.headline)
            Text("\(report.settings.method.title) · \(report.settings.vocabularyScope.title) · frequency bias \(report.settings.commonBias, specifier: "%.2f")")
                .font(.caption).foregroundStyle(.secondary)
            Text("Top 1: \(percent(report.top1)) · Top 5: \(percent(report.top5)) · Accepted: \(percent(report.acceptance)) · Accepted & correct: \(percent(report.acceptedCorrect))")
                .font(.callout.monospacedDigit())
            Text("\(report.rows.count) evaluation takes · \(report.trainingIDs.count) fitting takes · \(report.vocabularyCount) buildable words. Every evaluation take stays in the denominator, including missing sounds and rejected guesses.")
                .font(.caption).foregroundStyle(.secondary)
            Text("\(report.rows.filter { !$0.representable }.count) expected words could not be built or are outside the vocabulary. Missing trained sounds: \(report.missingPhones.isEmpty ? "none" : report.missingPhones.joined(separator: ", ")).")
                .font(.caption).foregroundStyle(.secondary)
            if let collisions = report.collisionCount {
                Text("\(collisions) candidate words share their modeled unit sequence with another candidate; this model cannot use unit identity to distinguish those words.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let summary = report.trainingSummary { Text(summary).font(.caption).foregroundStyle(.secondary) }
            if let frequency = report.frequencyTop1 {
                Text("Frequency-only top 1: \(percent(frequency)); no motion evidence.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("\(Set(report.rows.compactMap(\.sessionID)).count) known evaluation sessions; \(report.rows.filter { $0.sessionID == nil }.count) takes have unknown sessions. This is a word-group holdout, not a test on wholly separate sessions.")
                .font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("Expected words and predictions") {
                LazyVStack(alignment: .leading, spacing: 5) {
                    ForEach(report.rows) { row in
                        Text("\(row.top1 ? "✓" : "×") \(row.expected) → \(row.candidates.first ?? "no candidate")\(row.accepted ? "" : " (rejected)")")
                            .font(.caption)
                    }
                }
            }
            if report.role == .validation, report.rows.contains(where: { !$0.top1 || !$0.accepted }) {
                Button("Collect more contexts for validation errors", action: model.requestMoreForFailing)
                    .disabled(model.experimentLocked || model.isEvaluating || model.motion.status != .streaming || model.session?.isFinished == false)
                Text("Requests new words containing sounds from mistaken words. This is a collection hint, not a diagnosis of which sound failed.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var comparisons: some View {
        DisclosureGroup("Learning comparison — validation only, frequency bias disabled") {
            VStack(alignment: .leading, spacing: 8) {
                Text("All methods use the same fitting and validation takes. Coverage is how many expected words occur in the fixed list; overall accuracy includes every take. Acceptance thresholds need separate calibration for sequence scores.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(model.comparisonReports.indices, id: \.self) { i in
                    let report = model.comparisonReports[i]
                    let eligible = report.rows.filter { $0.inVocabulary == true }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(report.settings.vocabularyScope.title) · \(report.settings.method.title)").font(.caption.weight(.semibold))
                        Text("Top 1 \(percent(report.top1)) · Top 5 \(percent(report.top5)) · Frequency-only \(percent(report.frequencyTop1 ?? 0)) · In list \(eligible.count)/\(report.rows.count)")
                            .font(.caption.monospacedDigit())
                        Text("Within list: \(eligible.filter(\.top1).count)/\(eligible.count) correct. Every take is included in the overall scores above.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func percent(_ value: Double) -> String { String(format: "%.1f%%", value * 100) }
}

struct TakeReviewDetails: View {
    @Environment(AppModel.self) private var model
    var take: Template

    var body: some View {
        let quality = take.quality
        let visit = model.dataset.visits?.first { $0.id == take.sessionID }
        VStack(alignment: .leading, spacing: 5) {
            Text("\(model.role(of: take)?.rawValue ?? "Unassigned") · \(take.condition?.rawValue ?? "Condition unknown") · \(take.reviewDecision.rawValue)")
                .font(.caption).foregroundStyle(take.reviewDecision == .pending ? Color.orange : Color.secondary)
            if let purpose = take.purpose, purpose != .training {
                Text(purpose.rawValue + " · diagnostic only, excluded from learning")
                    .font(.caption).foregroundStyle(.blue)
            }
            if let rate = quality.sampleRate {
                Text(String(format: "Measured %.1f Hz", rate)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if let visit {
                Text("Session \(visit.startedAt.formatted(date: .abbreviated, time: .shortened)) · \(visit.id.uuidString.prefix(8))\(visit.notes.isEmpty ? "" : " · " + visit.notes)")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Legacy take · session unknown").font(.caption).foregroundStyle(.secondary)
            }
            if !quality.issues.isEmpty {
                Text(quality.issues.joined(separator: " · ")).font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Button("Accept") { model.reviewTake(take.id, decision: .accepted) }
                    .disabled(quality.invalid || take.reviewDecision == .accepted)
                Button("Exclude") { model.reviewTake(take.id, decision: .excluded) }
                    .disabled(take.reviewDecision == .excluded)
                Button("Re-record") { model.rerecordTake(take.id) }
                    .disabled(model.motion.status != .streaming)
                Button("Review later") { model.reviewTake(take.id, decision: .pending) }
                    .disabled(take.reviewDecision == .pending)
            }
            .buttonStyle(.borderless).font(.caption)
            .disabled(model.experimentLocked || model.isEvaluating || model.session?.isFinished == false)
        }
    }
}
