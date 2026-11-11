import SwiftUI

/// One continuous microphone recording and playback for the text shown on the practice page.
struct VoicePracticePanel: View {
    @Environment(AppModel.self) private var model
    let target: VoicePracticeTarget
    let title: String
    var draftTarget: VoicePracticeTarget? = nil

    var body: some View {
        let voice = model.voicePractice
        let runs = recentRuns(for: target)
        let otherTargets = voice.runs.filter { belongsToItem($0.target, target) && $0.target != target }
        let otherRuns = (Array(runs.dropFirst(2)) + otherTargets).sorted { $0.createdAt > $1.createdAt }
        VStack(alignment: .leading, spacing: 9) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { actions(runs) }
                VStack(alignment: .leading, spacing: 8) { actions(runs) }
            }
            if let error = voice.error {
                Label(error, systemImage: "exclamationmark.circle")
                    .font(.callout).foregroundStyle(.orange)
            } else if let latest = runs.first {
                Text("Latest run · \(Duration.seconds(latest.duration).formatted(.time(pattern: .minuteSecond)))")
                    .font(.caption).foregroundStyle(.secondary)
            } else if !voice.isBusy {
                Text("Record one continuous take. Audio stays on this Mac.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !otherRuns.isEmpty {
                DisclosureGroup("Other saved runs") {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(otherRuns) { run in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(run.target.text).font(.callout).lineLimit(1)
                                    Text("\(run.createdAt.formatted(date: .abbreviated, time: .shortened)) · \(Duration.seconds(run.duration).formatted(.time(pattern: .minuteSecond)))")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                playButton(run, label: "Play full run")
                            }
                        }
                    }
                    .padding(.top, 6)
                }
                .font(.callout)
                .disabled(voice.isBusy)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        .onDisappear { stopPlayback(for: target) }
        .onChange(of: target) { previous, _ in
            stopPlayback(for: previous)
            voice.clearError()
        }
    }

    @ViewBuilder private func actions(_ runs: [VoicePracticeRun]) -> some View {
        Button("Record with microphone", systemImage: "mic") {
            Task { await model.startVoicePractice(target) }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!model.canStartVoicePractice)
        .accessibilityLabel("Record \(title) with microphone")

        if let latest = runs.first {
            playButton(latest, label: runs.count == 1 ? "Play full run" : "Play latest run")
        }
        if runs.count > 1 { playButton(runs[1], label: "Play previous run") }
    }

    private func playButton(_ run: VoicePracticeRun, label: LocalizedStringKey) -> some View {
        let voice = model.voicePractice
        let playing = voice.playingRunID == run.id
        return Button(playing ? "Stop playback" : label, systemImage: playing ? "stop.fill" : "play.fill") {
            if playing {
                voice.stopPlayback()
            } else {
                model.referenceSpeaker.stop()
                voice.play(run)
            }
        }
        .disabled(model.navigationLocked)
        .help(run.createdAt.formatted(date: .abbreviated, time: .shortened))
    }

    private func stopPlayback(for target: VoicePracticeTarget) {
        let voice = model.voicePractice
        guard let playing = voice.runs.first(where: { $0.id == voice.playingRunID }) else { return }
        if belongsToItem(playing.target, target) || playing.target == draftTarget {
            voice.stopPlayback()
        }
    }

    private func belongsToItem(_ lhs: VoicePracticeTarget, _ rhs: VoicePracticeTarget) -> Bool {
        lhs.mode == rhs.mode && lhs.language == rhs.language && lhs.itemID == rhs.itemID
    }

    private func recentRuns(for target: VoicePracticeTarget) -> [VoicePracticeRun] {
        let voice = model.voicePractice
        let savedRuns = voice.recentRuns(for: target)
        guard let draftTarget, draftTarget.text.utf8.elementsEqual(target.text.utf8), draftTarget != target else { return savedRuns }
        return (savedRuns + voice.recentRuns(for: draftTarget)).sorted { $0.createdAt > $1.createdAt }
    }
}
