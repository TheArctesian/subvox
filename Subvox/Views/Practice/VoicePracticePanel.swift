import SwiftUI

/// One continuous microphone recording and playback for the text shown on the practice page.
struct VoicePracticePanel: View {
    @Environment(AppModel.self) private var model
    let target: VoicePracticeTarget
    let title: String
    var draftTarget: VoicePracticeTarget? = nil
    var referenceTitle: LocalizedStringKey? = nil
    var recordTitle: LocalizedStringKey = "Record with microphone"
    var showsOtherRuns = true

    var body: some View {
        let voice = model.voicePractice
        let runs = recentRuns(for: target)
        let otherTargets = showsOtherRuns ? voice.runs.filter { belongsToItem($0.target, target) && $0.target != target } : []
        let otherRuns = showsOtherRuns ? (Array(runs.dropFirst(2)) + otherTargets).sorted { $0.createdAt > $1.createdAt } : []
        let active = (runs + otherRuns).first { $0.id == voice.playingRunID }
        VStack(alignment: .leading, spacing: 10) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { actions(runs) }
                VStack(alignment: .leading, spacing: 8) { actions(runs) }
            }
            if let active {
                VoiceRunTransport(title: transportTitle(for: active, in: runs))
            }
            if let error = voice.error, voice.errorTarget == target || otherRuns.contains(where: { $0.target == voice.errorTarget }) {
                Label(error, systemImage: "exclamationmark.circle")
                    .font(.callout).foregroundStyle(.orange)
            } else if active == nil, let latest = runs.first {
                Text("Latest run · \(Duration.seconds(latest.duration).formatted(.time(pattern: .minuteSecond)))")
                    .font(.caption).foregroundStyle(.secondary)
            } else if runs.isEmpty && !voice.isBusy {
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
            if voice.errorTarget == previous { voice.clearError() }
        }
    }

    @ViewBuilder private func actions(_ runs: [VoicePracticeRun]) -> some View {
        Button(recordTitle, systemImage: "mic") {
            Task { await model.startVoicePractice(target) }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!model.canStartVoicePractice)
        .accessibilityLabel("Record \(title) with microphone")

        if let referenceTitle {
            ReferenceButton(text: target.text, language: target.language, compact: false, subject: title,
                            listenTitle: referenceTitle)
        }
        if let latest = runs.first {
            playButton(latest, label: runs.count == 1 ? "Play full run" : "Play latest run")
        }
        if runs.count > 1 { playButton(runs[1], label: "Play previous run") }
    }

    private func playButton(_ run: VoicePracticeRun, label: LocalizedStringKey) -> some View {
        let voice = model.voicePractice
        let playing = voice.playingRunID == run.id
        return Button(label, systemImage: playing ? "speaker.wave.2.fill" : "play.fill") {
            if !playing {
                model.referenceSpeaker.stop()
                voice.play(run)
            }
        }
        .disabled(model.navigationLocked || playing)
        .help(run.createdAt.formatted(date: .abbreviated, time: .shortened))
    }

    private func transportTitle(for run: VoicePracticeRun, in runs: [VoicePracticeRun]) -> Text {
        if run == runs.first { return Text("Latest run") }
        if run == runs.dropFirst().first { return Text("Previous run") }
        return Text(run.createdAt.formatted(date: .abbreviated, time: .shortened))
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

struct VoiceRunTransport: View {
    @Environment(AppModel.self) private var model
    let title: Text

    var body: some View {
        let voice = model.voicePractice
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                title.lineLimit(1)
                Spacer(minLength: 8)
                Button("Back 5 seconds", systemImage: "gobackward.5") {
                    voice.seekPlayback(to: voice.playbackPosition - 5)
                }
                .help("Back 5 seconds")
                Button(voice.playbackPaused ? "Resume" : "Pause", systemImage: voice.playbackPaused ? "play.fill" : "pause.fill") {
                    voice.togglePlaybackPause()
                }
                .help(voice.playbackPaused ? "Resume" : "Pause")
                Button("Forward 5 seconds", systemImage: "goforward.5") {
                    voice.seekPlayback(to: voice.playbackPosition + 5)
                }
                .help("Forward 5 seconds")
                Button("Stop playback", systemImage: "stop.fill") { voice.stopPlayback() }
                    .help("Stop playback")
                Button(voice.playbackLoops ? "Turn loop off" : "Loop recording",
                       systemImage: voice.playbackLoops ? "repeat.circle.fill" : "repeat") {
                    voice.togglePlaybackLoop()
                }
                .help(voice.playbackLoops ? "Turn loop off" : "Loop recording")
                Picker("Playback speed", selection: Binding(get: { voice.playbackRate }, set: voice.setPlaybackRate)) {
                    Text("0.75×").tag(Float(0.75))
                    Text("1×").tag(Float(1))
                    Text("1.25×").tag(Float(1.25))
                }
                .labelsHidden()
                .fixedSize()
            }
            .font(.callout)
            .labelStyle(.iconOnly)

            TimelineView(.periodic(from: .now, by: 0.15)) { _ in
                let duration = max(voice.playbackDuration, 0.01)
                let position = min(voice.playbackPosition, duration)
                HStack(spacing: 9) {
                    Text(Duration.seconds(position).formatted(.time(pattern: .minuteSecond)))
                    Slider(value: Binding(get: { min(voice.playbackPosition, duration) }, set: voice.seekPlayback), in: 0...duration) {
                        Text("Playback position")
                    }
                    .labelsHidden()
                    Text(Duration.seconds(duration).formatted(.time(pattern: .minuteSecond)))
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }
        }
        .padding(9)
        .background(.background, in: RoundedRectangle(cornerRadius: 7))
    }
}
