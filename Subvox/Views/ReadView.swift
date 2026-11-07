import SwiftUI

/// Teleprompter for the reading corpus. Reading the same lines aloud, whispered and silently
/// gives three aligned takes of every diphone; the voiced ones carry the speech timing.
struct ReadView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let reading = model.reading
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                LiveMotionPanel()
                switch reading.phase {
                case .setup:
                    SetupPanel()
                case .reading:
                    TeleprompterPanel()
                case .processing(let done, let total):
                    GroupBox {
                        VStack(spacing: 10) {
                            ProgressView(value: Double(done), total: Double(max(total, 1)))
                            Text("Aligning line \(min(done + 1, total)) of \(total)…")
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                    }
                case .finished(let summary):
                    SummaryPanel(summary: summary)
                case .failed(let message):
                    GroupBox {
                        VStack(alignment: .leading, spacing: 10) {
                            Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                            Button("Back", action: reading.reset)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                    }
                }
            }
            .padding(20)
        }
    }
}

private struct SetupPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var reading = model.reading
        @Bindable var recorder = model.recorder
        let total = reading.script.lines.count
        let planned = reading.plannedLines
        let words = planned.reduce(0) { $0 + Lexicon.words(in: $1).count }

        return VStack(alignment: .leading, spacing: 16) {
            Text("Read the corpus line by line while Subvox records motion and audio. When you stop, every word, phone and jaw movement is labelled on the motion timeline. Read the same lines once aloud and once silently to get matching takes.")
                .foregroundStyle(.secondary)

            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Text", selection: $reading.script) {
                        ForEach(ReadingScript.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 360)

                    if reading.script == .bank {
                        HStack(spacing: 20) {
                            Stepper("From line \(reading.firstLine + 1)", value: $reading.firstLine,
                                    in: 0...max(total - 1, 0), step: 10)
                            Stepper("\(reading.lineCount) lines", value: $reading.lineCount, in: 10...total, step: 10)
                        }
                        .monospacedDigit()
                    }

                    Picker("Condition", selection: $recorder.condition) {
                        ForEach(SessionRecorder.Condition.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 420)

                    Text("\(planned.count) lines, \(words) words, about \(max(1, words / 110)) min.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    if recorder.condition == .silent {
                        Label("Silent: no audio, so word timing is estimated from the motion in each line.",
                              systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if let error = model.audio.error {
                        Label(error, systemImage: "mic.slash").font(.caption).foregroundStyle(.red)
                    }

                    HStack {
                        Button("Start reading", systemImage: "play.fill") {
                            Task { await model.reading.start(recorder: model.recorder, audio: model.audio) }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(planned.isEmpty || model.motion.status != .streaming || model.recorder.isRecording)
                        if model.motion.status != .streaming {
                            Label(model.motion.status.label, systemImage: "airpodspro")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        } else if model.recorder.isRecording {
                            Text("Stop the current recording first.").font(.caption).foregroundStyle(.orange)
                        }
                    }
                }
                .padding(6)
            }
        }
    }
}

private struct TeleprompterPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let reading = model.reading
        let lines = reading.lines
        let i = reading.position

        return GroupBox {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Label(model.recorder.condition.rawValue, systemImage: "record.circle")
                        .foregroundStyle(.red)
                    Spacer()
                    Text("Line \(i + 1) of \(lines.count)").monospacedDigit()
                }
                .font(.callout)
                ProgressView(value: Double(i), total: Double(max(lines.count, 1)))

                Text(i > 0 ? lines[i - 1] : " ")
                    .font(.title3)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
                Text(reading.current ?? "")
                    .font(.system(size: 34, weight: .medium, design: .serif))
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, minHeight: 130, alignment: .topLeading)
                    .id(i)
                    .transition(.opacity)
                Text(i + 1 < lines.count ? lines[i + 1] : "End")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                HStack {
                    Button("Re-read previous", systemImage: "arrow.left", action: reading.back)
                        .keyboardShortcut(.leftArrow, modifiers: [])
                        .disabled(i == 0)
                    Spacer()
                    Button("Stop", action: reading.finish)
                        .keyboardShortcut(.cancelAction)
                    Button("Next line", systemImage: "arrow.right", action: reading.next)
                        .keyboardShortcut(.space, modifiers: [])
                        .buttonStyle(.borderedProminent)
                }
                Text("Read the line at your normal pace, then press Space. Pause briefly between lines and keep your head still. Esc stops early and keeps what you've read.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .animation(.easeOut(duration: 0.15), value: i)
            .padding(10)
        }
    }
}

private struct SummaryPanel: View {
    @Environment(AppModel.self) private var model
    var summary: ReadingController.Summary

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Label("Labelled \(summary.lines) lines", systemImage: "checkmark.circle.fill")
                    .font(.headline)
                    .foregroundStyle(.green)
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 6) {
                    GridRow {
                        Text("Words").foregroundStyle(.secondary)
                        Text("\(summary.words) (\(summary.timedFromSpeech) timed from speech, \(summary.words - summary.timedFromSpeech) estimated)")
                    }
                    GridRow {
                        Text("Phones").foregroundStyle(.secondary)
                        Text("\(summary.phones)")
                    }
                    GridRow {
                        Text("Jaw classes").foregroundStyle(.secondary)
                        Text(summary.jawCounts.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }
                            .joined(separator: " · "))
                    }
                }
                .monospacedDigit()
                Text("Word timings may come from speech; all phone and jaw boundaries are estimates. These exports are not yet used by Train or Decode. labels.csv sits next to motion.csv and audio.caf on the same clock.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Show in Finder", systemImage: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([summary.directory.appending(path: "labels.csv")])
                    }
                    Button("Read again", action: model.reading.reset)
                        .buttonStyle(.borderedProminent)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}
