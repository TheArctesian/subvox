import SwiftUI

struct PracticeHomeView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Subvox")
                        .font(.system(size: 52, weight: .semibold))
                        .accessibilityAddTraits(.isHeader)
                    Text("Choose what you’d like to practise today.")
                        .font(.title3).foregroundStyle(.secondary)
                }

                HStack(spacing: 32) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Reading").font(.title.weight(.semibold)).accessibilityAddTraits(.isHeader)
                        Text("Read short passages aloud and silently with your AirPods.")
                            .foregroundStyle(.secondary)
                        Button("Open Reading", systemImage: TrainMode.reading.symbol) {
                            model.openPractice(.reading)
                        }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .padding(.top, 4)
                    }
                    Spacer()
                    Image(systemName: "waveform")
                        .font(.system(size: 78, weight: .light))
                        .foregroundStyle(Color.accentColor)
                        .accessibilityHidden(true)
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))

                HStack(alignment: .top, spacing: 32) {
                    VStack(alignment: .leading, spacing: 0) {
                        sectionTitle("Speech and poetry")
                        practiceRow(.speech, detail: "Practise a speech from the library")
                        practiceRow(.poetry, detail: "Read a poem in two passes")
                        practiceRow(.free, detail: "Practise with your own text")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    VStack(alignment: .leading, spacing: 0) {
                        sectionTitle("Language and pronunciation")
                        practiceRow(.language, detail: "Practise phrases in your chosen language")
                        practiceRow(.sounds, detail: "Work on individual sounds")
                        practiceRow(.minimalPairs, detail: "Compare similar sounds")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack(spacing: 16) {
                    Button("Devices", systemImage: AppModel.Screen.devices.symbol) { model.screen = .devices }
                    if model.researchTools {
                        Button("Try model", systemImage: AppModel.Screen.decode.symbol) { model.screen = .decode }
                    }
                }
            }
            .frame(maxWidth: 820, alignment: .leading)
            .padding(.horizontal, 36).padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .disabled(model.navigationLocked)
    }

    private func sectionTitle(_ title: LocalizedStringKey) -> some View {
        Text(title).font(.title3.weight(.semibold))
            .accessibilityAddTraits(.isHeader)
            .padding(.bottom, 8)
    }

    private func practiceRow(_ mode: TrainMode, detail: LocalizedStringKey) -> some View {
        VStack(spacing: 0) {
            Divider()
            Button {
                model.openPractice(mode)
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: mode.symbol)
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 24)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(mode.title).font(.headline)
                        Text(detail).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold)).foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
                .padding(.vertical, 12)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}
