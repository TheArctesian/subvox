import SwiftUI

@main
struct SubvoxApp: App {
    @State private var model = AppModel()
    static var isTestHost: Bool { ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .frame(minWidth: 980, minHeight: 680)
                // As a unit-test host the app never touches sensors: no Motion/Camera/Microphone
                // prompts, and no disturbing the permission of a copy the user is running.
                .task { if !Self.isTestHost { model.start() } }
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("Practice") {
                Button("Home", action: model.openHome)
                    .keyboardShortcut("0", modifiers: .command)
                    .disabled(model.navigationLocked)
                Button("Start") {
                    if model.screen == .train && model.trainMode == .reading { model.startSentenceTraining() } else { model.startRequest += 1 }
                }.keyboardShortcut(.return, modifiers: .command)
                    .disabled(model.screen != .train || model.trainMode == .memorization || !model.sentenceTraining.canStartRecording)
                Button("Show or Hide IPA") {
                    let defaults = UserDefaults.standard
                    defaults.set(!(defaults.object(forKey: "practice.showIPA") as? Bool ?? true), forKey: "practice.showIPA")
                }.keyboardShortcut("i", modifiers: .command)
                Button(UserDefaults.standard.object(forKey: "ui.soundCues") as? Bool ?? true ? "Turn Off Sound Cues" : "Turn On Sound Cues") {
                    let defaults = UserDefaults.standard
                    defaults.set(!(defaults.object(forKey: "ui.soundCues") as? Bool ?? true), forKey: "ui.soundCues")
                }
                Menu("Practise Language") {
                    ForEach(Array(AppLanguage.allCases.enumerated()), id: \.offset) { index, language in
                        Button(language.name) { model.language = language }
                            .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: [.command, .option])
                            .disabled(model.navigationLocked)
                    }
                }
                Divider()
                ForEach(TrainMode.allCases) { mode in
                    Button(mode.title) { model.openPractice(mode) }
                        .keyboardShortcut(KeyEquivalent(mode.shortcut), modifiers: .command)
                        .disabled(model.navigationLocked)
                }
            }
        }
        Settings {
            ContributeView()
                .environment(model)
                .frame(minWidth: 650, idealWidth: 740, minHeight: 520, idealHeight: 660)
        }
        .windowResizability(.contentMinSize)
    }
}
