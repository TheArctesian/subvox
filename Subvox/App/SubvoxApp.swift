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
                Button("Start") {
                    if model.screen == .train && model.trainMode == .reading { model.startSentenceTraining() } else { model.startRequest += 1 }
                }.keyboardShortcut(.return, modifiers: .command)
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
                    }
                }
                Divider()
                ForEach(TrainMode.allCases) { mode in
                    Button(mode.title) { if mode != model.trainMode { model.closeOverlays() }; model.screen = .train; model.trainMode = mode }
                        .keyboardShortcut(KeyEquivalent(mode.shortcut), modifiers: .command)
                }
            }
        }
    }
}
