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
                ForEach(TrainMode.allCases) { mode in
                    Button(mode.rawValue) { model.screen = .train; model.trainMode = mode }
                        .keyboardShortcut(KeyEquivalent(mode.shortcut), modifiers: .command)
                }
            }
        }
    }
}
