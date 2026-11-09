import AppKit
import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var model

    /// A sidebar row: one of the Train modes, or another top-level screen.
    enum SidebarItem: Hashable { case train(TrainMode), screen(AppModel.Screen) }

    var body: some View {
        if model.onboarded { main } else { WelcomeView() }
    }

    private var main: some View {
        NavigationSplitView {
            List(selection: selection) {
                Section("Train") {
                    ForEach(TrainMode.allCases) { mode in
                        Label(mode.title, systemImage: mode.symbol).tag(SidebarItem.train(mode))
                    }
                }
                Section {
                    ForEach(model.visibleScreens.filter { $0 != .train }) { screen in
                        Label(LocalizedStringKey(screen.rawValue), systemImage: screen.symbol).tag(SidebarItem.screen(screen))
                    }
                }
            }
            .disabled(model.navigationLocked)
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            switch model.screen {
            case .train: TrainView()
            case .decode: DecodeView()
            case .devices: DevicesView()
            }
        }
        .toolbar {
            ToolbarItem(placement: .status) {
                // The one AirPods status in the app; clicking it opens Devices to fix it.
                Button { model.screen = .devices } label: {
                    MotionStatusBadge(status: model.motion.status, rate: model.motion.sampleRate, text: model.motion.statusText)
                }
                .buttonStyle(.plain).disabled(model.navigationLocked)
                .help(model.navigationLocked ? "Finish or stop the active take to open Devices." : "AirPods and microphone: open Devices")
                .accessibilityLabel(model.motion.status == .streaming ? "AirPods connected" : "AirPods: " + model.motion.statusText)
                .accessibilityHint("Opens Devices")
            }
            ToolbarItem(placement: .primaryAction) { LanguageMenu() }
            if let activity = model.backgroundActivity {
                ToolbarItem(placement: .status) {
                    ProgressView().controlSize(.small)
                        .help(activity).accessibilityLabel(activity)
                }
            }
        }
        // Each page carries its own large heading, so the window title would only repeat it.
        .navigationTitle(model.screen == .train ? model.trainMode.rawValue : model.screen.rawValue)
        .modifier(HiddenWindowTitle())
    }

    private var selection: Binding<SidebarItem?> {
        Binding(get: { model.screen == .train ? .train(model.trainMode) : .screen(model.screen) },
                set: { item in
                    guard !model.navigationLocked else { return }
                    switch item {
                    case .train(let mode)?:
                        if mode != model.trainMode { model.closeOverlays(); model.referenceSpeaker.stop() }
                        model.trainMode = mode; model.screen = .train
                    case .screen(let screen)?: model.screen = screen
                    case nil: break
                    }
                })
    }
}

/// The app-wide practice language, top right of the window.
struct LanguageMenu: View {
    @Environment(AppModel.self) private var model
    @State private var pendingLanguage: String?
    @State private var confirmRestart = false
    @State private var restartFailed = false
    var body: some View {
        @Bindable var model = model
        Menu {
            Picker("I'm practising", selection: $model.language) {
                ForEach(AppLanguage.allCases) { Text($0.name).tag($0) }
            }.pickerStyle(.inline)
            Picker("I speak", selection: $model.nativeLanguage) {
                ForEach(["en", "fr", "es", "zh"], id: \.self) { Text(Self.nativeName($0)).tag($0) }
            }.pickerStyle(.inline)
            Menu("App language") {
                ForEach(Self.interfaceLanguages, id: \.code) { option in
                    Button { pendingLanguage = option.code; confirmRestart = true } label: {
                        if Self.currentInterface == option.code { Label(option.name, systemImage: "checkmark") } else { Text(option.name) }
                    }
                }
            }
        } label: {
            Label("Practising \(model.language.localizedName)", systemImage: "globe")
                .labelStyle(.titleAndIcon)
        }
        .disabled(model.navigationLocked)
        .confirmationDialog("Apply app language and restart?", isPresented: $confirmRestart, titleVisibility: .visible) {
            Button("Apply and restart") {
                guard let pendingLanguage, model.canRestartInterface else { return }
                Self.setInterface(pendingLanguage) { restartFailed = true }
            }.disabled(!model.canRestartInterface)
        } message: {
            Text(model.canRestartInterface ? "Subvox will reopen in the selected language." : "Finish the active work before restarting Subvox.")
        }
        .alert("Subvox couldn’t restart", isPresented: $restartFailed) { Button("OK", role: .cancel) {} }
        .help("The language you're practising (⌥⌘1–6), and the language you speak")
        .accessibilityLabel("Practising \(model.language.localizedName); you speak \(Self.nativeName(model.nativeLanguage))")
    }

    /// The interface language, separate from the practice language. macOS reads it at launch,
    /// so changing it relaunches the app.
    static let interfaceLanguages: [(code: String, name: String)] = [
        ("", String(localized: "Same as the Mac")), ("en", "English"), ("fr", "Français"), ("es", "Español (España)"), ("es-419", "Español (Latinoamérica)"),
        ("zh-Hans", "简体中文"), ("zh-Hant", "繁體中文")]
    static var currentInterface: String { (UserDefaults.standard.array(forKey: "AppleLanguages") as? [String])?.first.flatMap { code in
        interfaceLanguages.map(\.code).filter { !$0.isEmpty && (code == $0 || code.hasPrefix($0 + "-")) }
            .max(by: { $0.count < $1.count }) } ?? "" }
    static func setInterface(_ code: String, onFailure: @escaping @MainActor () -> Void) {
        let defaults = UserDefaults.standard
        let previous = defaults.persistentDomain(forName: Bundle.main.bundleIdentifier ?? "")?["AppleLanguages"]
        defaults.set(code.isEmpty ? nil : [code], forKey: "AppleLanguages")
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: configuration) { app, error in
            DispatchQueue.main.async {
                if app != nil, error == nil { NSApp.terminate(nil) }
                else { defaults.set(previous, forKey: "AppleLanguages"); onFailure() }
            }
        }
    }

    static func nativeName(_ code: String) -> String {
        switch code {
        case "fr": String(localized: "French")
        case "es": String(localized: "Spanish")
        case "zh": String(localized: "Chinese")
        default: String(localized: "English")
        }
    }
}

/// Each page has a large heading, so the toolbar title would only repeat it (macOS 15+; the
/// window keeps its title for the Window menu and Mission Control).
private struct HiddenWindowTitle: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 15.0, *) { content.toolbar(removing: .title) } else { content }
    }
}
