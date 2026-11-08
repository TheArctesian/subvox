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
                        Label(mode.rawValue, systemImage: mode.symbol).tag(SidebarItem.train(mode))
                    }
                }
                Section {
                    ForEach(model.visibleScreens.filter { $0 != .train }) { screen in
                        Label(screen.rawValue, systemImage: screen.symbol).tag(SidebarItem.screen(screen))
                    }
                }
            }
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
                MotionStatusBadge(status: model.motion.status, rate: model.motion.sampleRate, text: model.motion.statusText)
            }
        }
        .navigationTitle(model.screen == .train ? model.trainMode.rawValue : model.screen.rawValue)
    }

    private var selection: Binding<SidebarItem?> {
        Binding(get: { model.screen == .train ? .train(model.trainMode) : .screen(model.screen) },
                set: { item in
                    switch item {
                    case .train(let mode)?: model.trainMode = mode; model.screen = .train
                    case .screen(let screen)?: model.screen = screen
                    case nil: break
                    }
                })
    }
}
