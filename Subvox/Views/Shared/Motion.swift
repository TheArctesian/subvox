import AppKit
import SwiftUI

/// The app's motion roles. Every authored animation goes through one of these, so durations stay
/// consistent and Reduce Motion is honoured in one place (SwiftUI does not drop explicit
/// animations on its own). Readings, timers, progress and the pacing highlight never animate.
enum Motion: CaseIterable {
    /// Colour, label, symbol and opacity changes in place.
    case state
    /// Anything leaving.
    case exit
    /// Disclosure and IPA reveal.
    case disclosure
    /// A card or row entering the page flow.
    case arrive
    /// Selection highlight and list reorder.
    case reorder
    /// Live level smoothing.
    case meter

    var base: Animation {
        switch self {
        case .state: .easeOut(duration: 0.12)
        case .exit: .easeIn(duration: 0.10)
        case .disclosure: .easeOut(duration: 0.18)
        case .arrive: .spring(response: 0.35, dampingFraction: 1)
        case .reorder: .spring(response: 0.30, dampingFraction: 1)
        case .meter: .linear(duration: 0.08)
        }
    }

    func animation(reduceMotion: Bool) -> Animation? { reduceMotion ? nil : base }

    /// For event handlers outside a view's environment.
    static var reduceMotion: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }
}

/// Runs a state change under a motion role, or instantly with Reduce Motion on.
@discardableResult
func withMotion<Result>(_ role: Motion, _ body: () throws -> Result) rethrows -> Result {
    try withAnimation(role.animation(reduceMotion: Motion.reduceMotion), body)
}

private struct MotionModifier<Value: Equatable>: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var role: Motion
    var value: Value
    func body(content: Content) -> some View {
        content.animation(role.animation(reduceMotion: reduceMotion), value: value)
    }
}

extension View {
    /// Animates changes driven by `value` with a motion role; instant under Reduce Motion.
    func motion(_ role: Motion, value: some Equatable) -> some View {
        modifier(MotionModifier(role: role, value: value))
    }
}

/// Insert and removal transitions built from the roles above.
extension AnyTransition {
    /// Fades in on `state`, out on `exit`.
    static var stateFade: AnyTransition {
        .asymmetric(insertion: .opacity.animation(Motion.state.animation(reduceMotion: Motion.reduceMotion)),
                    removal: .opacity.animation(Motion.exit.animation(reduceMotion: Motion.reduceMotion)))
    }
}

extension AnyTransition {
    /// For rows entering a list that is reordering: the new row waits for the moving rows to
    /// pass before fading in, so their text never overlaps mid-flight.
    static var fadeInAfterReorder: AnyTransition {
        let reduce = Motion.reduceMotion
        return .asymmetric(insertion: .opacity.animation(Motion.state.animation(reduceMotion: reduce)?.delay(0.2)),
                           removal: .opacity.animation(Motion.exit.animation(reduceMotion: reduce)))
    }
}

/// One phase palette for every capture screen, so a colour means the same thing everywhere.
enum PhaseColor {
    static func relax(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 1, green: 0.74, blue: 0.35) : Color(red: 0.62, green: 0.39, blue: 0.08)
    }
    static func speak(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.34, green: 0.83, blue: 0.52) : Color(red: 0.10, green: 0.46, blue: 0.30)
    }
    static func mouth(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.45, green: 0.67, blue: 1) : Color(red: 0.19, green: 0.37, blue: 0.78)
    }
}

/// A disclosure whose chevron and body animate on the `disclosure` role (instant under Reduce Motion).
struct MotionDisclosure<Content: View>: View {
    private let title: Text
    @ViewBuilder private let content: () -> Content
    @State private var expanded = false

    init(_ title: LocalizedStringKey, @ViewBuilder content: @escaping () -> Content) {
        self.title = Text(title); self.content = content
    }
    @_disfavoredOverload
    init<S: StringProtocol>(_ title: S, @ViewBuilder content: @escaping () -> Content) {
        self.title = Text(title); self.content = content
    }

    var body: some View {
        DisclosureGroup(isExpanded: Binding(get: { expanded }, set: { value in withMotion(.disclosure) { expanded = value } }),
                        content: content, label: { title })
    }
}

extension Binding where Value == Bool {
    /// The same binding, with writes animated on a motion role.
    func motion(_ role: Motion) -> Binding<Bool> {
        Binding(get: { wrappedValue }, set: { value in withMotion(role) { wrappedValue = value } })
    }
}
