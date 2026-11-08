# macOS UI checklist for the Train modes

Researched 7 Nov 2026, mainly from the Apple Human Interface Guidelines: [sidebars](https://developer.apple.com/design/human-interface-guidelines/sidebars), [tab views](https://developer.apple.com/design/human-interface-guidelines/tab-views), [segmented controls](https://developer.apple.com/design/human-interface-guidelines/segmented-controls), [toolbars](https://developer.apple.com/design/human-interface-guidelines/toolbars), [settings](https://developer.apple.com/design/human-interface-guidelines/settings), [onboarding](https://developer.apple.com/design/human-interface-guidelines/onboarding), [accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility), [feedback](https://developer.apple.com/design/human-interface-guidelines/feedback) and [keyboards](https://developer.apple.com/design/human-interface-guidelines/keyboards). The practice-app patterns come from secondary sources only, mostly reviews and case studies of ELSA Speak and Duolingo. Items marked *(judgement)* are our own synthesis, not something a source states.

## Navigation
- [x] List the five practice modes in the sidebar under a **Train** section, above Try model and Devices. The HIG allows either a sidebar or a tab view for closely related modes; the sidebar keeps all five visible, and on macOS 15 and later a tab view's tabs move into the toolbar, where a narrow window hides them.
- [x] Don't use a segmented control as the main mode switcher. Keep it for switching within a page, such as Record / Review / Progress or a language picker.
- [x] Give the modes short title-case noun labels and SF Symbols: Reading, Speech & Poetry, Language, Minimal Pairs, Free Practice.
- [x] Add a menu command for every toolbar or sidebar action. Switch modes with ⌘1 to ⌘5 from a **Practice** menu.
- [ ] Don't tint toolbars or add custom backgrounds, because they fight the system materials. (The existing toolbar already follows this.)

## The practice loop
- [x] Use the same loop in every mode: show the text, speak it, mouth it, get feedback, move on. Only the content and the scoring change. *(judgement)*
- [x] Keep rounds short and fixed, about 5 to 10 items, with visible progress ("3 of 8") and a summary at the end.
- [x] Give feedback right after processing. Show the target beside what was heard, per word and per sound (expected and produced IPA), not just pass or fail.
- [x] Make retrying cheap: Redo during a take, plus Retry and Next afterwards.
- [ ] Add side-by-side playback of the user's take and the reference audio. Not built yet: there is no native reference audio, so this needs TTS or recordings first.
- [x] Use no streaks. Show progress as items and sounds practised.

## Permissions and onboarding
- [x] Ask for microphone and camera access in context, at the first Record press. Show what the app is for and offer defaults so the user can start without any setup.
- [x] Keep onboarding optional and skippable, never show it again once skipped, and make it findable later.
- [x] Explain why an action is unavailable instead of silently disabling it: "Connect your AirPods to start", "Choose a local Whisper model".

## Accessibility
- [x] Give every control a VoiceOver label. Record buttons have state-aware labels.
- [x] Announce state changes that don't move focus, such as round results ("2 of 3 sounds matched"), with `AccessibilityNotification.Announcement`.
- [x] Never show right or wrong with colour alone. Pair a symbol (checkmark, xmark, triangle) and text with any colour, and respect Differentiate Without Colour.
- [x] Use system colours and semantic fonts so Increase Contrast and larger text work. Show the practice text at `.title2` or larger.
- [x] Respect Reduce Motion. The word highlight changes instantly instead of animating when it is on.
- [x] Make everything work with Full Keyboard Access using standard controls. The core loop has shortcuts: Return for Done, ⌘R for Redo, and ⌘1 to ⌘5 for the modes.
- [x] Use inline feedback, not alerts, for a bad take or a low score.
- [x] Keep hit targets at 28 × 28 pt (the default control size) or larger.
