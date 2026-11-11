//! Chimera articulator data for Subvox (dev tool only).
//!
//! `subvox-chimera coverage <ipa>...` prints one JSON line per phone: whether Chimera's male body
//! has a vowel target or an inventory consonant for it. Chimera's tokenizer silently drops symbols
//! it does not know, so every generator built on this tool checks phones one by one first.

use chimera::anatomy::human::BodyMix;
use chimera::phonetics::{inventory, planner};

pub const CHIMERA_REV: &str = "6259e880f5d0108b862e98cefd930aa8c5bc0dc5";

fn kind(ipa: &str, body: BodyMix) -> Option<&'static str> {
    if planner::is_vowel(ipa, body) {
        Some("vowel")
    } else if inventory::phoneme(ipa, body).is_some() {
        Some("consonant")
    } else {
        None
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() -> std::process::ExitCode {
    chimera::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((command, phones)) if command == "coverage" && !phones.is_empty() => {
            for ipa in phones {
                let k = kind(ipa, BodyMix::MALE);
                println!(
                    "{{\"ipa\":{},\"supported\":{},\"kind\":{},\"rev\":\"{CHIMERA_REV}\"}}",
                    json_string(ipa),
                    k.is_some(),
                    k.map_or("null".to_string(), json_string)
                );
            }
            std::process::ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: subvox-chimera coverage <ipa>...");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vowels_and_consonants_are_supported_and_unknown_symbols_are_not() {
        chimera::init();
        assert_eq!(kind("i", BodyMix::MALE), Some("vowel"));
        assert_eq!(kind("y", BodyMix::MALE), Some("vowel"));
        assert_eq!(kind("p", BodyMix::MALE), Some("consonant"));
        assert_eq!(kind("ʂ", BodyMix::MALE), Some("consonant"));
        assert_eq!(kind("ɕ", BodyMix::MALE), None);
    }
}
