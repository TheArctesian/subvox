//! Chimera articulator data for Subvox (dev tool only).
//!
//! `subvox-chimera coverage <ipa>...` prints one JSON line per phone: whether Chimera's male body
//! has a vowel target or an inventory consonant for it. Chimera's tokenizer silently drops symbols
//! it does not know, so every generator built on this tool checks phones one by one first.
//!
//! `subvox-chimera tables <out.py>` writes the male body's calibrated vowel articulations and the
//! consonant targets for jaw and lips as a Python module (ml/subvox_ml/chimera_tables.py).

use chimera::anatomy::human::{ArticKey, ArticPatch, BodyMix};
use chimera::phonetics::{inventory, planner, vowel_data};

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

/// The keys a jaw-coupled earbud could plausibly reflect, in the order the tables store them.
const KEYS: [ArticKey; 4] = [ArticKey::Jaw, ArticKey::LipAp, ArticKey::LipProt, ArticKey::LipWidth];

fn python_value(x: Option<f64>) -> String {
    match x {
        // Chimera's own track rounding (1e-4); -0.0 prints as 0.0 so the file is stable.
        Some(x) => format!("{:?}", (x * 1e4).round() / 1e4 + 0.0),
        None => "None".to_string(),
    }
}

fn row(ipa: &str, patch: &ArticPatch) -> String {
    let values: Vec<String> = KEYS.iter().map(|k| python_value(patch.get(*k))).collect();
    format!("    \"{ipa}\": ({}),\n", values.join(", "))
}

fn tables() -> String {
    let body = BodyMix::MALE;
    let mut out = String::new();
    out.push_str(&format!(
        "\"\"\"Chimera articulator values for the adult male body. Generated; do not edit.\n\n\
Regenerate: cargo run --manifest-path Tools/chimera/Cargo.toml -- tables ml/subvox_ml/chimera_tables.py\n\
Derived from Chimera ({rev}), Copyright (c) 2026 Jameson Hodge, MIT License (see NOTICE).\n\n\
Units are Chimera's template-space adult-male cm: jaw is the incisor opening, lipAp the lip\n\
aperture (negative = compressed), lipProt the protrusion; lipWidth runs 0 rounded to 1 spread.\n\
Vowel values are fitted to reference formants, not measured jaws. A consonant lists only the keys\n\
Chimera sets for it; None means the planner leaves that articulator to coarticulation.\"\"\"\n\n\
CHIMERA_REV = \"{rev}\"\nBODY = \"male\"\nKEYS = (\"jaw\", \"lipAp\", \"lipProt\", \"lipWidth\")\n\nVOWELS = {{\n",
        rev = CHIMERA_REV
    ));
    for v in vowel_data::VOWELS {
        let patch = planner::vowel_artic(v.ipa, body).expect("every reference vowel has a calibrated articulation");
        out.push_str(&row(v.ipa, &patch));
    }
    out.push_str("}\n\nCONSONANTS = {\n");
    for ipa in inventory::consonant_symbols() {
        let def = inventory::phoneme(ipa, body).expect("every listed consonant resolves");
        out.push_str(&row(ipa, &def.targets));
    }
    out.push_str("}\n");
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
        Some((command, rest)) if command == "tables" && rest.len() == 1 => match std::fs::write(&rest[0], tables()) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("cannot write {}: {e}", rest[0]);
                std::process::ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: subvox-chimera coverage <ipa>... | tables <out.py>");
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

    #[test]
    fn tables_are_repeatable_and_keep_unspecified_consonant_keys_empty() {
        chimera::init();
        let first = tables();
        assert_eq!(first, tables());
        assert!(first.contains("    \"p\": (None, -0.25, None, None),\n"));
        assert!(first.contains("    \"i\": ("));
        assert!(!first.contains("-0.0,"));
    }
}
