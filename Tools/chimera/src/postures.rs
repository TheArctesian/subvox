//! Side-view mouth outlines for the practice sounds (Corpus/practice/articulation-v1.json).
//!
//! Each sound is drawn at Chimera's target posture for the adult male body. Where Chimera has no
//! target, the rule used is written into the output next to the outline; a sound with no honest
//! posture (tones, the alveolo-palatals) is listed under `withoutDiagram` with the reason.

use chimera::anatomy::geom::Vec2;
use chimera::anatomy::human::{
    clamp_to_wall, rest_tip, ArticKey, ArticPatch, HumanArtic, HumanFrame, HumanVocalTract, BODY_MALE, NEUTRAL,
};
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::{inventory, planner};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Tissue compression Chimera's voice allows at a contact before it pushes the tongue back (cm).
const MAX_PENETRATION: f64 = 0.1;
/// Velum port Chimera's planner opens for a nasal consonant (planner.rs, "nasality").
const NASAL_VELUM: f64 = 0.45;
/// Height (cm above the glottis) below which the drawing is cropped: the mouth, nose and upper
/// throat stay in view, the larynx tube does not.
const VIEW_FLOOR: f64 = 3.5;
const TONES: [&str; 5] = ["˥", "˧˥", "˨˩˦", "˥˩", "·"];

#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// Chimera symbol whose target posture is drawn.
    pub base: String,
    /// Chimera vowel whose lip values replace the base's (ɥ: y lips on a j tongue).
    pub lips_from: Option<&'static str>,
    pub velum_open: bool,
    /// How the posture was chosen, when it is not Chimera's own target for this exact symbol.
    pub rule: Option<&'static str>,
    /// What the app tells the learner: "start" (a moving sound, drawn where it begins) or
    /// "approximate" (built from related targets). None when the drawing is the sound's own posture.
    pub note: Option<&'static str>,
}

fn supported(ipa: &str) -> bool {
    planner::is_vowel(ipa, BodyMix::MALE) || inventory::phoneme(ipa, BodyMix::MALE).is_some()
}

/// Which posture represents a practice sound, or why none does.
pub fn spec(ipa: &str) -> Result<Spec, &'static str> {
    if TONES.contains(&ipa) {
        return Err("tone: pitch, not a mouth shape");
    }
    let plain = |base: &str, rule, note| Ok(Spec { base: base.to_string(), lips_from: None, velum_open: false, rule, note });
    match ipa {
        "ɕ" | "tɕ" | "tɕʰ" => return Err("alveolo-palatal: Chimera has no target for it"),
        "ɹ̩" => return Err("Mandarin apical vowel after z c s: Chimera has no target for it"),
        "ɥ" => return Ok(Spec { base: "j".into(), lips_from: Some("y"), velum_open: false, rule: Some("rounded j: y lips on the j tongue"), note: Some("approximate") }),
        "ʈʂ" | "ʈʂʰ" => return plain("ʈ", Some("affricate: its closure, ʈ"), Some("start")),
        "ɡ" => return plain("g", None, None),
        "ɚ" => return plain("ɝ", None, None),
        "aɪ" | "aʊ" | "eɪ" | "oʊ" | "ɔɪ" => {
            let onset = &ipa[..ipa.chars().next().unwrap().len_utf8()];
            return plain(onset, Some("diphthong: where it starts"), Some("start"));
        }
        _ => {}
    }
    if let Some(oral) = ipa.strip_suffix('\u{303}') {
        if planner::is_vowel(oral, BodyMix::MALE) {
            return Ok(Spec { base: oral.into(), lips_from: None, velum_open: true, rule: Some("nasal vowel: the oral vowel with the soft palate lowered"), note: Some("approximate") });
        }
    }
    if let Some(base) = ipa.strip_suffix('ʰ') {
        if supported(base) {
            return plain(base, Some("aspirated: aspiration is breath, not a different mouth shape"), None);
        }
    }
    if let Some(base) = ipa.strip_suffix('\u{329}') {
        if supported(base) {
            return plain(base, Some("syllabic: the consonant's posture"), None);
        }
    }
    if supported(ipa) { plain(ipa, None, None) } else { Err("Chimera has no target for it") }
}

fn vowel(ipa: &str) -> ArticPatch {
    planner::vowel_artic(ipa, BodyMix::MALE).expect("checked vowel")
}

/// The articulator state drawn for a spec: vowels at their calibrated target, consonants over a
/// schwa, with the tongue tip resting behind the teeth wherever the sound does not place it.
pub fn posture(s: &Spec) -> HumanArtic {
    let mut a = if planner::is_vowel(&s.base, BodyMix::MALE) {
        vowel(&s.base).apply_to(&NEUTRAL)
    } else {
        let def = inventory::phoneme(&s.base, BodyMix::MALE).expect("checked consonant");
        let mut a = def.targets.apply_to(&vowel("ə").apply_to(&NEUTRAL));
        if def.targets.get(ArticKey::TtX).is_none() && def.targets.get(ArticKey::TtY).is_none() {
            let tip = rest_tip(a.jaw, a.tb_x, a.tb_y);
            a.tt_x = tip.x;
            a.tt_y = tip.y;
        }
        if def.nasal == Some(true) {
            a.velum = NASAL_VELUM;
        }
        a
    };
    if let Some(v) = s.lips_from {
        let lips = vowel(v);
        for k in [ArticKey::LipAp, ArticKey::LipProt, ArticKey::LipWidth] {
            if let Some(x) = lips.get(k) {
                a.set(k, x);
            }
        }
    }
    if s.velum_open {
        a.velum = NASAL_VELUM;
    }
    a
}

/// The frame at a posture, with the tongue held at the palate the way Chimera's voice holds it
/// (`HumanVoice::enforce_contacts`): stop targets aim past contact, so a tip or dorsum deeper than
/// the allowed compression is moved back along the contact normal until none is.
pub fn settled_frame(tract: &mut HumanVocalTract, mut a: HumanArtic) -> HumanFrame {
    for _ in 0..50 {
        let frame = tract.compute_new(&a);
        let k = tract.scale();
        let lim = -MAX_PENETRATION * k;
        let mut moved = false;
        let mut fix = |x: &mut f64, y: &mut f64, d: f64, n: Vec2| {
            if d < lim {
                let depth = lim - d;
                let sx = (if *x > 0.5 { 1.0 / 0.86 } else { 1.0 }) / k;
                *x += n.x * depth * sx;
                *y += n.y * depth / k;
                moved = true;
            }
        };
        let (ct, ctn, cb, cbn) = (tract.contact_tip, tract.contact_tip_n, tract.contact_body, tract.contact_body_n);
        fix(&mut a.tt_x, &mut a.tt_y, ct, ctn);
        fix(&mut a.tb_x, &mut a.tb_y, cb, cbn);
        if !moved {
            return frame;
        }
    }
    tract.compute_new(&a)
}

/// Outline parts, in drawing order. The tongue is kept off the palate: stop targets aim past
/// contact on purpose (inventory.rs), which the static geometry does not correct.
pub fn parts(frame: &HumanFrame) -> Vec<(&'static str, Vec<Vec2>)> {
    vec![
        ("nasalCavity", frame.nasal_cavity.clone()),
        ("outer", frame.outer.clone()),
        ("inner", frame.inner.clone()),
        ("upperTeeth", frame.upper_teeth.clone()),
        ("lowerTeeth", frame.lower_teeth.clone()),
        ("jaw", frame.jaw_bone.clone()),
        ("velum", frame.velum.clone()),
        ("tongue", clamp_to_wall(&frame.tongue, &frame.outer)),
        ("upperLip", frame.upper_lip.clone()),
        ("lowerLip", frame.lower_lip.clone()),
    ]
}

fn rounded(x: f64) -> Value {
    json!((x * 100.0).round() / 100.0 + 0.0)
}

fn bounds_of(points: impl Iterator<Item = Vec2>, b: &mut [f64; 4]) {
    for p in points.filter(|p| p.y >= VIEW_FLOOR) {
        b[0] = b[0].min(p.x);
        b[1] = b[1].min(p.y);
        b[2] = b[2].max(p.x);
        b[3] = b[3].max(p.y);
    }
}

/// `sounds` maps a practice language to its sound symbols, in corpus order.
pub fn build(sounds: &BTreeMap<String, Vec<String>>, rev: &str) -> Value {
    let mut tract = HumanVocalTract::new(BODY_MALE);
    let mut drawn: BTreeMap<String, (Spec, Vec<(&'static str, Vec<Vec2>)>)> = BTreeMap::new();
    let mut map = Map::new();
    let mut missing = Map::new();
    for (language, list) in sounds {
        for ipa in list {
            let key = format!("{language}:{ipa}");
            match spec(ipa) {
                Ok(s) => {
                    let id = format!("{}{}{}", s.base, if s.velum_open { "+nasal" } else { "" }, s.lips_from.map_or(String::new(), |v| format!("+{v}lips")));
                    drawn.entry(id.clone()).or_insert_with(|| (s.clone(), parts(&settled_frame(&mut tract, posture(&s)))));
                    let mut entry = Map::new();
                    entry.insert("posture".into(), json!(id));
                    if let Some(rule) = s.rule {
                        entry.insert("rule".into(), json!(rule));
                    }
                    if let Some(note) = s.note {
                        entry.insert("note".into(), json!(note));
                    }
                    map.insert(key, Value::Object(entry));
                }
                Err(reason) => {
                    missing.insert(key, json!(reason));
                }
            }
        }
    }
    let mut bounds = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut postures = Map::new();
    for (id, (_, ps)) in &drawn {
        let mut obj = Map::new();
        for (name, pts) in ps {
            bounds_of(pts.iter().copied(), &mut bounds);
            obj.insert((*name).into(), Value::Array(pts.iter().map(|p| json!([rounded(p.x), rounded(p.y)])).collect()));
        }
        postures.insert(id.clone(), Value::Object(obj));
    }
    json!({
        "version": 1,
        "chimeraRev": rev,
        "body": "male",
        "units": "cm, x forward, y up, glottis at the origin",
        "view": bounds.iter().map(|x| rounded(*x)).collect::<Vec<_>>(),
        "parts": parts(&HumanFrame::default()).iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        "sounds": map,
        "withoutDiagram": missing,
        "postures": postures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_cover_aspiration_nasal_vowels_diphthongs_and_refuse_alveolo_palatals() {
        chimera::init();
        assert_eq!(spec("pʰ").unwrap().base, "p");
        let nasal = spec("ɑ̃").unwrap();
        assert!(nasal.velum_open && nasal.base == "ɑ");
        assert_eq!(spec("aɪ").unwrap().base, "a");
        assert_eq!(spec("ɥ").unwrap().lips_from, Some("y"));
        assert_eq!(spec("ɻ̩").unwrap().base, "ɻ");
        assert!(spec("tɕʰ").is_err() && spec("˥").is_err() && spec("ɹ̩").is_err());
        assert_eq!(spec("i").unwrap(), Spec { base: "i".into(), lips_from: None, velum_open: false, rule: None, note: None });
    }

    #[test]
    fn nasals_lower_the_velum_and_bilabials_close_the_lips() {
        chimera::init();
        let m = posture(&spec("m").unwrap());
        assert_eq!(m.velum, NASAL_VELUM);
        assert!(m.lip_ap < 0.0);
        assert!(posture(&spec("b").unwrap()).velum < 0.0);
    }

    #[test]
    fn settled_tongue_stays_within_the_allowed_compression_for_every_stop() {
        chimera::init();
        let mut tract = HumanVocalTract::new(BODY_MALE);
        for ipa in ["k", "g", "t", "d", "ʈ", "c", "q"] {
            settled_frame(&mut tract, posture(&spec(ipa).unwrap()));
            let lim = -MAX_PENETRATION * tract.scale() - 1e-9;
            assert!(tract.contact_tip >= lim && tract.contact_body >= lim, "{ipa}");
        }
    }

    #[test]
    fn output_is_repeatable_and_shares_one_posture_across_languages() {
        chimera::init();
        let sounds: BTreeMap<String, Vec<String>> =
            [("en".to_string(), vec!["p".into(), "aɪ".into()]), ("fr".to_string(), vec!["p".into(), "ɑ̃".into(), "˥".into()])].into();
        let out = build(&sounds, "rev");
        assert_eq!(out, build(&sounds, "rev"));
        assert_eq!(out["sounds"]["en:p"]["posture"], "p");
        assert_eq!(out["sounds"]["fr:p"]["posture"], "p");
        assert_eq!(out["postures"].as_object().unwrap().len(), 3);
        assert!(out["withoutDiagram"]["fr:˥"].is_string());
    }
}
