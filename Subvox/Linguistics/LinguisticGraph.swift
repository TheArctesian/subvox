import Foundation

struct LinguisticWord: Codable, Sendable {
    var word: String
    var alternatives: [PronunciationVariant]
}
struct LinguisticDiphone: Codable, Hashable, Sendable {
    var left: String
    var right: String
    var crossesWord: Bool
    var key: String { left + " → " + right }
}

/// Alternative pronunciations form a lattice. Unknown words break diphone chains, matching
/// Tools/corpus/coverage.py. Coverage is fractional annotation expectation, never recognition accuracy.
struct LinguisticGraph: Codable, Sendable {
    var words: [LinguisticWord]
    init(text: String, profile: PronunciationProfile, lexicon: Lexicon) {
        words = Lexicon.words(in: text).map { .init(word: $0, alternatives: profile.alternatives($0, lexicon: lexicon)) }
    }
    var complete: Bool { !words.isEmpty && words.allSatisfy { !$0.alternatives.isEmpty } }
    var phonemeCoverage: [String: Double] {
        var counts: [String: Double] = [:]
        for word in words {
            for variant in word.alternatives {
                for phone in variant.ipa { counts[phone, default: 0] += 1 / Double(word.alternatives.count) }
            }
        }
        return counts
    }
    var diphoneCoverage: [LinguisticDiphone: Double] {
        var counts: [LinguisticDiphone: Double] = [:]
        var previous: [PronunciationVariant] = []
        for word in words {
            for variant in word.alternatives {
                for (a,b) in zip(variant.ipa, variant.ipa.dropFirst()) {
                    counts[.init(left:a,right:b,crossesWord:false), default:0] += 1 / Double(word.alternatives.count)
                }
                for before in previous {
                    if let a = before.ipa.last, let b = variant.ipa.first {
                        counts[.init(left:a,right:b,crossesWord:true),default:0] += 1 / Double(previous.count * word.alternatives.count)
                    }
                }
            }
            previous = word.alternatives
        }
        return counts
    }

    /// Weak sentence-level targets: expected gesture counts plus expected gesture transitions.
    /// No token is assigned a time or a distinct sensor class. /p b m/ have identical priors.
    /// The final coordinate is unsupported-articulation mass, not a claim about tongue sensing.
    var gestureSignature: [Double]? {
        guard complete else { return nil }
        var single = Array(repeating:0.0,count:8), transition = Array(repeating:0.0,count:64)
        let phones = phonemeCoverage, pairs = diphoneCoverage
        var internalTransitions=0.0
        func addTransition(_ a:[Double],_ b:[Double],_ count:Double) {
            for i in 0..<8 { for j in 0..<8 { transition[i*8+j] += count*a[i]*b[j] } }
        }
        for phone in phones.keys.sorted() {
            let count=phones[phone]!
            let path=ArticulatoryInventory.gesturePath(phone)
            guard !path.isEmpty else { return nil }
            for g in path {for k in 0..<8 {single[k] += count*g[k]/Double(path.count)}}
            for (a,b) in zip(path,path.dropFirst()) {addTransition(a,b,count);internalTransitions+=count}
        }
        let orderedPairs=pairs.keys.sorted {a,b in a.key==b.key ? !a.crossesWord && b.crossesWord:a.key<b.key}
        for pair in orderedPairs {
            let count=pairs[pair]!
            guard let a = ArticulatoryInventory.gesturePath(pair.left).last,
                  let b = ArticulatoryInventory.gesturePath(pair.right).first else { return nil }
            addTransition(a,b,count)
        }
        let n = max(1,phones.keys.sorted().reduce(0) {$0+phones[$1]!}), m = max(1,orderedPairs.reduce(0) {$0+pairs[$1]!}+internalTransitions)
        return single.map { $0/n } + transition.map { $0/m }
    }
    var hasUnconfirmedVariants: Bool { words.contains { $0.alternatives.contains { !$0.speakerConfirmed } } }
}

enum PhoneTimingMethod: String, Codable, Sendable {
    case localAcousticAlignment, cameraWarpFromAcoustic, manualAnnotation
}
struct PhoneBoundary: Codable, Sendable {
    var ipa: String
    var start: Double
    var end: Double
    /// Nil means unavailable. Backend likelihoods must not be relabelled as probabilities.
    var confidence: Double?
    var backendScore: Double?
    var word: String?
    var wordIndex: Int?
    var pronunciationVariantIDs: [String]?
}
struct PhoneAlignmentProvenance: Codable, Sendable {
    var method: PhoneTimingMethod
    var backend: String
    var modelSHA256: String?
    var pronunciationSHA256: String
    var audioSHA256: String?
    var rawOutputPath: String?
    var sourceAlignmentID: UUID?
    var warpCost: Double?
    var note: String
    var phoneMapSHA256: String?
}
struct PhoneAlignment: Codable, Identifiable, Sendable {
    var id = UUID()
    var condition: SpeechCondition
    var phones: [PhoneBoundary]
    var provenance: PhoneAlignmentProvenance
    var reviewed = false
    // Additional labels in subvox_ml.ipa.INVENTORY. Boundary validation
    // needs the label set, independent of the English gesture priors.
    private static let additionalIPA = Set("ɲ ʁ ɥ y ø œ ɛ̃ ɑ̃ ɔ̃ œ̃ β x ɣ ʎ ʝ pʰ tʰ kʰ tɕ tɕʰ ɕ ʈʂ ʈʂʰ ʂ ʐ ts tsʰ ɤ ɹ̩ ɻ̩ ɻ".split(separator: " ").map(String.init))

    func validate(in range: ClosedRange<Double>) throws {
        guard range.lowerBound.isFinite,range.upperBound.isFinite,
              provenance.method != .localAcousticAlignment || condition == .normal,
              provenance.method != .cameraWarpFromAcoustic || (condition == .silent && provenance.sourceAlignmentID != nil && provenance.warpCost?.isFinite == true),
              !phones.isEmpty, phones.count <= 1000,
              phones.allSatisfy({ (ArticulatoryInventory.byIPA[$0.ipa] != nil || Self.additionalIPA.contains($0.ipa)) && $0.start.isFinite && $0.end.isFinite &&
                  $0.end > $0.start && $0.start >= range.lowerBound && $0.end <= range.upperBound &&
                  ($0.confidence.map { $0.isFinite && (0...1).contains($0) } ?? true) && ($0.backendScore?.isFinite ?? true) }),
              zip(phones,phones.dropFirst()).allSatisfy({ $1.start >= $0.end - 1e-7 }) else { throw LinguisticFailure.alignment }
    }
}

/// Camera observations may support overlapping distributions for different IPA labels.
/// Never construct a one-to-one phoneme→viseme taxonomy from the linguistic inventory.
struct SensorOverlap: Codable, Sendable {
    var first: String
    var second: String
    var standardisedDistance: Double
    var firstExamples: Int
    var secondExamples: Int
    var interpretation: String
    var modality: String = "camera"
}
enum SensorObservability {
    struct Example {
        var ipa: String
        var movement: [Double]
        var sessionID: UUID
        var sentenceGroup: String
        var modality: String = "camera"
    }
    static func learn(_ examples: [Example]) -> [SensorOverlap] {
        if Set(examples.map(\.modality)).count>1 {
            return Dictionary(grouping:examples,by:\.modality).keys.sorted().flatMap {key in learn(examples.filter {$0.modality==key})}
        }
        let valid = examples.filter { !$0.movement.isEmpty && $0.movement.allSatisfy(\.isFinite) }
        let groups = Dictionary(grouping:valid,by:\.ipa)
        let keys = groups.keys.sorted()
        var result: [SensorOverlap] = []
        for (i,a) in keys.enumerated() { for b in keys.dropFirst(i+1) {
            let x = groups[a]!, y = groups[b]!
            // Avoid declaring equivalence from many frames of one take.
            guard x.count >= 3, y.count >= 3, Set(x.map(\.sentenceGroup)).count >= 2,
                  Set(y.map(\.sentenceGroup)).count >= 2, let dims = x.first?.movement.count,
                  (x+y).allSatisfy({ $0.movement.count == dims }) else { continue }
            func mean(_ rows:[Example],_ k:Int)->Double { rows.reduce(0) { $0+$1.movement[k] }/Double(rows.count) }
            var distance = 0.0
            for k in 0..<dims {
                let u=mean(x,k),v=mean(y,k)
                let variance=(x.reduce(0) {$0+pow($1.movement[k]-u,2)}+y.reduce(0) {$0+pow($1.movement[k]-v,2)})/Double(x.count+y.count)
                distance += pow(u-v,2)/max(1e-4,variance)
            }
            let d=sqrt(distance/Double(dims))
            result.append(.init(first:a,second:b,standardisedDistance:d,firstExamples:x.count,secondExamples:y.count,
                interpretation:d < 1 ? "Overlapping observed movement; retain both IPA hypotheses" : "Different sample means; separation is unverified until held-out evaluation",modality:x[0].modality))
        } }
        return result.sorted { $0.standardisedDistance < $1.standardisedDistance }
    }

    /// Reviewed acoustic/visual estimates only. One pooled observation per phone occurrence,
    /// with source/session identity retained. Duration-spread legacy annotations cannot enter.
    static func examples(_ take:RepresentationTake)->[Example] {
        guard take.discardedReason==nil,let alignment=take.phoneAlignment,alignment.reviewed,
              alignment.condition==take.recording.condition,
              [.localAcousticAlignment,.cameraWarpFromAcoustic,.manualAnnotation].contains(alignment.provenance.method),
              (try? alignment.validate(in:take.recording.cueStart...take.recording.cueEnd)) != nil,
              MotionSourceSafety.source(take.recording) != nil else {return []}
        return alignment.phones.flatMap {phone -> [Example] in
            let points=take.camera.filter {$0.t>=phone.start && $0.t<=phone.end && ($0.mouth?.confidence ?? 0)>=0.5}
                .compactMap(\.mouth)
            var examples:[Example]=[]
            if points.count>=3 {
                let n=Double(points.count)
                let movement=[points.reduce(0){$0+$1.opening}/n,points.reduce(0){$0+$1.width}/n,points.reduce(0){$0+$1.outerAspect}/n]
                examples.append(.init(ipa:phone.ipa,movement:movement,sessionID:take.sessionID,sentenceGroup:take.sentenceID))
            }
            let motion=take.recording.samples.filter {$0.t>=phone.start && $0.t<=phone.end}.compactMap {TeacherIMUFeatures.base($0,mode:.invariant)}
            if motion.count>=3 {
                let mean=(0..<7).map {k in motion.reduce(0){$0+$1[k]}/Double(motion.count)}
                examples.append(.init(ipa:phone.ipa,movement:mean,sessionID:take.sessionID,sentenceGroup:take.sentenceID,modality:"AirPods invariant"))
            }
            return examples
        }
    }
}
