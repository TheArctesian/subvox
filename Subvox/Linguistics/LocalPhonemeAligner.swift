import CryptoKit
import Foundation
import Observation

/// Optional MFA 3.x local-file adapter. Whisper word attention is not phone alignment.
/// Models must already exist locally; names/URLs are never passed to a download command.
@MainActor @Observable
final class LocalPhonemeAligner {
    var executablePath: String { didSet { defaults.set(executablePath,forKey:"PhoneAligner.executable") } }
    var modelPath: String { didSet { defaults.set(modelPath,forKey:"PhoneAligner.model") } }
    /// Acoustic model phone labels must match this explicit mapping. No guessed remapping.
    var phoneMapPath: String { didSet { defaults.set(phoneMapPath,forKey:"PhoneAligner.phoneMap") } }
    private let defaults: UserDefaults
    init(defaults: UserDefaults = .standard) {
        self.defaults=defaults
        executablePath=defaults.string(forKey:"PhoneAligner.executable") ?? ""
        modelPath=defaults.string(forKey:"PhoneAligner.model") ?? ""
        phoneMapPath=defaults.string(forKey:"PhoneAligner.phoneMap") ?? ""
    }
    var isConfigured: Bool {
        FileManager.default.isExecutableFile(atPath:executablePath) &&
        FileManager.default.isReadableFile(atPath:modelPath) &&
        FileManager.default.isReadableFile(atPath:phoneMapPath)
    }
    struct Configuration: Sendable { let executable:String; let model:String; let phoneMap:String }
    var configuration: Configuration { .init(executable:executablePath,model:modelPath,phoneMap:phoneMapPath) }

    /// JSON map: {"AA1":"ɑ", "P":"p", ...}. Distinct model stress symbols remain intact.
    nonisolated static func dictionary(text:String,profile:PronunciationProfile,lexicon:Lexicon,map:[String:String]) throws -> String {
        let words=Array(Set(Lexicon.words(in:text))).sorted()
        var lines:[String]=[]
        for word in words {
            let variants=profile.alternatives(word,lexicon:lexicon)
            guard !variants.isEmpty else { throw LinguisticFailure.pronunciation }
            for variant in variants {
                var labels:[String]=[]
                for (i,ipa) in variant.ipa.enumerated() {
                    // Prefer the original stress-bearing symbol only if the user mapping agrees.
                    let original=variant.arpabet.flatMap { i < $0.count ? $0[i]:nil }
                    if let original,map[original]==ipa { labels.append(original) }
                    else if let key=map.keys.sorted().first(where:{map[$0]==ipa}) { labels.append(key) }
                    else { throw LinguisticFailure.pronunciation }
                }
                lines.append(word+"\t"+labels.joined(separator:" "))
            }
        }
        return Array(Set(lines)).sorted().joined(separator:"\n")+"\n"
    }

    nonisolated static func align(configuration:Configuration,audio:URL,range:ClosedRange<Double>,sessionAudioOrigin:Double,
                                  text:String,profile:PronunciationProfile,lexicon:Lexicon,output:URL) async throws -> PhoneAlignment {
        guard range.lowerBound>=0,range.upperBound>range.lowerBound,sessionAudioOrigin.isFinite,
              FileManager.default.isExecutableFile(atPath:configuration.executable),
              FileManager.default.fileExists(atPath:configuration.model),!configuration.model.contains("://") else { throw LinguisticFailure.alignment }
        let map=try JSONDecoder().decode([String:String].self,from:Data(contentsOf:URL(fileURLWithPath:configuration.phoneMap)))
        let dir=output.appending(path:"phone-alignment-\(UUID())")
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
        let wav=dir.appending(path:"speech.wav"),lab=dir.appending(path:"speech.lab"),dict=dir.appending(path:"speaker.dict"),result=dir.appending(path:"speech.json")
        try LocalWhisper.convertToWAV(audio,destination:wav,timeRange:range)
        try text.write(to:lab,atomically:true,encoding:.utf8)
        try dictionary(text:text,profile:profile,lexicon:lexicon,map:map).write(to:dict,atomically:true,encoding:.utf8)
        let child=Process(),log=dir.appending(path:"mfa.log")
        FileManager.default.createFile(atPath:log.path,contents:nil)
        let handle=try FileHandle(forWritingTo:log);defer { try? handle.close() }
        child.executableURL=URL(fileURLWithPath:configuration.executable)
        child.arguments=["align_one",wav.path,lab.path,dict.path,configuration.model,result.path,
                         "--output_format","json","--temporary_directory",dir.appending(path:"temporary").path,"--num_jobs","1"]
        child.standardOutput=handle;child.standardError=handle
        try child.run()
        defer { if child.isRunning { child.terminate() } }
        let deadline=ContinuousClock.now.advanced(by:.seconds(180))
        while child.isRunning {
            try Task.checkCancellation()
            guard ContinuousClock.now<deadline else { throw LinguisticFailure.alignment }
            try await Task.sleep(for:.milliseconds(100))
        }
        guard child.terminationStatus==0 else { throw LinguisticFailure.alignment }
        var provenance=PhoneAlignmentProvenance(method:.localAcousticAlignment,backend:"MFA 3.x align_one; local acoustic model; uncalibrated confidence",
            modelSHA256:try digest(URL(fileURLWithPath:configuration.model)),pronunciationSHA256:try profile.fingerprint(),
            audioSHA256:try digest(wav),rawOutputPath:result.path,
            note:"Acoustic forced-alignment estimates, not measured ground-truth boundaries. The selected pronunciation remains uncertain. Missing phone confidence stays nil. Audio hash refers to the exact cropped WAV.")
        provenance.phoneMapSHA256=try digest(URL(fileURLWithPath:configuration.phoneMap))
        var aligned=try parse(Data(contentsOf:result),phoneMap:map,origin:sessionAudioOrigin+range.lowerBound,
                             duration:range.upperBound-range.lowerBound,provenance:provenance)
        for index in Set(aligned.phones.compactMap(\.wordIndex)) {
            let indices=aligned.phones.indices.filter {aligned.phones[$0].wordIndex==index}
            guard let word=indices.first.flatMap({aligned.phones[$0].word}) else {continue}
            let sequence=indices.map {aligned.phones[$0].ipa}
            let variants=profile.alternatives(word,lexicon:lexicon).filter {$0.ipa==sequence}.map(\.id)
            for i in indices {aligned.phones[i].pronunciationVariantIDs=variants}
        }
        return aligned
    }
    nonisolated static func digest(_ url:URL) throws -> String {
        let handle=try FileHandle(forReadingFrom:url);defer {try? handle.close()}
        var sha=SHA256()
        while let bytes=try handle.read(upToCount:1024*1024),!bytes.isEmpty { sha.update(data:bytes) }
        return sha.finalize().map {String(format:"%02x",$0)}.joined()
    }
    /// MFA exports phone tiers as [start, end, label] entries; confidence is not fabricated.
    nonisolated static func parse(_ data:Data,phoneMap:[String:String],origin:Double,duration:Double,
                                  provenance:PhoneAlignmentProvenance) throws -> PhoneAlignment {
        guard let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],
              let tiers=object["tiers"] as? [String:Any] else {throw LinguisticFailure.alignment}
        let candidates=tiers.filter {$0.key=="phones" || $0.key.hasSuffix(" - phones")}
        guard candidates.count==1,let tier=candidates.first?.value as? [String:Any],let entries=tier["entries"] as? [[Any]] else {throw LinguisticFailure.alignment}
        var phones:[PhoneBoundary]=[]
        let wordTiers=tiers.filter {$0.key=="words" || $0.key.hasSuffix(" - words")}
        let words=(wordTiers.count==1 ? (wordTiers.first?.value as? [String:Any])?["entries"] as? [[Any]]:nil) ?? []
        for entry in entries {
            guard entry.count>=3,let start=entry[0] as? Double,let end=entry[1] as? Double,let label=entry[2] as? String else {throw LinguisticFailure.alignment}
            if ["","sil","sp"].contains(label) {continue}
            guard let ipa=phoneMap[label],ArticulatoryInventory.byIPA[ipa] != nil else {throw LinguisticFailure.alignment}
            let wordIndex=words.indices.first {i in
                guard words[i].count>=3,let a=words[i][0] as? Double,let b=words[i][1] as? Double else {return false}
                return start>=a-1e-6 && end<=b+1e-6
            }
            phones.append(.init(ipa:ipa,start:origin+start,end:origin+end,confidence:nil,backendScore:nil,
                word:wordIndex.flatMap {words[$0][2] as? String},wordIndex:wordIndex))
        }
        let alignment=PhoneAlignment(condition:.normal,phones:phones,provenance:provenance)
        try alignment.validate(in:origin...(origin+duration));return alignment
    }
}
