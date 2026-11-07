import Foundation

/// A switch is a discontinuity, not an axis rotation to guess. Raw rows remain on disk.
enum MotionSourceSafety {
    static func source(_ recording:SentenceRecording)->Int? {
        let samples=recording.samples
        guard let source=samples.first?.sensorLocation,[1,2].contains(source),
              samples.allSatisfy({$0.sensorLocation==source}),
              zip(samples,samples.dropFirst()).allSatisfy({$1.t>$0.t && $1.t-$0.t<=0.12}),
              let first=samples.first,let last=samples.last,
              first.t<=recording.cueStart-2.3,last.t>=recording.cueEnd-0.1,
              recording.cueEnd-recording.cueStart>=0.5,recording.cueEnd-recording.cueStart<=45 else {return nil}
        return source
    }
    static func stable(_ samples:[MotionSample],now:Double,seconds:Double=1.5)->Bool {
        let recent=samples.filter {$0.t>=now-seconds && $0.t<=now}
        guard recent.count>=15,let first=recent.first,let last=recent.last,
              now-last.t<=0.15,last.t-first.t>=seconds-0.12,
              let source=last.sensorLocation,[1,2].contains(source) else {return false}
        return recent.allSatisfy {$0.sensorLocation==source} && zip(recent,recent.dropFirst()).allSatisfy {$1.t>$0.t && $1.t-$0.t<=0.12}
    }
}

struct RepresentationTake:Codable,Identifiable,Sendable {
    var id=UUID()
    var sentenceID:String
    var text:String
    var sessionID:UUID
    var repetition:Int
    var recording:SentenceRecording
    var camera:[CameraPoseReading]
    var discardedReason:String?
    var phoneAlignment:PhoneAlignment?
    /// Optional so historical source digests and frozen references retain their encoding.
    var wordFocusGuide:WordFocusGuide?
}
