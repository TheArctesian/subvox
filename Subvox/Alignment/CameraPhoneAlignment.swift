import Foundation

/// A separate visual warp for the mouthed take. It does not copy spoken durations,
/// use AirPods features to create their own labels, or claim acoustic evidence for silence.
enum CameraPhoneAlignment {
    struct Frame { var time:Double;var shape:[Double] }
    static func frames(_ readings:[CameraPoseReading],range:ClosedRange<Double>) -> [Frame] {
        readings.filter {$0.t>=range.lowerBound && $0.t<=range.upperBound && $0.usedReceiptTime != true && ($0.mouth?.confidence ?? 0)>=0.5}
            .compactMap {r in r.mouth.map {.init(time:r.t,shape:[$0.opening,$0.width,$0.outerAspect])} }
    }
    static func transfer(_ source:PhoneAlignment,spoken:[Frame],mouthed:[Frame],profileHash:String,allowPartial:Bool = false) throws -> PhoneAlignment {
        guard source.condition == .normal,source.provenance.method == .localAcousticAlignment,
              spoken.count>=12,mouthed.count>=12,spoken.count<=1500,mouthed.count<=1500,
              [spoken,mouthed].allSatisfy({frames in
                  frames.allSatisfy {$0.shape.count==3 && $0.shape.allSatisfy(\.isFinite)} &&
                  zip(frames,frames.dropFirst()).allSatisfy {$1.time>$0.time && $1.time-$0.time<=0.12}
              }) else {throw LinguisticFailure.alignment}
        let n=spoken.count,m=mouthed.count
        let all=spoken+mouthed
        let scale=(0..<3).map {k -> Double in
            let mean=all.reduce(0) {$0+$1.shape[k]}/Double(all.count)
            return max(0.01,sqrt(all.reduce(0) {$0+pow($1.shape[k]-mean,2)}/Double(all.count)))
        }
        var cost=Array(repeating:Array(repeating:Double.infinity,count:m),count:n)
        var parent=Array(repeating:Array(repeating:UInt8(0),count:m),count:n)
        for i in 0..<n {for j in 0..<m where abs(Double(i)/Double(n-1)-Double(j)/Double(m-1))<=0.45 {
            let local=sqrt((0..<3).reduce(0) {$0+pow((spoken[i].shape[$1]-mouthed[j].shape[$1])/scale[$1],2)}/3)
            if i==0 && j==0 {cost[i][j]=local;continue}
            var options:[(Double,UInt8)]=[]
            if i>0 && j>0 {options.append((cost[i-1][j-1],1))}
            if i>0 {options.append((cost[i-1][j]+0.05,2))}
            if j>0 {options.append((cost[i][j-1]+0.05,3))}
            if let best=options.min(by:{$0.0<$1.0}) {cost[i][j]=local+best.0;parent[i][j]=best.1}
        }}
        guard cost[n-1][m-1].isFinite else {throw LinguisticFailure.alignment}
        var path:[(Int,Int)]=[],i=n-1,j=m-1
        while true {
            path.append((i,j));if i==0 && j==0 {break}
            switch parent[i][j] {case 1:i-=1;j-=1;case 2:i-=1;case 3:j-=1;default:throw LinguisticFailure.alignment}
        }
        var mappedTimes = Array(repeating: 0.0, count: n)
        var counts = Array(repeating: 0, count: n)
        for (spokenIndex, mouthedIndex) in path {
            mappedTimes[spokenIndex] += mouthed[mouthedIndex].time - mouthed[0].time
            counts[spokenIndex] += 1
        }
        mappedTimes = mappedTimes.indices.map { mouthed[0].time + mappedTimes[$0] / Double(counts[$0]) }
        func mapped(_ time: Double) -> Double? {
            guard time >= spoken[0].time, time <= spoken[n - 1].time else { return nil }
            var lo = 0, hi = n - 1
            while hi - lo > 1 {
                let mid = (lo + hi) / 2
                if spoken[mid].time <= time { lo = mid } else { hi = mid }
            }
            // Interpolate within the observed camera warp, including sub-frame acoustic spans.
            let fraction = (time - spoken[lo].time) / (spoken[hi].time - spoken[lo].time)
            return mappedTimes[lo] + fraction * (mappedTimes[hi] - mappedTimes[lo])
        }
        var phones: [PhoneBoundary] = []
        for phone in source.phones {
            guard let start = mapped(phone.start), let end = mapped(phone.end), end > start else {
                if allowPartial { continue }
                throw LinguisticFailure.alignment
            }
            var warped = phone
            warped.start = start
            warped.end = end
            warped.confidence = nil
            phones.append(warped)
        }
        let alignment=PhoneAlignment(condition:.silent,phones:phones,provenance:.init(method:.cameraWarpFromAcoustic,
            backend:"independent camera-shape DTW v2",modelSHA256:nil,pronunciationSHA256:profileHash,audioSHA256:nil,rawOutputPath:nil,
            sourceAlignmentID:source.id,warpCost:cost[n-1][m-1]/Double(path.count),
            note:"Estimated visual correspondence with independent pacing. Confidence is uncalibrated; inspect before accepting. The mouthed warp uses no audio evidence."))
        try alignment.validate(in:mouthed[0].time...mouthed[m-1].time);return alignment
    }
}
