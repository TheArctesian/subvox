import SwiftUI

/// Only the prompt may scroll. Its parent reserves room for controls and plots,
/// even when a large font or a long instruction needs more than the viewport.
struct RecordingPromptPanel<Content:View>:View {
    var color:Color
    @ViewBuilder var content:()->Content
    var body:some View {
        GeometryReader {geometry in
            ScrollView(.vertical) {
                content().padding(20).frame(maxWidth:.infinity)
                    .frame(minHeight:max(0,geometry.size.height),alignment:.center)
            }
            .frame(width:geometry.size.width,height:geometry.size.height)
            .background(Color(nsColor:.controlBackgroundColor).opacity(0.6),in:RoundedRectangle(cornerRadius:24))
            .clipped()
        }.frame(minHeight:0,maxHeight:.infinity)
    }
}

enum NonSpeechMovement:Int,CaseIterable,Identifiable {
    case nodAndTurn,chew,swallow,shiftPosture
    var id:Int {rawValue}
    var instruction:String {
        switch self {
        case .nodAndTurn:"Nod and turn your head gently, keeping your mouth relaxed."
        case .chew:"Chew gently with an empty mouth. Do not speak or mouth words."
        case .swallow:"Swallow naturally, then relax your mouth."
        case .shiftPosture:"Shift posture and look around naturally without speaking."
        }
    }
    var title:String {
        switch self {case .nodAndTurn:"Nod, then turn";case .chew:"Gentle chewing";case .swallow:"Natural swallow";case .shiftPosture:"Shift your posture"}
    }
    func cue(at seconds:Double)->String {
        switch self {
        case .nodAndTurn:seconds.truncatingRemainder(dividingBy:8)<4 ? "Slowly nod down and up":"Slowly turn left and right"
        case .chew:"Move your jaw gently with your mouth empty"
        case .swallow:"Swallow naturally once, then rest"
        case .shiftPosture:"Shift your shoulders and look to either side"
        }
    }
}

/// Functional motion illustration built from native SF Symbols, not camera data.
struct NonSpeechMovementGuide:View {
    var movement:NonSpeechMovement
    var color:Color
    var demonstrating:Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var origin=Date()
    var body:some View {
        TimelineView(.animation(minimumInterval:1/24,paused:reduceMotion || !demonstrating)) {timeline in
            let elapsed=reduceMotion || !demonstrating ? 0:timeline.date.timeIntervalSince(origin)
            VStack(spacing:14) {
                illustration(at:elapsed)
                VStack(spacing:7) {
                    Text(movement.title).font(.system(size:30,weight:.semibold))
                    Text(movement.cue(at:elapsed)).font(.system(size:16)).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    if reduceMotion, movement == .nodAndTurn {Text("Nod up and down, then turn left and right.").font(.callout).foregroundStyle(.secondary)}
                }
                Text("Movement example, not live tracking").font(.caption).foregroundStyle(.tertiary)
            }.frame(maxWidth:560)
        }
        .accessibilityElement(children:.ignore)
        .accessibilityLabel("Example: "+movement.title+". "+movement.instruction)
        .onChange(of:movement) {_,_ in origin=Date()}
        .onChange(of:demonstrating) {_,active in if active {origin=Date()}}
    }
    private func illustration(at time:Double)->some View {
        let nod=movement == .nodAndTurn && time.truncatingRemainder(dividingBy:8)<4
        let turn=movement == .nodAndTurn && !nod
        let wave=sin(time*Double.pi/2)
        return ZStack {
            Circle().stroke(.primary.opacity(0.07),lineWidth:1).frame(width:164,height:164)
            Circle().trim(from:0.12,to:0.38).stroke(color.opacity(0.55),style:StrokeStyle(lineWidth:2,lineCap:.round))
                .frame(width:164,height:164).rotationEffect(.degrees(-100))
            switch movement {
            case .nodAndTurn:
                Image(systemName:"faceid").resizable().scaledToFit().frame(width:105,height:105)
                    .rotation3DEffect(.degrees(nod ? wave*18:0),axis:(x:1,y:0,z:0))
                    .rotation3DEffect(.degrees(turn ? wave*28:0),axis:(x:0,y:1,z:0))
            case .chew:
                Image(systemName:"mouth").resizable().scaledToFit().frame(width:105,height:82)
                    .scaleEffect(x:1+wave*0.035,y:0.82+0.22*abs(sin(time*3)))
            case .swallow:
                Image(systemName:"person.bust.fill").resizable().scaledToFit().frame(width:105,height:118).opacity(0.85)
                Capsule().fill(color).frame(width:5,height:12)
                    .offset(y:10+min(1,time.truncatingRemainder(dividingBy:7)/2)*25)
                    .opacity(time.truncatingRemainder(dividingBy:7)<2 ? 1:0.25)
            case .shiftPosture:
                Image(systemName:"figure.stand").resizable().scaledToFit().frame(width:90,height:120)
                    .rotationEffect(.degrees(wave*5),anchor:.bottom).offset(x:wave*7)
            }
            Image(systemName:movement == .swallow ? "arrow.down":nod ? "arrow.up.and.down":"arrow.left.and.right")
                .font(.system(size:14,weight:.medium)).foregroundStyle(color).offset(x:94,y:movement == .swallow ? 25:0)
        }.foregroundStyle(.primary).frame(width:216,height:180)
    }
}

/// Geometry assigns real space to every section. Large prompt content can
/// scroll, but cannot push controls or the live trace beyond the window.
struct RecordingViewport<Header:View,Prompt:View,Actions:View,Motion:View>:View {
    @ViewBuilder var header:()->Header
    @ViewBuilder var prompt:()->Prompt
    @ViewBuilder var actions:()->Actions
    @ViewBuilder var motion:()->Motion
    var body:some View {
        GeometryReader {geometry in
            let promptHeight=max(1,geometry.size.height-64-112-132-2)
            VStack(spacing:0) {
                header().frame(height:64).padding(.horizontal,24).clipped()
                Divider()
                prompt().frame(width:max(1,geometry.size.width-40),height:promptHeight).padding(.horizontal,20).clipped()
                actions().frame(height:112).padding(.horizontal,24).clipped()
                Divider()
                motion().frame(height:132).padding(.horizontal,12).clipped()
            }.frame(width:geometry.size.width,height:geometry.size.height,alignment:.top).clipped()
                .background(Color(nsColor:.windowBackgroundColor))
        }
    }
}
