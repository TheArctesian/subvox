import SwiftUI

struct MouthTeacherResultsView: View {
    let comparison: ControlledSignalComparison
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Camera teacher → AirPods test").font(.title2.weight(.semibold))
            if let reference = comparison.teacherReference {
                Text("Frozen after A · " + String(comparison.teacherFingerprint?.prefix(12) ?? "unavailable"))
                    .font(.caption).foregroundStyle(.secondary)
                Text("\(reference.trainingSources.count) recording sessions in training · source identities and hashes saved").font(.caption).foregroundStyle(.secondary)
                Text("Primary comparison: camera-taught linear decoder versus raw AirPods linear decoder. Other arms are controls.").font(.caption.weight(.semibold))
                ForEach(reference.groups.indices, id: \.self) { i in
                    let group = reference.groups[i]
                    VStack(alignment: .leading, spacing: 8) {
                        Text(group.condition == .silent ? "Mouthed phrases" : "Spoken phrases").font(.headline)
                        if let model = group.model {
                            Text(String(format: "Source validation: %.1f%% less mouth-velocity prediction error · shifted teacher: %.1f%%", model.sourceValidationGain*100, model.shiftedTeacherGain*100))
                            Text("Selected before new B: \(model.mode == .body ? "directional" : "gravity-invariant") IMU context · ridge \(model.ridge.formatted())")
                                .font(.caption).foregroundStyle(.secondary)
                            Text(group.validationScope).font(.caption).foregroundStyle(.secondary)
                            if model.sourceValidationGain <= 0 || model.sourceValidationGain <= model.shiftedTeacherGain {
                                Text("Source validation did not establish a useful aligned teacher advantage. The student remains experimental; the B transfer test is still reported.")
                                    .font(.caption).foregroundStyle(.orange)
                            }
                        }
                        if let note = group.note { Text(note).foregroundStyle(.orange) }
                        if let score = comparison.teacherScores?.first(where: { $0.condition == group.condition }) {
                            ScrollView(.horizontal) {
                              Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 7) {
                                GridRow {
                                    Text("Input / control"); Text("Matched / planned"); Text("Scored"); Text("Time-shift")
                                }.font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                ForEach(score.arms.indices, id: \.self) { j in
                                    let arm = score.arms[j]
                                    GridRow {
                                        Text(arm.arm.label)
                                        Text("\(arm.correct) / 9").monospacedDigit()
                                        Text("\(arm.evaluated)").monospacedDigit()
                                        Text("\(arm.shiftedCorrect) / \(arm.evaluated)").monospacedDigit()
                                    }.font(.callout)
                                }
                              }.fixedSize(horizontal: true, vertical: false).padding(.bottom, 6)
                            }
                            Text("\(score.collected) collected · \(9-score.collected) unrecorded. Unmatched valid takes count as incorrect.").font(.caption).foregroundStyle(.secondary)
                            if score.sensorSourceChanged {
                                Text("Core Motion reported a different headphone sensor after reseating. This may affect transfer independently of head motion.").font(.caption).foregroundStyle(.orange)
                            } else if score.sensorLocations.isEmpty {
                                Text("Headphone sensor side is unavailable in this recording.").font(.caption).foregroundStyle(.secondary)
                            }
                        } else {
                            Text("Training A is saved. Block B is still needed for the transfer test.").foregroundStyle(.secondary)
                        }
                    }.padding(.vertical, 8)
                }
            } else {
                Text("Complete all 18 source takes in A to train and freeze the teacher model.").foregroundStyle(.secondary)
            }
            if let note = comparison.teacherNote { Text(note).foregroundStyle(.orange) }
            Text("Source validation selects the model and is a development estimate. Previously examined sessions are training data now. Only the new reseated B block tests transfer. Student predictions use AirPods alone; camera mouth and camera head results are independent controls. A strong head-only result suggests residual head-motion confounding. This remains a three-phrase diagnostic, not a vocabulary decoder.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}
