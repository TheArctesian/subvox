"""Optional second stage: measured numeric camera supervision, never an inference input.

Existing paired-sentence collection supplies local Whisper proposals, optional local acoustic
phone alignment and explicit review. Audio agreement is not a test of motion information.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from .data import load_dataset,atomic_json,digest,RATE


def camera_targets(readings,recording,count,baseline):
    selected=[]
    identities=set()
    for r in readings:
        m=r.get("mouth")
        if not m or m.get("confidence",0)<.5 or r.get("usedReceiptTime"):
            continue
        value=[m.get(k,float("nan")) for k in ("opening","width","outerAspect")]
        if not np.isfinite(value).all():continue
        selected.append((r["t"],value,m["confidence"]))
        identities.add((r["cameraID"],r.get("estimator")))
    target=np.zeros((count,3),dtype=np.float32);weight=np.zeros(count,dtype=np.float32)
    if len(selected)<3 or len(identities)!=1:return target,weight
    selected.sort(key=lambda x:x[0]);times=np.array([s[0] for s in selected]);values=np.array([s[1] for s in selected])
    confidence=np.array([s[2] for s in selected])
    start=max(recording["samples"][0]["t"],recording["cueStart"]-(2.5 if baseline else 0))
    for i,t in enumerate(start+np.arange(count)/RATE):
        j=int(np.searchsorted(times,t))
        if j==0 or j>=len(times) or times[j]-times[j-1]>.12:continue
        fraction=(t-times[j-1])/(times[j]-times[j-1])
        target[i]=values[j-1]*(1-fraction)+values[j]*fraction
        weight[i]=min(confidence[j-1],confidence[j])
    return target,weight


def enrich(dataset,base_run,output):
    dataset,base_run,output=Path(dataset),Path(base_run),Path(output)
    if output.exists():raise ValueError("Teacher snapshot exists; cannot overwrite")
    if not (base_run/"comparison.json").exists():raise ValueError("Complete the real-data CTC stage first")
    examples,language,manifest=load_dataset(dataset)
    comparison=json.loads((base_run/"comparison.json").read_text())
    base_examples,base_language,_=load_dataset(comparison["dataset"])
    # A new preparation may change labels/splits. Do not present different
    # cohorts as an auxiliary-loss ablation, including on repeated teacher runs.
    keys=("id","condition","session","groups","fold_roles","activity","raw","targets")
    def cohort(rows):return digest(sorted(({k:r[k] for k in keys} for r in rows),key=lambda r:r["id"]))
    if cohort(examples)!=cohort(base_examples) or digest(language)!=digest(base_language):
        raise ValueError("Teacher comparison requires the same examples, labels, pronunciations and splits as its frozen base run; train a new base run first")
    originals={}
    for path,sha in manifest["sources"].items():
        snapshot=manifest.get("source_snapshots",{}).get(path)
        raw=(dataset/snapshot).read_bytes() if snapshot else Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError("Teacher source differs from the base dataset")
        originals[path]=json.loads(raw)
    result=copy.deepcopy(examples);valid_frames=0;whisper_proposals=0;reviewed_alignments=0
    for row in result:
        source=originals[row["provenance"]["source"]]
        identity=row["id"].removesuffix("/rest");recording=None;readings=[];voice=None;alignment=None
        kind=row["provenance"]["kind"]
        if kind=="controlled diagnostic":
            for block in source["blocks"]:
                for take in block["takes"]:
                    if take["id"]==identity:recording=take["recording"];readings=block.get("control",{}).get("readings",[])
        elif kind=="representation pilot":
            for take in source["takes"]:
                if take["id"]==identity:recording=take["recording"];readings=take["camera"]
        elif kind=="reviewed sentence pair":
            for pair in source["pairs"]:
                for key,alignment_key in (("spoken","spokenPhones"),("silent","mouthedPhones")):
                    if (pair.get(key) or {}).get("id")==identity:
                        recording=pair[key];readings=pair.get("cameraReadings",[])
                        voice=dict(transcript=pair.get("transcript"),match_rate=pair.get("transcriptMatchRate"),
                                   status="fallible local ASR proposal; pair accepted by user; not evidence of motion accuracy")
                        alignment=pair.get(alignment_key)
        if recording:
            target,weight=camera_targets(readings,recording,len(row["raw"]),not row["activity"])
        else:target=np.zeros((len(row["raw"]),3));weight=np.zeros(len(row["raw"]))
        # Weights stay on every row: `fit` applies the teacher loss and its normalisation only to
        # the rows it trains on, so a fold's development rows are never supervised.
        row["camera_teacher"]=target.tolist();row["camera_weight"]=weight.tolist()
        row["teacher_provenance"]=dict(method="camera measured mouth shape; timestamp interpolation only within 120 ms",
            uncertainty="Vision confidence is an uncalibrated weight, not sensor accuracy",whisper=voice,
            reviewed_phone_alignment=alignment if alignment and alignment.get("reviewed") else None,
            note="Phone annotations retained for audit; CTC still marginalises alignment. No word-duration phone boundaries.")
        valid_frames+=int((weight>0).sum());whisper_proposals+=int(voice is not None);reviewed_alignments+=int(bool(alignment and alignment.get("reviewed")))
    if valid_frames<100:raise ValueError("Not enough numeric camera teacher frames in training; keep real-data-only models")
    manifest=copy.deepcopy(manifest)
    manifest.update(examples_hash=digest(result),teacher_stage=dict(base_run=str(base_run),base_dataset=str(dataset),
        base_examples_hash=digest(examples),training_teacher_frames=valid_frames,whisper_proposals=whisper_proposals,
        reviewed_phone_alignments=reviewed_alignments,inference_modalities=["AirPods motion"]))
    output.mkdir(parents=True)
    atomic_json(output/"examples.json",result);atomic_json(output/"language.json",language);atomic_json(output/"manifest.json",manifest)
    if (dataset/"source_material").exists():shutil.copytree(dataset/"source_material",output/"source_material")
    return manifest["teacher_stage"]
