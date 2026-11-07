from __future__ import annotations
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from . import VERSION
from .data import MIN_TRANSCRIPT_MATCH, prepare, load_dataset, atomic_json, digest, resample, features, words, pronunciations, group_id
from .decoding import Lexicon, BigramLM, decode, edit_alignment, homophone_equal, summary


def prediction(model, checkpoint, language, row):
    from .network import emission, evidence
    lp,tokens,following,activity,fallback=emission(model,checkpoint,row)
    lexicon=Lexicon(language,tokens)
    lm=BigramLM(checkpoint["lm_training_text"])
    plain=decode(lp,tokens,lexicon)
    assisted=decode(lp,tokens,lexicon,lm=lm)
    # Only models trained with the next-phone head get transition rescoring; reported separately.
    transition=decode(lp,tokens,lexicon,next_log_probs=following) if checkpoint["config"].get("transitions")=="next_phone" else None
    score=evidence(lp)
    threshold=(checkpoint.get("rejection") or {}).get("evidence_threshold")
    active=(not checkpoint["config"]["activity_aux"] or activity>=.5) and (threshold is None or score>threshold)
    def accept(result):
        return bool(result["words"]) and any(result["phones"]) and active and not result["ambiguous"]
    probabilities=np.exp(lp)
    accepted=accept(plain)
    reason=None if accepted else ("Motion evidence below the rejection threshold fitted on training non-speech clips" if threshold is not None and score<=threshold
                                  else "No supported unambiguous word path, insufficient phoneme evidence, or activity rejection")
    return dict(text=plain["words"] if accepted else "",text_with_lm=assisted["words"] if accept(assisted) else "",
                text_with_transitions=(transition["words"] if accept(transition) else "") if transition else None,
                phonemes=plain["phones"],accepted=accepted,lm_accepted=accept(assisted),
                lexical_hypothesis=plain["words"],homophones=plain["homophones"],alternatives=plain["ranked"],lm_alternatives=assisted["ranked"],
                transition_alternatives=transition["ranked"] if transition else None,
                margin=plain["margin"],activity_score=activity,activity_score_is_calibrated=False,
                evidence_score=score,evidence_threshold=threshold,
                invariant_fallback=fallback,source=row["quality"]["source"],
                mean_blank_probability=float(probabilities[:,0].mean()),mean_posterior_entropy=float(-(probabilities*lp).sum(1).mean()),
                rejection_reason=reason,
                provenance="motion-only neural CTC; word/phone scores and temporal correspondences are model estimates")


def fold_split(examples, fold):
    """Training and validation rows of one cross-validation fold; `None` trains on everything."""
    if fold is None:
        return list(examples), []
    return ([r for r in examples if r["fold_roles"][fold] == "training"],
            [r for r in examples if r["fold_roles"][fold] == "validation"])


def score_rows(rows, language, transitions):
    same = homophone_equal(language)
    for row in rows:
        expected = words(row["reference_text"])
        row["homophone_word_errors"] = edit_alignment(expected, words(row["text"]), same)[0]
        if transitions:
            row["transition_word_errors"] = edit_alignment(expected, words(row.get("text_with_transitions") or ""))[0]
    return rows


def evaluate_model(model_path, dataset_path, fold, output=None):
    from .network import load_model
    output=Path(output) if output else Path(model_path)/"development-report.json"
    if output.exists():raise ValueError("Evaluation output already saved; choose a new output to preserve the frozen report")
    model,checkpoint,manifest,language=load_model(model_path)
    examples,_,dataset_manifest=load_dataset(dataset_path)
    _,validation=fold_split(examples,fold)
    rows=[]
    for row in validation:
        if row["condition"]!=checkpoint["config"]["condition"]:
            continue
        if row["session"] in manifest["training_sessions"] or set(row["groups"])&set(manifest["training_groups"]):
            raise ValueError("Evaluation leaks a training session/sentence group")
        guessed=prediction(model,checkpoint,language,row)
        phones=[p for p in guessed["phonemes"] if p!="|"]
        refs=[[p for p in target if p!="|"] for target in row["targets"]]
        errors,alignment,reference=min(((*edit_alignment(ref,phones),ref) for ref in refs),key=lambda x:x[0])
        expected=words(row["text"])
        rows.append(dict(id=row["id"],fold=fold,session=row["session"],groups=row["groups"],reference_text=row["text"],activity=row["activity"],
                         phone_errors=errors,phone_count=len(reference),phone_alignment=alignment,
                         word_errors=edit_alignment(expected,words(guessed["text"]))[0],word_count=len(expected),
                         lm_word_errors=edit_alignment(expected,words(guessed["text_with_lm"]))[0],
                         **guessed))
    excluded=[e for e in dataset_manifest["excluded"] if e["fold_roles"][fold]=="validation" and e["condition"]==checkpoint["config"]["condition"]]
    for missing in excluded:
        # Invalid non-speech clips cannot establish successful rejection. They
        # remain in quality_exclusions, outside the scored false-accept rate.
        if not words(missing["text"]):continue
        try:
            paths,_=pronunciations(missing["text"],language)
            reference=min(([p for p in path if p!="|"] for path in paths),key=len)
        except ValueError:reference=[]
        expected=words(missing["text"])
        rows.append(dict(id=missing["id"],fold=fold,session=missing["session"],groups=missing["groups"],reference_text=missing["text"],activity=1,
            phone_errors=len(reference),phone_count=len(reference),phone_alignment=[(p,"<del>") for p in reference],
            word_errors=len(expected),word_count=len(expected),lm_word_errors=len(expected),accepted=False,text="",text_with_lm="",
            rejection_reason="Excluded by predeclared recording/label quality rules; retained in planned denominator"))
    score_rows(rows,language,checkpoint["config"].get("transitions")=="next_phone")
    report=dict(model=str(model_path),fold=fold,condition=checkpoint["config"]["condition"],experiment=checkpoint["config"]["name"],
                interpretation="Retrospective development data used for checkpoint selection; not an unbiased final test or fresh prospective reseat score",
                summary=summary(rows),rows=rows,dataset_sizes=dict(training=len(manifest["training_ids"]),development=len(rows)),
                quality_exclusions=excluded,
                reference_provenance="Recorded prompts/user corrections; pronunciation proposals can be wrong for this mixed-accent speaker",
                missing_training_phones=sorted({p for r in validation for path in r["targets"] for p in path if p!="|" and p not in checkpoint["coverage"]}))
    training_words={w for text in checkpoint["lm_training_text"] for w in text}
    development_words={w for r in rows if r["activity"] for w in words(r["reference_text"])}
    report["vocabulary"]=dict(training_word_types=len(training_words),development_word_types=len(development_words),
        unseen_development_word_types=sorted(development_words-training_words),
        interpretation="Sentence decoding containing unseen word types, not isolated-word accuracy")
    atomic_json(output,report)
    return report


def raw_dtw_baseline(examples, folds):
    """Matched-cohort whole-utterance baseline; cannot compose unseen sentences."""
    def pooled(raw):
        f=features(np.asarray(raw))[:,:16]
        indices=np.linspace(0,len(f)-1,min(80,len(f))).round().astype(int)
        return f[indices]
    def dtw(a,b,scale):
        previous=np.full(len(b)+1,np.inf);previous[0]=0
        for x in a:
            current=np.full(len(b)+1,np.inf)
            for j,y in enumerate(b,1):
                current[j]=min(previous[j],current[j-1],previous[j-1])+float(np.sqrt(np.mean(((x-y)/scale)**2)))
            previous=current
        return float(previous[-1]/(len(a)+len(b)))
    reports={}
    for condition in ("Normal speech","Silent mouthing"):
      errors=0;count=0;predictions=[]
      for fold in folds:
        training,validation=fold_split(examples,fold)
        training=[r for r in training if r["condition"]==condition and r["activity"]]
        validation=[r for r in validation if r["condition"]==condition and r["activity"]]
        if not training or not validation:continue
        templates=[pooled(r["raw"]) for r in training]
        scale=np.concatenate(templates).std(0).clip(1e-4)
        for row in validation:
            frame=pooled(row["raw"])
            distances=[dtw(frame,t,scale) for t in templates]
            winner=training[int(np.argmin(distances))]
            e=edit_alignment(words(row["text"]),words(winner["text"]))[0]
            errors+=e;count+=len(words(row["text"]))
            predictions.append(dict(fold=fold,expected=row["text"],predicted=winner["text"],word_errors=e))
      if predictions:
        reports[condition]=dict(word_error_rate=errors/max(1,count),utterances=len(predictions),predictions=predictions,
                                limit="Whole-sentence raw template retrieval; unlike CTC, cannot compose a sentence absent from training. Historical native word/jaw/camera baselines remain unchanged.")
    return reports


def train(dataset, output, epochs=60, experiments=("plain","diphone","augmented","activity")):
    """Per experiment and condition: one model per cross-validation fold (selected on that fold's
    development rows), their validation rows pooled into one report, then a final refit on every
    recording for the median selected epoch count. The refit is the frozen model the app uses;
    its scores come from the folds, never from rows it trained on."""
    from .network import fit
    examples,language,manifest=load_dataset(dataset)
    output=Path(output)
    if output.exists():raise ValueError("Run output exists; frozen runs are immutable")
    output.mkdir(parents=True)
    import shutil
    snapshot=output/"source_snapshot"
    shutil.copytree(Path(__file__).parent,snapshot/"subvox_ml",ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml","uv.lock"):
        source=Path(__file__).parent.parent/name
        if source.exists():shutil.copy2(source,snapshot/name)
    code_hash=digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    folds=[f["fold"] for f in manifest["split"]["folds"] if f["training"] and f["validation"]]
    comparisons=[];model_paths=[]
    for experiment in experiments:
        if experiment not in ("plain","diphone","augmented","activity","teacher"):
            raise ValueError("Unknown ablation")
        for condition,slug in (("Normal speech","spoken"),("Silent mouthing","mouthed")):
            config=dict(name=experiment,condition=condition,epochs=epochs,seed=42,code_hash=code_hash,
                        transitions="none" if experiment=="plain" else "next_phone",
                        augmentation=experiment in ("augmented","activity"),activity_aux=experiment=="activity",teacher_aux=experiment=="teacher",
                        dataset_hash=manifest["examples_hash"],language_hash=digest(language),reserved_final_groups=manifest["reserved_final_groups"])
            reports=[];selected=[]
            for fold in folds:
                training,validation=fold_split(examples,fold)
                if not any(r["condition"]==condition and r["activity"] for r in validation):continue
                destination=output/"folds"/f"fold-{fold}"/(experiment+"-"+slug)
                fitted=fit(training,validation,language,destination,dict(config,fold=fold))
                selected.append(fitted["refit_epochs"])
                reports.append(evaluate_model(destination,dataset,fold))
            if not reports:raise ValueError(f"No fold has {condition} development recordings")
            refit_epochs=max(5,int(np.median(selected)))
            destination=output/(experiment+"-"+slug)
            fit(examples,[],language,destination,dict(config,epochs=refit_epochs,fold=None,cross_validated_epochs=selected))
            rows=[row for report in reports for row in report["rows"]]
            pooled=dict(model=str(destination),condition=condition,experiment=experiment,
                        interpretation="Pooled validation rows of the cross-validation fold models; the refit beside this report trained on all of them, so these scores describe the method, not this file's weights on unseen data",
                        summary=summary(rows),rows=rows,folds=[dict(fold=r["fold"],model=r["model"],summary=r["summary"]) for r in reports],
                        missing_training_phones=sorted({p for r in reports for p in r["missing_training_phones"]}))
            atomic_json(destination/"development-report.json",pooled)
            comparisons.append(dict(condition=condition,experiment=experiment,folds=len(reports),refit_epochs=refit_epochs,**pooled["summary"]))
            model_paths.append(str(destination))
    baseline=raw_dtw_baseline(examples,folds)
    report=dict(schema=2,version=VERSION,dataset=str(dataset),models=model_paths,
                comparisons=comparisons,raw_sequence_baseline=baseline,split=manifest["split"],promoted_former_tests=manifest["promoted_former_tests"],
                admission=manifest.get("admission"),
                final_test_consumed=False,teacher_stage=manifest.get("teacher_stage","not used in stage-one training"),code_hash=code_hash,
                historical_baselines=manifest.get("historical_baselines",[]),
                limits=["Tiny personal dataset; uncertain prompt/pronunciation references.","Cross-validated development scores informed epoch selection; they are not an independent final evaluation.",
                        "Fresh same-fit/reseated/unseen-sentence capture must use these frozen weights, not refit them.",
                        "Historical camera/word/jaw baseline scores use different protocols and cannot establish a same-cohort improvement."])
    atomic_json(output/"comparison.json",report)
    if isinstance(manifest.get("teacher_stage"),dict):
        base=json.loads((Path(manifest["teacher_stage"]["base_run"])/"comparison.json").read_text())
        report["models"]=base["models"]+report["models"]
        report["comparisons"]=base["comparisons"]+report["comparisons"]
        report["teacher_comparison_note"]="Base and teacher stages retain identical raw examples/targets/folds. Camera is a training-only auxiliary; audio/phone timing proposals are retained as fallible reviewed provenance."
        atomic_json(output/"comparison.json",report)
    return report


def infer(model_path, sample_path):
    from .network import load_model
    value=json.loads(Path(sample_path).read_text())
    recording=value.get("recording",value)
    raw,quality=resample(recording)
    row=dict(raw=raw.tolist(),quality=quality)
    model,checkpoint,manifest,language=load_model(model_path)
    if value.get("expectedText") and group_id(value["expectedText"]) in manifest.get("reserved_final_groups",checkpoint["config"].get("reserved_final_groups",[])):
        raise ValueError("This sentence remains reserved for the untouched final test")
    result=prediction(model,checkpoint,language,row)
    result["model_sha256"]=manifest["weights_sha256"]
    result["quality"]=quality
    expected=value.get("expectedText")
    if expected is not None:
        # Reference is used AFTER inference only; never enters features, logits or search.
        result["evaluation"]=dict(expected=expected,word_errors=edit_alignment(words(expected),words(result["text"]))[0],word_count=len(words(expected)))
    return result


def evaluate_fresh(session_path):
    from .network import load_model
    visit=json.loads(Path(session_path).read_text())
    if not visit["complete"] or visit["role"] not in ("sameFit","reseated") or visit.get("modalities") != ["AirPods motion"]:
        raise ValueError("Need a closed, explicitly confirmed motion-only prospective test session")
    reports=[]
    for condition,path in visit["modelPaths"].items():
        model,checkpoint,manifest,language=load_model(path)
        if visit["startedUnix"]<manifest["created_unix"] or visit["modelFingerprints"][condition]!=manifest["weights_sha256"]:
            raise ValueError("Test predates model freeze or references different weights")
        if visit["session"]["id"] in manifest["training_sessions"]+manifest["development_sessions"]:
            raise ValueError("Prospective session was used during fitting/checkpoint selection")
        used=set(manifest["training_groups"]+manifest["development_groups"])
        planned=[p for p in visit["planned"] if p["condition"]==condition]
        takes=[t for t in visit["takes"] if not t.get("discardedReason") and t["recording"]["condition"]==condition]
        keys=[(t["sentenceID"],t["repetition"]) for t in takes]
        if len(set(keys))!=len(keys):raise ValueError("Duplicate test repetition cells")
        rows=[]
        for prompt in planned:
            group=group_id(prompt["text"])
            if group in used or group in manifest.get("reserved_final_groups",checkpoint["config"].get("reserved_final_groups",[])):
                raise ValueError("Prospective sentence overlaps training/development/final reservation")
            take=next((t for t in takes if t["sentenceID"]==group and t["repetition"]==prompt["repetition"]),None)
            result=dict(text="",text_with_lm="",phonemes=[],accepted=False,rejection_reason="Unrecorded/invalid take")
            if take:
                try:
                    raw,q=resample(take["recording"])
                    result=prediction(model,checkpoint,language,dict(raw=raw.tolist(),quality=q))
                except ValueError as error:result["rejection_reason"]=str(error)
            targets,_=pronunciations(prompt["text"],language)
            refs=[[p for p in t if p!="|"] for t in targets]
            guessed=[p for p in result["phonemes"] if p!="|"]
            errors,alignment,ref=min(((*edit_alignment(r,guessed),r) for r in refs),key=lambda t:t[0])
            expected=words(prompt["text"])
            rows.append(dict(groups=[group],reference_text=prompt["text"],activity=1,phone_errors=errors,phone_count=len(ref),phone_alignment=alignment,
                             word_errors=edit_alignment(expected,words(result["text"]))[0],word_count=len(expected),
                             lm_word_errors=edit_alignment(expected,words(result["text_with_lm"]))[0],**result))
        score_rows(rows,language,checkpoint["config"].get("transitions")=="next_phone")
        reports.append(dict(condition=condition,summary=summary(rows),rows=rows,model_fingerprint=manifest["weights_sha256"]))
    report=dict(role=visit["role"],session=visit["session"]["id"],reports=reports,
                interpretation="Prospective frozen-model test on new sentence groups; fit status is user-confirmed. Separate same-fit/reseat sessions are not a causal randomized experiment.")
    output=Path(session_path).parent/"frozen-evaluation.json"
    if output.exists():raise ValueError("Frozen evaluation already saved; refusing to overwrite")
    atomic_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description="Subvox real-motion Conv/GRU CTC")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--root",required=True);p.add_argument("--language",required=True);p.add_argument("--output",required=True);p.add_argument("--corrections");p.add_argument("--folds",type=int,default=4)
    p.add_argument("--min-transcript-match",type=float,default=MIN_TRANSCRIPT_MATCH,help="admit unreviewed pairs at or above this Whisper match; negative requires user acceptance")
    p=sub.add_parser("train");p.add_argument("--dataset",required=True);p.add_argument("--output",required=True);p.add_argument("--epochs",type=int,default=60);p.add_argument("--experiments",default="plain,diphone,augmented,activity")
    p=sub.add_parser("infer");p.add_argument("--model",required=True);p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("evaluate");p.add_argument("--model",required=True);p.add_argument("--dataset",required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--output")
    p=sub.add_parser("evaluate-fresh");p.add_argument("--session",required=True)
    p=sub.add_parser("prepare-teachers");p.add_argument("--dataset",required=True);p.add_argument("--base-run",required=True);p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        corrections=json.loads(Path(args.corrections).read_text()) if args.corrections else None
        result=prepare(args.root,args.language,args.output,corrections,args.folds,
                       None if args.min_transcript_match<0 else args.min_transcript_match)
        result=dict(dataset=args.output,split=result["split"],excluded=len(result["excluded"]),promoted_former_tests=len(result["promoted_former_tests"]),
                    unreviewed_pairs=len(result["admission"]["unreviewed_pairs"]))
    elif args.command=="prepare-teachers":
        from .teachers import enrich
        result=enrich(args.dataset,args.base_run,args.output)
    elif args.command=="train":
        if not 1<=args.epochs<=500:raise ValueError("Epochs must be in 1...500")
        result=train(args.dataset,args.output,args.epochs,args.experiments.split(","))
    elif args.command=="infer":
        result=infer(args.model,args.input);atomic_json(args.output,result)
    elif args.command=="evaluate-fresh":result=evaluate_fresh(args.session)
    else:result=evaluate_model(args.model,args.dataset,args.fold,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":main()
