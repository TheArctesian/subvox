"""Algorithm fixtures verify code, never establish anatomical/sensor accuracy."""
import itertools
import math
import tempfile
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np
import torch
from subvox_ml.data import assign_folds,discover,resample,features,pronunciations,digest,atomic_json,load_dataset
from subvox_ml.decoding import prefix_beam,Lexicon,decode,BigramLM,edit_alignment,summary
from subvox_ml.network import MotionCTC,ctc_variant_loss,next_phone_targets,next_phone_targets_batch,augment,standardise
from subvox_ml.data import canonical,group_id,FEATURE_NAMES
from subvox_ml.teachers import camera_targets,enrich


def language():
    return dict(inventory=[dict(ipa=p) for p in ("p","b","a")],entries={
        "pa":[dict(ipa=["p","a"],speakerConfirmed=True)],
        "ba":[dict(ipa=["b","a"],speakerConfirmed=True)],
        "pah":[dict(ipa=["p","a"],speakerConfirmed=False)]})


def recording(quiet=False):
    samples=[]
    for i in range(300):
        t=100+i/50
        samples.append(dict(t=t,ax=0 if quiet else math.sin(t)*.01,ay=0,az=0,rx=0,ry=0,rz=0,gx=0,gy=0,gz=-1,sensorLocation=1))
    return dict(cueStart=103,cueEnd=105.9,samples=samples,condition="Silent mouthing")


class SequenceTests(unittest.TestCase):
    def test_non_speech_visits_keep_empty_targets_and_separate_movement_groups(self):
        from subvox_ml.data import discover
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            takes=[dict(id="nod",text="",sentenceID="nod-group",recording=dict(recording(),id="nod")),
                   dict(id="chew",text="",sentenceID="chew-group",recording=dict(recording(),id="chew"))]
            path=root/"SequenceLearning/Collections/negative/session.json"
            atomic_json(path,dict(id="negative-visit",complete=True,role="negativeTraining",session=dict(id="negative-session"),takes=takes))
            rows,_,_=discover(root,language())
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r["text"]=="" and r["original_role"]=="Training" for r in rows))
            self.assertTrue(set(rows[0]["groups"]).isdisjoint(rows[1]["groups"]))
            self.assertEqual(pronunciations(rows[0]["text"],language())[0],[[]])
            takes[0]["text"]="Nod and turn your head"
            atomic_json(path,dict(id="negative-visit",complete=True,role="negativeTraining",session=dict(id="negative-session"),takes=takes))
            with self.assertRaisesRegex(ValueError,"instructions are not transcripts"):discover(root,language())
    def test_ctc_beam_matches_brute_force_with_repeated_labels(self):
        probabilities=np.array([[.3,.7],[.4,.6],[.6,.4]])
        expected={}
        for path in itertools.product(range(2),repeat=3):
            output=[];prior=None
            for token in path:
                if token and token!=prior:output.append(token)
                prior=token
            expected[tuple(output)]=expected.get(tuple(output),0)+np.prod([probabilities[i,p] for i,p in enumerate(path)])
        actual=dict(prefix_beam(np.log(probabilities),width=100,topk=2))
        for key,value in expected.items():self.assertAlmostEqual(math.exp(actual[key]),value,places=6)

    def test_pronunciation_variants_are_marginalised_not_selected_by_accent(self):
        logits=torch.tensor([[1.,2.,0.],[1.,0.,2.],[2.,1.,0.]],requires_grad=True)
        lp=logits.log_softmax(-1)
        costs=[ctc_variant_loss(lp,[p]) for p in ([1],[2])]
        marginal=ctc_variant_loss(lp,[[1],[2]])
        expected=-torch.logsumexp(-torch.stack(costs),0)+math.log(2)
        self.assertAlmostEqual(float(marginal.detach()),float(expected.detach()),places=5)
        marginal.backward();self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_next_phone_context_has_soft_posterior_without_guessed_boundaries(self):
        lp=torch.randn(12,5).log_softmax(-1)
        target,mass=next_phone_targets(lp,[2,2,1,3])
        self.assertEqual(tuple(target.shape),(12,5))
        self.assertTrue(torch.isfinite(target).all())
        self.assertLessEqual(float(mass.max()),1.0001)
        self.assertFalse(target.requires_grad)
        self.assertEqual(float(target[:,0].sum()),0)
        # Batched with a longer, padded neighbour, each row's posteriors stay its own.
        other=torch.randn(20,5).log_softmax(-1)
        padded=torch.stack([torch.cat([lp,torch.zeros(8,5)]),other])
        soft,batch_mass=next_phone_targets_batch(padded,torch.tensor([12,20]),[[2,2,1,3],[4,3]],5)
        torch.testing.assert_close(soft[0,:12],target);torch.testing.assert_close(batch_mass[0,:12],mass)
        self.assertEqual(float(batch_mass[0,12:].sum()),0)

    def test_real_temporal_network_learns_via_gradient_updates(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        model=MotionCTC(5);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.randn(1,25,32);length=torch.tensor([25]);losses=[]
        for _ in range(20):
            logits,_,_,_=model(x,length);loss=ctc_variant_loss(logits[0].log_softmax(-1),[[2,3]])
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        self.assertLess(losses[-1],losses[0]*.6)

    def test_folds_keep_sessions_and_sentence_groups_disjoint_and_respect_roles(self):
        rows=[dict(id=f"{s}-{g}",session=s,groups=[g],original_role="flexible") for s,g in itertools.product("ABC",["one","two","three"])]
        rows.append(dict(id="pilot",session="P",groups=["four"],original_role="Training"))
        roles,split=assign_folds(rows,set(),folds=3)
        self.assertEqual(split["k"],3)
        for f in range(3):
            train=[r for r in rows if roles[r["id"]][f]=="training"];val=[r for r in rows if roles[r["id"]][f]=="validation"]
            self.assertTrue(train and val)
            self.assertFalse({r["session"] for r in train}&{r["session"] for r in val})
            self.assertFalse({r["groups"][0] for r in train}&{r["groups"][0] for r in val})
        self.assertEqual(roles["pilot"],["training"]*3)
        self.assertEqual(sum(roles[r["id"]].count("validation") for r in rows),3)
        with self.assertRaises(ValueError):assign_folds(rows,{"one"})

    def test_paragraph_groups_remain_connected_across_folds(self):
        rows=[dict(id="a",session="A",groups=["one","two"],original_role="flexible"),
              dict(id="b",session="B",groups=["two"],original_role="flexible")]
        with self.assertRaises(ValueError):assign_folds(rows,set())

    def test_unreviewed_pairs_enter_only_with_a_strong_transcript_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            def pair(identity,review,match):
                take=lambda kind:dict(recording(),id=f"{identity}-{kind}")
                return dict(id=identity,text=f"pa {identity}",promptIDs=[],role="Training",review=review,transcriptMatchRate=match,
                            recordingSessionID="S",spoken=take("spoken"),silent=take("silent"))
            atomic_json(Path(tmp)/"sentence-training.json",dict(roles={},pairs=[pair("good","Needs review",.9),pair("weak","Needs review",.6),
                                                                                pair("dropped","Excluded",1.0),pair("ok","Accepted",None)]))
            admitted=lambda threshold:sorted({r["provenance"]["pair"] for r in discover(tmp,language(),min_transcript_match=threshold)[0]})
            self.assertEqual(admitted(.85),["good","ok"])
            self.assertEqual(admitted(None),["ok"])
            rows=discover(tmp,language())[0]
            self.assertIn("review pending",next(r for r in rows if r["provenance"]["pair"]=="good")["provenance"]["reference"])

    def test_timestamp_resampling_retains_subtle_motion_and_rejects_source_switches_and_gaps(self):
        r=recording(quiet=True);raw,q=resample(r)
        self.assertTrue(any("low motion" in f for f in q["flags"]))
        self.assertEqual(raw.shape[1],9)
        r["samples"][220]["sensorLocation"]=2
        with self.assertRaisesRegex(ValueError,"source switched"):resample(r)
        r=recording();r["samples"]=r["samples"][:200]+r["samples"][212:]
        with self.assertRaisesRegex(ValueError,"gap"):resample(r)

    def test_observed_tail_is_not_extended_and_derivatives_use_seconds(self):
        r=recording();r["samples"]=[s for s in r["samples"] if s["t"]<=105.84]
        raw,q=resample(r)
        self.assertGreater(q["end_gap"],.04)
        ramp=np.zeros((50,9),np.float32);ramp[:,0]=np.arange(50)/50;ramp[:,8]=-1
        f=features(ramp)
        np.testing.assert_allclose(f[1:,16],1,atol=1e-5)

    def test_orientation_perturbation_preserves_invariant_channels_and_validation_is_deterministic(self):
        raw,_=resample(recording())
        rng=np.random.default_rng(4);perturbed=augment(raw,rng)
        self.assertTrue(np.isfinite(perturbed).all())
        norm=dict(mean=[0]*32,scale=[1]*32)
        a,_=standardise(raw,norm,1,[1]);b,_=standardise(raw,norm,1,[1])
        self.assertTrue(torch.equal(a,b))
        c,fallback=standardise(raw,norm,2,[1])
        self.assertTrue(fallback);self.assertEqual(float(c[:,:9].abs().sum()),0)
        self.assertGreater(float(c[:,9:16].abs().sum()),0)

    def test_homophone_ties_are_accepted_with_alternatives_and_lm_uses_only_supplied_text(self):
        lang=language();tokens=["<blank>","a","b","p"];lex=Lexicon(lang,tokens)
        p=np.full((5,4),.001);p[0,3]=.99;p[1,0]=.99;p[2,1]=.99;p[3:,0]=.99;p/=p.sum(1,keepdims=True)
        plain=decode(np.log(p),tokens,lex)
        self.assertFalse(plain["ambiguous"])
        self.assertEqual((plain["words"],plain["homophones"]),("pa",["pah"]))
        from subvox_ml.decoding import homophone_equal
        self.assertEqual(edit_alignment(["pah","ba"],["pa","pa"],homophone_equal(lang))[0],1)
        assisted=decode(np.log(p),tokens,lex,BigramLM([["pa"]]*10))
        self.assertEqual(assisted["words"],"pa");self.assertFalse(assisted["ambiguous"])

    def test_decoder_places_word_boundaries_and_ends_only_on_complete_words(self):
        tokens=["<blank>","a","b","p"];lex=Lexicon(language(),tokens)
        def emit(*frames):
            p=np.full((len(frames),4),.001)
            for t,token in enumerate(frames):p[t,tokens.index(token)]=.99
            return np.log(p/p.sum(1,keepdims=True))
        self.assertEqual(decode(emit("p","a","b","a"),tokens,lex)["words"],"pa ba")
        # A trailing unfinished word is never returned; the beam ends on whole words only.
        unfinished=decode(emit("p","a","<blank>","b"),tokens,lex)
        self.assertEqual(unfinished["words"],"pa")

    def test_next_phone_head_breaks_an_acoustic_tie_between_transitions(self):
        lang=dict(inventory=[dict(ipa=p) for p in ("p","b","a")],entries={"ap":[dict(ipa=["a","p"])],"ab":[dict(ipa=["a","b"])]})
        tokens=["<blank>","a","b","p"];lex=Lexicon(lang,tokens)
        p=np.array([[.01,.97,.01,.01],[.02,.02,.48,.48]]);following=np.log(np.array([[.01,.01,.9,.08]]*2))
        self.assertTrue(decode(np.log(p),tokens,lex)["ambiguous"])
        self.assertEqual(decode(np.log(p),tokens,lex,next_log_probs=following)["words"],"ab")

    def test_legacy_separator_output_folds_into_blank(self):
        from subvox_ml.decoding import without_separator
        lp,tokens=without_separator(np.log([[.2,.5,.3]]),["<blank>","|","a"])
        self.assertEqual(tokens,["<blank>","a"])
        np.testing.assert_allclose(np.exp(lp),[[.7,.3]])

    def test_too_many_variants_and_unknown_words_stay_explicit(self):
        with self.assertRaises(ValueError):pronunciations("unknown",language())
        self.assertEqual(pronunciations("pa ba",language())[0],[["p","a","|","b","a"]])

    def test_error_metrics_include_rejection_and_avoid_false_precision_from_repetitions(self):
        error,alignment=edit_alignment(["p","a"],["p"])
        self.assertEqual(error,1);self.assertIn(("a","<del>"),alignment)
        row=dict(activity=1,groups=["one"],phone_errors=2,phone_count=2,word_errors=1,word_count=1,lm_word_errors=1,
                 accepted=False,phone_alignment=alignment)
        result=summary([row]*3)
        self.assertEqual(result["word_error_rate_no_lm"],1)
        self.assertIsNone(result["wer_cluster_bootstrap_95_interval"])

    def test_teacher_inference_reads_motion_only_and_camera_head_is_optional(self):
        from subvox_ml.network import emission
        model=MotionCTC(5,teacher=True).eval()
        raw,_=resample(recording())
        checkpoint=dict(normalisation=dict(mean=[0]*32,scale=[1]*32),trained_sources=[1],tokens=["<blank>","a","b","p","m"])
        row=dict(raw=raw.tolist(),quality=dict(source=1))
        a=emission(model,checkpoint,row)
        row["camera_teacher"]=[[float("nan")]*3]*len(raw)
        b=emission(model,checkpoint,row)
        np.testing.assert_array_equal(a[0],b[0])

    def test_camera_teacher_interpolation_refuses_missing_and_mixed_camera_data(self):
        r=recording();readings=[dict(t=100+i/30,cameraID="cam",estimator="fixture",mouth=dict(opening=.1,width=.7,outerAspect=.3,confidence=.9)) for i in range(180)]
        target,weight=camera_targets(readings,r,100,False)
        self.assertGreater((weight>0).sum(),90)
        missing=[v for v in readings if not 103.5<v["t"]<104.5]
        _,w=camera_targets(missing,r,100,False)
        self.assertLess((w>0).sum(),(weight>0).sum()-30)
        readings[2]["cameraID"]="another-camera"
        self.assertEqual(float(camera_targets(readings,r,100,False)[1].sum()),0)

    def test_teacher_snapshot_preserves_inputs_and_refuses_a_changed_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/"data";dataset.mkdir();(dataset/"source_material").mkdir()
            r=recording();raw,q=resample(r)
            readings=[dict(t=100+i/30,cameraID="cam",estimator="fixture",mouth=dict(opening=.1,width=.7,outerAspect=.3,confidence=.9)) for i in range(180)]
            original=dict(blocks=[dict(control=dict(readings=readings),takes=[dict(id=i,recording=r) for i in ("t","v")])])
            source_bytes=canonical(original);sha=hashlib.sha256(source_bytes).hexdigest()
            (dataset/"source_material/source.json").write_bytes(source_bytes)
            lang=language();atomic_json(dataset/"language.json",lang)
            rows=[dict(id=i,fold_roles=[role],condition="Silent mouthing",session=role,groups=[role],raw=raw.tolist(),targets=[["p","a"]],quality=q,activity=1,
                       provenance=dict(source="original.json",kind="controlled diagnostic")) for i,role in (("t","training"),("v","validation"))]
            atomic_json(dataset/"examples.json",rows)
            atomic_json(dataset/"manifest.json",dict(schema=2,examples_hash=digest(rows),language_hash=digest(lang),canonical_language_hash=digest(lang),
                sources={"original.json":sha},source_snapshots={"original.json":"source_material/source.json"}))
            import shutil
            base_dataset=root/"base-dataset";shutil.copytree(dataset,base_dataset)
            base=root/"base";base.mkdir();atomic_json(base/"comparison.json",dict(dataset=str(base_dataset)))
            out=root/"teacher";enrich(dataset,base,out)
            new,_,m=load_dataset(out)
            self.assertEqual(new[0]["raw"],rows[0]["raw"])
            self.assertEqual(new[0]["targets"],rows[0]["targets"])
            self.assertGreater(sum(new[0]["camera_weight"]),100)
            rows[0]["session"]="changed-session"
            atomic_json(dataset/"examples.json",rows)
            changed=json.loads((dataset/"manifest.json").read_text());changed["examples_hash"]=digest(rows)
            atomic_json(dataset/"manifest.json",changed)
            with self.assertRaisesRegex(ValueError,"same examples"):
                enrich(dataset,base,root/"invalid-teacher")
            self.assertFalse((root/"invalid-teacher").exists())

    def test_frozen_fresh_evaluation_refuses_reuse_and_counts_unrecorded_takes(self):
        from subvox_ml.cli import evaluate_fresh
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model_dir=root/"model";model_dir.mkdir();model=MotionCTC(5)
            config=dict(teacher_aux=False,activity_aux=False,condition="Silent mouthing")
            norm=dict(mean=[0]*32,scale=[1]*32)
            checkpoint=dict(state_dict=model.state_dict(),tokens=["<blank>","|","a","b","p"],normalisation=norm,
                            trained_sources=[1],config=config,lm_training_text=[["pa"]],coverage={"p":1,"a":1})
            torch.save(checkpoint,model_dir/"weights.pt")
            sha=hashlib.sha256((model_dir/"weights.pt").read_bytes()).hexdigest()
            atomic_json(model_dir/"language.json",language())
            manifest=dict(weights_sha256=sha,language_hash=digest(language()),feature_names=FEATURE_NAMES,config=config,normalisation=norm,
                          created_unix=100,training_sessions=["train"],development_sessions=["dev"],training_groups=[group_id("pa")],development_groups=[])
            atomic_json(model_dir/"model.json",manifest)
            visit=dict(complete=True,role="reseated",modalities=["AirPods motion"],modelPaths={"Silent mouthing":str(model_dir)},
                       modelFingerprints={"Silent mouthing":sha},startedUnix=99,session=dict(id="new"),takes=[],
                       planned=[dict(text="pah",condition="Silent mouthing",repetition=0)])
            path=root/"session.json";atomic_json(path,visit)
            with self.assertRaisesRegex(ValueError,"predates"):evaluate_fresh(path)
            visit["startedUnix"]=101;visit["session"]["id"]="train";atomic_json(path,visit)
            with self.assertRaisesRegex(ValueError,"used during"):evaluate_fresh(path)
            visit["session"]["id"]="new";atomic_json(path,visit)
            report=evaluate_fresh(path)
            self.assertEqual(report["reports"][0]["summary"]["utterances"],1)
            self.assertEqual(report["reports"][0]["summary"]["word_error_rate_no_lm"],1)
            self.assertEqual(hashlib.sha256((model_dir/"weights.pt").read_bytes()).hexdigest(),sha)
            with self.assertRaisesRegex(ValueError,"overwrite"):evaluate_fresh(path)

    def test_early_checkpoints_recover_final_reservation_without_rewriting_weights(self):
        from subvox_ml.network import load_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model_dir=root/"run"/"plain-mouthed";model_dir.mkdir(parents=True)
            dataset=root/"dataset";dataset.mkdir()
            model=MotionCTC(5);config=dict(teacher_aux=False,dataset_hash="frozen-dataset")
            norm=dict(mean=[0]*32,scale=[1]*32)
            torch.save(dict(state_dict=model.state_dict(),tokens=["<blank>","|","a","b","p"],
                            config=config,normalisation=norm),model_dir/"weights.pt")
            atomic_json(model_dir/"language.json",language())
            atomic_json(model_dir/"model.json",dict(weights_sha256=hashlib.sha256((model_dir/"weights.pt").read_bytes()).hexdigest(),
                language_hash=digest(language()),feature_names=FEATURE_NAMES,config=config,normalisation=norm))
            atomic_json(model_dir.parent/"comparison.json",dict(dataset=str(dataset)))
            atomic_json(dataset/"manifest.json",dict(examples_hash="frozen-dataset",reserved_final_groups=[group_id("pa")]))
            before={p.name:p.read_bytes() for p in model_dir.iterdir()}
            _,_,manifest,_=load_model(model_dir)
            self.assertEqual(manifest["reserved_final_groups"],[group_id("pa")])
            self.assertEqual(before,{p.name:p.read_bytes() for p in model_dir.iterdir()})
            atomic_json(dataset/"manifest.json",dict(examples_hash="different-dataset",reserved_final_groups=[]))
            with self.assertRaisesRegex(ValueError,"dataset identity mismatch"):load_model(model_dir)

    def test_reevaluation_cannot_replace_a_frozen_report(self):
        from subvox_ml.cli import evaluate_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/"development-report.json"
            atomic_json(report,dict(frozen=True));before=report.read_bytes()
            with self.assertRaisesRegex(ValueError,"preserve the frozen report"):evaluate_model(root,root/"unused-dataset",0)
            self.assertEqual(report.read_bytes(),before)


if __name__=="__main__":unittest.main()
