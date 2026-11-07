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
from subvox_ml.data import canonical,group_id,FEATURE_NAMES,FEATURE_WIDTH as W
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
        x=torch.randn(1,25,W);length=torch.tensor([25]);losses=[]
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
        self.assertEqual(raw.shape[1],13);self.assertEqual(q["attitude"],"missing")  # fixture has no quaternion
        r["samples"][220]["sensorLocation"]=2
        with self.assertRaisesRegex(ValueError,"source switched"):resample(r)
        r=recording();r["samples"]=r["samples"][:200]+r["samples"][212:]
        with self.assertRaisesRegex(ValueError,"gap"):resample(r)

    def test_automatic_pair_admission_ignores_word_timing_but_preserves_per_take_quality_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            def pair(identity,review="Needs review",match=1.0):
                return dict(id=identity,text="pa ba",promptIDs=[],role="Training",review=review,transcriptMatchRate=match,
                            error="Too few words have usable audio-aligned times.",alignment=None,recordingSessionID="S",
                            spoken=dict(recording(quiet=True),id=identity+"-spoken"),silent=dict(recording(),id=identity+"-silent"))
            partial=pair("partial")
            partial["silent"]["samples"]=partial["silent"]["samples"][:200]+partial["silent"]["samples"][212:]
            dropped=pair("dropped","Excluded")
            reserved=pair("reserved");reserved["promptIDs"]=["final-group"]
            confirmed=pair("confirmed","Accepted",.6)
            path=Path(tmp)/"sentence-training.json"
            atomic_json(path,dict(roles={"final-group":"Final test"},pairs=[partial,dropped,reserved,confirmed,pair("weak",match=.6)]))
            original=path.read_bytes()
            rows,_,_=discover(tmp,language())
            self.assertEqual({r["provenance"]["pair"] for r in rows},{"partial","confirmed"})
            valid,excluded={},{}
            for row in rows:
                try:valid[row["id"]]=resample(row["recording"])[1]
                except ValueError as error:excluded[row["id"]]=str(error)
            self.assertIn("partial-spoken",valid)
            self.assertIn("low motion", " ".join(valid["partial-spoken"]["flags"]))
            self.assertIn("gap",excluded["partial-silent"])
            self.assertEqual(path.read_bytes(),original,"Automatic decisions must preserve original recordings and user overrides")

    def test_observed_tail_is_not_extended_and_derivatives_use_seconds(self):
        r=recording();r["samples"]=[s for s in r["samples"] if s["t"]<=105.84]
        raw,q=resample(r)
        self.assertGreater(q["end_gap"],.04)
        ramp=np.zeros((50,9),np.float32);ramp[:,0]=np.arange(50)/50;ramp[:,8]=-1
        f=features(ramp)
        np.testing.assert_allclose(f[1:,FEATURE_NAMES.index("d_ax")],1,atol=1e-5)

    def test_orientation_perturbation_preserves_invariant_channels_and_validation_is_deterministic(self):
        raw,_=resample(recording())
        rng=np.random.default_rng(4);perturbed=augment(raw,rng)
        self.assertTrue(np.isfinite(perturbed).all())
        norm=dict(mean=[0]*W,scale=[1]*W)
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
        checkpoint=dict(normalisation=dict(mean=[0]*W,scale=[1]*W),trained_sources=[1],tokens=["<blank>","a","b","p","m"])
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
            atomic_json(dataset/"manifest.json",dict(schema=3,examples_hash=digest(rows),language_hash=digest(lang),canonical_language_hash=digest(lang),
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
            norm=dict(mean=[0]*W,scale=[1]*W)
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
            norm=dict(mean=[0]*W,scale=[1]*W)
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

    def test_forced_choice_prefers_the_spelled_sentence_and_lists_unfit_candidates(self):
        from subvox_ml.evaluation import forced_choice
        tokens=["<blank>","a","b","p"]
        p=np.full((6,4),.01)
        for t,token in enumerate(["p","<blank>","a","<blank>","p","a"]):p[t,tokens.index(token)]=.97
        lp=np.log(p/p.sum(1,keepdims=True))
        result=forced_choice(lp,{"pa pa":[[3,1,3,1]],"ba ba":[[2,1,2,1]],"long":[[3,1]*4]},"pa pa")
        self.assertEqual(result["fc_rank"],1);self.assertGreater(result["fc_margin"],0)
        self.assertEqual(result["fc_unfit"],["long"])

    def test_likelihood_ratio_removes_a_preference_for_common_phones(self):
        from subvox_ml.evaluation import forced_choice
        # A blank-heavy model that always leans to "a": raw scores favour the a-rich candidate
        # whatever the motion; against the same prior the clip's real evidence for "p b" wins.
        tokens=["<blank>","a","b","p"]
        prior=np.log([.7,.2,.05,.05])
        p=np.tile(np.exp(prior),(6,1));p[1]=[.6,.15,.05,.2];p[4]=[.6,.15,.2,.05]
        lp=np.log(p/p.sum(1,keepdims=True))
        candidates={"pb":[[3,2]],"aa":[[1,1]]}
        self.assertEqual(forced_choice(lp,candidates,"pb")["fc_rank"],2)
        self.assertEqual(forced_choice(lp,candidates,"pb",log_prior=prior)["fc_rank"],1)

    def test_distractors_exclude_trained_and_reserved_sentences_and_match_length(self):
        from subvox_ml.evaluation import distractor_candidates
        with tempfile.TemporaryDirectory() as tmp:
            bank=Path(tmp)/"bank.txt";bank.write_text("pa ba\nba pa\npa pa\nba ba\npa\npa ba pa ba\n")
            lang=language();index={"<blank>":0,"a":1,"b":2,"p":3}
            excluded={group_id("ba pa"),group_id("ba ba")}
            chosen=distractor_candidates("pa ba",lang,index,excluded,count=2,bank=bank)
            self.assertEqual(set(chosen),{"pa ba","pa pa","pa"})
            # Punctuation and case don't change a sentence's group, so its distractors stay the same.
            self.assertEqual(set(chosen)-{"pa ba"},set(distractor_candidates("Pa ba!",lang,index,excluded,count=2,bank=bank))-{"Pa ba!"})

    def test_distractor_bank_resolves_in_app_runtime_and_checkout_layouts(self):
        from subvox_ml.evaluation import find_distractor_bank
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/"Runtime"/"hash";(runtime/"subvox_ml").mkdir(parents=True);(runtime/"Corpus").mkdir()
            (runtime/"Corpus"/"diphones.txt").write_text("pa\n")
            self.assertEqual(find_distractor_bank(runtime/"subvox_ml"),runtime/"Corpus"/"diphones.txt")
            checkout=Path(tmp)/"repo";(checkout/"ml"/"subvox_ml").mkdir(parents=True);(checkout/"Corpus").mkdir()
            (checkout/"Corpus"/"diphones.txt").write_text("pa\n")
            self.assertEqual(find_distractor_bank(checkout/"ml"/"subvox_ml"),checkout/"Corpus"/"diphones.txt")

    def test_label_randomisation_null_separates_signal_from_chance(self):
        from subvox_ml.evaluation import permutation_null
        def rows(correct):
            out=[]
            for g in range(6):
                options=[f"s{g}-{k}" for k in range(5)];truth=options[0]
                for rep_ in range(3):
                    best=truth if correct else options[1+(g+rep_)%4]
                    out.append(dict(groups=[str(g)],reference_text=truth,fc_rank=1 if correct else 2,fc_scores={o:float(o==best) for o in options}))
            return out
        self.assertLess(permutation_null(rows(True),draws=500)["p_value"],.01)
        self.assertGreater(permutation_null(rows(False),draws=500)["p_value"],.5)

    def test_transmitted_information_and_cohort_entropy_on_known_cases(self):
        from subvox_ml.articulatory import transmitted_information,cohort_entropy,coverage_check
        identity=transmitted_information([(x,x) for x in "abcd"]*2500)
        self.assertAlmostEqual(identity["bits"],2,places=2)
        rng=np.random.default_rng(0)
        self.assertLess(abs(transmitted_information([(x,y) for x,y in zip(rng.choice(4,10000),rng.choice(4,10000))])["bits"]),.05)
        lex=dict(entries={w:[dict(ipa=list(w))] for w in ("pa","ba","ma","ta")})
        self.assertAlmostEqual(cohort_entropy(lex,"jaw_round_labial")["conditional_entropy_bits"],.75*math.log2(3))
        with self.assertRaises(ValueError):coverage_check(dict(inventory=[dict(ipa="q")]))

    def test_prior_correction_is_identity_at_zero_and_exposes_a_phone_hidden_under_blank(self):
        from subvox_ml.decoding import prior_correct
        from subvox_ml.network import greedy_phones
        p=np.array([[.9,.06,.02,.02]]*3+[[.6,.36,.02,.02]]);lp=np.log(p)
        prior=np.log([.9,.04,.03,.03])
        np.testing.assert_allclose(prior_correct(lp,prior,0),lp,atol=1e-9)
        self.assertEqual(greedy_phones(torch.tensor(lp)),[])
        corrected=prior_correct(lp,prior,.5)
        np.testing.assert_allclose(np.logaddexp.reduce(corrected,axis=1),0,atol=1e-9)
        self.assertEqual(greedy_phones(torch.tensor(corrected)),[1])

    def test_rest_baseline_normalisation_cancels_a_per_recording_gain_and_offset(self):
        from subvox_ml.network import baseline_stats
        rng=np.random.default_rng(3)
        def take(gain,offset):
            active=np.zeros((120,9),np.float32);rest=np.zeros((80,9),np.float32)
            active[:,:6]=np.sin(np.arange(120)[:,None]/4+np.arange(6))*.02;rest[:,:6]=rng.normal(0,.003,(80,6)) if gain==1 else base_rest[:,:6]
            active[:,8]=rest[:,8]=-1
            active[:,:6]=active[:,:6]*gain+offset;rest[:,:6]=rest[:,:6]*gain+offset
            return dict(raw=active.tolist(),baseline_raw=rest.tolist(),activity=1),rest
        global base_rest
        first,base_rest=take(1,0)
        second,_=take(1.8,np.linspace(-.01,.01,6))
        def normalised(row):
            median,scale,_=baseline_stats(row,[1e-6]*W)
            return (features(np.asarray(row["raw"]))-median)/scale
        columns=list(range(6))+[FEATURE_NAMES.index("d_"+k) for k in ("ax","ay","az","rx","ry","rz")]  # raw axes and their derivatives are affine per channel
        np.testing.assert_allclose(normalised(first)[:,columns],normalised(second)[:,columns],atol=1e-3)
        quiet=dict(raw=first["raw"],baseline_raw=[[0]*8+[-1]]*80,activity=1)
        self.assertTrue(np.isfinite(baseline_stats(quiet,[.1]*W)[1]).all())
        self.assertTrue(baseline_stats(dict(raw=first["raw"],baseline_raw=None),[.1]*W)[2])

    def test_coarse_schedule_and_class_map_are_fixed_and_complete(self):
        from subvox_ml.network import coarse_weight,coarse_units
        self.assertEqual([round(coarse_weight(e,8),2) for e in (0,3,8)],[1.0,.65,.3])
        lang=dict(language(),entries={"pa":[dict(ipa=["p","a"])],"ba":[dict(ipa=["b","a"])],"ap":[dict(ipa=["a","p"])]})
        alphabet,mapping,count=coarse_units(["<blank>","a","b","p"],lang)
        self.assertEqual(mapping[0],0);self.assertTrue(all(m>0 for m in mapping[1:]))
        self.assertEqual(mapping[2],mapping[3])  # b and p share every observable feature

    def test_validated_fit_saves_selected_and_fixed_budget_checkpoints_with_optional_heads(self):
        from subvox_ml.network import fit,load_model
        raw,q=resample(recording());rest,rq=resample(recording(),baseline=True)
        def row(i,split,session):
            return [dict(id=f"{i}",text="pa",targets=[["p","a"]],condition="Silent mouthing",activity=1,raw=raw.tolist(),quality=q,session=session,groups=[split]),
                    dict(id=f"{i}/rest",text="",targets=[[]],condition="Silent mouthing",activity=0,raw=rest.tolist(),quality=rq,session=session,groups=[split])]
        training=[r for i in range(5) for r in row(i,"t","A")];validation=row(9,"v","B")
        lang=dict(language(),inventory=[dict(ipa=p) for p in ("p","b","a","m")])
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(name="t",condition="Silent mouthing",epochs=3,seed=1,transitions="next_phone",augmentation=True,activity_aux=False,
                        teacher_aux=False,coarse_aux=True,normalisation="rest_baseline")
            manifest=fit(training,validation,lang,Path(tmp)/"m",config)
            self.assertIn("development_coarse_error",manifest["history"][-1])
            _,selected,_,_=load_model(Path(tmp)/"m");_,final,_,_=load_model(Path(tmp)/"m","final-epoch")
            self.assertIn("baseline",selected["normalisation"]);self.assertEqual(len(selected["label_log_prior"]),len(selected["tokens"]))
            self.assertGreater(selected["coarse_classes"],0)
            # Both checkpoints carry their own rejection threshold, or the fixed one accepts everything.
            self.assertIsNotNone(selected["rejection"]);self.assertIsNotNone(final["rejection"])
            refit=fit(training,[],lang,Path(tmp)/"r",dict(config,epochs=2))
            with self.assertRaisesRegex(ValueError,"no final-epoch"):load_model(Path(tmp)/"r","final-epoch")
            self.assertEqual(refit["best_epoch"],2)

    def test_jaw_cycle_counter_tracks_bursts_and_ignores_drift(self):
        from subvox_ml.rhythm import bandpass,count_cycles,mad,gate
        t=np.arange(0,6,1/50)
        for n in range(2,9):
            rng=np.random.default_rng(n)
            x=np.where((t>1)&(t<1+n/4),np.sin(2*np.pi*4*(t-1)),0)+.8*np.sin(2*np.pi*.5*t)+rng.normal(0,.25,len(t))
            rest=bandpass((.8*np.sin(2*np.pi*.5*t[:100])+rng.normal(0,.25,100))[:,None])[:,0]
            self.assertLessEqual(abs(count_cycles(bandpass(x[:,None])[:,0],mad(rest))-n),1)
        self.assertEqual(count_cycles(bandpass((.8*np.sin(2*np.pi*.5*t))[:,None])[:,0],.05),0)
        impulse=np.zeros(500);impulse[250]=1
        response=np.abs(np.fft.rfft(bandpass(impulse[:,None])[:,0]));hz=np.fft.rfftfreq(500,1/50)
        gain=lambda f:response[np.argmin(abs(hz-f))]
        self.assertGreater(gain(4),.9);self.assertLess(gain(.5),.32);self.assertLess(gain(15),.32)
        rows=[dict(cycles=c,groups=[str(c)]) for c in range(1,9)]
        self.assertTrue(gate(rows,list(range(1,9)),draws=200)["rho"]>.99)

    def test_shared_model_conditions_change_output_and_keep_blank_biases_separate(self):
        from subvox_ml.network import MotionCTC
        torch.manual_seed(0)
        model=MotionCTC(5,conditions=2)
        with torch.no_grad():model.blank_bias.copy_(torch.tensor([0.,1.]))
        x=torch.randn(1,30,W);lengths=torch.tensor([30])
        spoken=model(x,lengths,conditions=torch.tensor([0]))[0];silent=model(x,lengths,conditions=torch.tensor([1]))[0]
        self.assertFalse(torch.allclose(spoken,silent))
        model.zero_grad();model(x,lengths,conditions=torch.tensor([0]))[0].sum().backward()
        self.assertEqual(float(model.blank_bias.grad[1]),0);self.assertNotEqual(float(model.blank_bias.grad[0]),0)

    def test_order_measures_reward_motion_order_and_cancel_for_order_blind_models(self):
        from subvox_ml.evaluation import forced_choice,order_candidates,input_dependence,motion_nulls
        tokens=["<blank>","a","b","p","i"]
        order=[3,1,2,4]  # p a b i, each phone peaked in its own frames
        p=np.full((12,5),.02);p[:,0]=.5
        for k,token in enumerate(order):p[3*k+1,token]=.9
        lp=np.log(p/p.sum(1,keepdims=True))
        rng=np.random.default_rng(0)
        orders={"truth":[order],**order_candidates(order,rng,count=5)}
        self.assertTrue(all(sorted(v[0])==sorted(order) and v[0]!=order for k,v in orders.items() if k!="truth"))
        self.assertEqual(forced_choice(lp,orders,"truth")["fc_rank"],1)
        flat=np.log(np.tile(np.exp(lp).mean(0),(12,1)))  # order-blind: same phones, no timing
        self.assertGreater(input_dependence(lp,{"shuffle":flat},[order])["indep_shuffle"],0)
        same=forced_choice(flat,orders,"truth",null_log_probs=[flat])
        self.assertTrue(all(abs(v)<1e-5 for v in same["fc_scores"].values()))
        raw=np.arange(60*9,dtype=float).reshape(60,9)
        nulls=motion_nulls(raw,np.random.default_rng(1))
        for copy in nulls.values():self.assertEqual(copy.shape,raw.shape)
        self.assertEqual(sorted(nulls["reverse"][:,0]),sorted(raw[:,0]))
        # Joins are crossfaded over 3 frames, so a cut steps at most a quarter of its jump.
        smooth=np.sin(np.arange(200)/8)[:,None].repeat(9,1)
        faded=motion_nulls(smooth,np.random.default_rng(2))
        self.assertLess(np.abs(np.diff(faded["shuffle"][:,0])).max(),.6)  # a hard cut here can step by up to 2
        np.testing.assert_array_equal(nulls["shuffle"],motion_nulls(raw,np.random.default_rng(1))["shuffle"])

    def test_jaw_angle_channels_recover_a_jaw_rotation_and_drop_drift_and_sign_flips(self):
        from subvox_ml.data import attitude_channels
        t=np.arange(0,6,1/50)
        angle=np.radians(2)*np.sin(2*np.pi*3*t)+np.radians(10)*np.sin(2*np.pi*.05*t)  # 3 Hz jaw on slow posture drift
        q=np.stack([np.cos(angle/2),np.sin(angle/2),0*t,0*t],1)
        channels=attitude_channels(q)
        jaw=np.radians(2)*np.sin(2*np.pi*3*t)
        middle=slice(50,250)  # away from filter edges
        self.assertGreater(np.corrcoef(channels[middle,0],jaw[middle])[0,1],.95)
        flipped=q.copy();flipped[150:]*=-1
        np.testing.assert_allclose(attitude_channels(flipped),channels,atol=1e-5)
        np.testing.assert_array_equal(attitude_channels(np.tile([1.,0,0,0],(300,1))),0)
        mirrored=q*[1,-1,1,1]  # rotation about -x: the other ear's view
        np.testing.assert_allclose(attitude_channels(mirrored)[:,3],channels[:,3],atol=1e-5)

    def test_local_encoder_has_a_bounded_receptive_field_and_ignores_padding(self):
        torch.manual_seed(0)
        local=MotionCTC(5,encoder="local").eval();gru=MotionCTC(5).eval()
        x=torch.randn(1,120,W);lengths=torch.tensor([120])
        bumped=x.clone();bumped[0,10]+=5
        with torch.no_grad():
            changed=lambda m:(m(bumped,lengths)[0]-m(x,lengths)[0]).abs().amax(-1)[0]>1e-6
            reach=torch.nonzero(changed(local)).flatten()
            self.assertLessEqual(int(reach.max()),10+18)  # stem ±3 frames, blocks ±15
            self.assertTrue(bool(changed(gru)[30]))       # the bidirectional GRU reaches past the local limit
            padded=torch.cat([x,torch.zeros(1,30,W)],1)
            torch.testing.assert_close(local(padded,lengths)[0][:,:120],local(x,lengths)[0])
        self.assertLessEqual(sum(p.numel() for p in local.parameters()),1.5*sum(p.numel() for p in gru.parameters()))

    def test_motion_contrast_logs_margins_against_partner_and_rest_motion(self):
        from subvox_ml.network import fit
        raw,q=resample(recording());rest,rq=resample(recording(quiet=True),baseline=True)
        rows=lambda split,session,n:[r for i in range(n) for r in (
            dict(id=f"{split}{i}",text="pa",targets=[["p","a"]],condition="Silent mouthing",activity=1,raw=(raw*(1+i)).tolist(),quality=q,session=session,groups=[split]),
            dict(id=f"{split}{i}/rest",text="",targets=[[]],condition="Silent mouthing",activity=0,raw=rest.tolist(),quality=rq,session=session,groups=[split]))]
        lang=dict(language(),inventory=[dict(ipa=p) for p in ("p","b","a","m")])
        with tempfile.TemporaryDirectory() as tmp:
            manifest=fit(rows("t","A",3),rows("v","B",1),lang,Path(tmp)/"m",dict(name="c",condition="Silent mouthing",epochs=2,seed=1,
                         transitions="none",augmentation=False,activity_aux=False,teacher_aux=False,motion_contrast=True))
            self.assertIn("development_margin",manifest["history"][-1])

    def test_side_backfill_needs_passing_held_out_sessions_and_never_relabels_known_takes(self):
        from subvox_ml.side import infer_sides
        rng=np.random.default_rng(0)
        def take(session,side,known=True,tilt=None):
            g=np.zeros((60,9));g[:,6]=(.3 if side==1 else -.3) if tilt is None else tilt;g[:,8]=-.95
            g[:,6]+=rng.normal(0,.02,60)
            return dict(id=f"{session}-{rng.integers(1e9)}",session=session,raw=g.tolist(),quality=dict(source=side if known else -1))
        rows=[take(f"L{i}",1) for i in range(3) for _ in range(4)]+[take(f"R{i}",2) for i in range(3) for _ in range(4)]
        unknown=[take("U",2,known=False) for _ in range(4)]
        inferred,report=infer_sides(rows+unknown)
        self.assertTrue(report["accepted"]);self.assertEqual(set(inferred.values()),{2});self.assertEqual(len(inferred),4)
        self.assertFalse(any(r["id"] in inferred for r in rows))
        mixed=[take("M",1,known=False),take("M",2,known=False)]
        self.assertEqual(infer_sides(rows+mixed)[1]["assigned_sessions"]["M"]["side"],None)
        noisy=[take(f"N{i}",1+i%2,tilt=rng.normal(0,.3)) for i in range(6) for _ in range(4)]
        self.assertFalse(infer_sides(noisy+unknown)[1]["accepted"])

    def test_prompt_banks_are_unique_balanced_and_avoid_accent_variable_words(self):
        from subvox_ml.prompts import generate_grid,generate_frames,SLOTS
        grid=generate_grid()
        self.assertEqual(grid,generate_grid());self.assertEqual(len(set(grid)),len(grid))
        for word in (w for slot in SLOTS.values() for w in slot):
            self.assertGreaterEqual(sum(f" {word} " in f" {t.lower().rstrip('.')} " for t in grid),4)
        frames=generate_frames()
        self.assertTrue(all(t.startswith("Say ") and t.endswith(" again.") for t,*_ in frames))
        # Post-vocalic r varies with accent; none of the slot or frame words may end a vowel with it.
        self.assertFalse([w for w in [w for s_ in SLOTS.values() for w in s_]+[f[3] for f in frames] if w.endswith("r") or "ar" in w or "or" in w])

    def test_signal_quality_separates_jaw_motion_from_nodding(self):
        from subvox_ml.rhythm import signal_quality
        t=np.arange(0,4,1/50);rng=np.random.default_rng(0)
        def clip(jaw,nod):
            raw=np.zeros((len(t),9));raw[:,3]=jaw*np.sin(2*np.pi*4*t)+nod*np.sin(2*np.pi*.5*t)+rng.normal(0,.01,len(t));raw[:,8]=-1
            return raw
        rest=clip(0,0)
        talking=signal_quality(clip(1,0),rest);nodding=signal_quality(clip(1,1),rest)
        self.assertGreater(talking["jaw_snr"],100)
        self.assertGreater(nodding["contamination"],10*talking["contamination"])
        self.assertLess(abs(nodding["jaw_snr"]/talking["jaw_snr"]-1),.1)

    def test_reevaluation_cannot_replace_a_frozen_report(self):
        from subvox_ml.cli import evaluate_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/"development-report.json"
            atomic_json(report,dict(frozen=True));before=report.read_bytes()
            with self.assertRaisesRegex(ValueError,"preserve the frozen report"):evaluate_model(root,root/"unused-dataset",0)
            self.assertEqual(report.read_bytes(),before)


if __name__=="__main__":unittest.main()
