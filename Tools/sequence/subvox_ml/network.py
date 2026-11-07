"""Small CPU Conv/GRU CTC models, fit separately for speech and silent mouthing."""
from __future__ import annotations
import copy
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import features, words, digest, atomic_json, FEATURE_NAMES
from .decoding import edit_alignment, without_separator, SEPARATOR

# Results are deterministic for a given thread count; it's recorded with each model.
THREADS = min(4, os.cpu_count() or 1)


class MotionCTC(nn.Module):
    def __init__(self, vocabulary, hidden=32, teacher=False):
        super().__init__()
        self.convolution = nn.Sequential(nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,3,padding=1),nn.GELU())
        self.encoder = nn.GRU(32,hidden,batch_first=True,bidirectional=True)
        self.phones = nn.Linear(hidden*2,vocabulary)
        self.next_phone = nn.Linear(hidden*2,vocabulary)
        self.activity = nn.Linear(hidden*2,1)
        self.camera = nn.Linear(hidden*2,3) if teacher else None

    def forward(self, frames, lengths):
        x = self.convolution(frames.transpose(1,2)).transpose(1,2)
        packed = nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_ = self.encoder(packed)
        encoded,_ = nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=frames.shape[1])
        return self.phones(encoded), self.next_phone(encoded), self.activity(encoded).squeeze(-1),self.camera(encoded) if self.camera is not None else None


def fits(path, frames):
    """CTC needs a frame per phone plus a blank between repeated phones."""
    return len(path)+sum(x==y for x,y in zip(path,path[1:])) <= frames


def batch_ctc(log_probs, lengths, label_paths):
    """Per-row CTC losses for a padded batch, every pronunciation variant in one CTCLoss call.
    Returns (marginal loss per row, cost of each row's variants) so callers can pick a variant."""
    rows,targets,target_lengths,input_lengths=[],[],[],[]
    for i,paths in enumerate(label_paths):
        valid=[q for q in paths if fits(q,int(lengths[i]))]
        if not valid:raise ValueError("No CTC path can fit this recording")
        for q in valid:
            rows.append(i);targets.extend(q);target_lengths.append(len(q));input_lengths.append(int(lengths[i]))
    index=torch.tensor(rows,dtype=torch.long)
    costs=nn.functional.ctc_loss(log_probs.transpose(0,1).index_select(1,index),torch.tensor(targets,dtype=torch.long),
                                 torch.tensor(input_lengths,dtype=torch.long),torch.tensor(target_lengths,dtype=torch.long),
                                 blank=0,reduction="none",zero_infinity=False)
    marginals=[];variants=[]
    for i in range(len(label_paths)):
        mine=costs[index==i];k=len(mine)
        lengths_i=[n for r,n in zip(rows,target_lengths) if r==i]
        # Marginalise pronunciation alternatives, not an arbitrarily chosen accent variant.
        denominator=max(1,sum(lengths_i)/k) if any(lengths_i) else float(lengths[i])
        marginals.append((-torch.logsumexp(-mine,dim=0)+math.log(k))/denominator)
        variants.append([q for q in label_paths[i] if fits(q,int(lengths[i]))])
    return torch.stack(marginals),[(v,costs[index==i].detach()) for i,v in enumerate(variants)]


def ctc_variant_loss(log_probs, paths):
    return batch_ctc(log_probs[None],torch.tensor([len(log_probs)]),[paths])[0][0]


@torch.no_grad()
def next_phone_targets_batch(log_probs, lengths, targets, vocabulary):
    """CTC forward/backward state posteriors give SOFT next-phone targets for a padded batch
    (one loop over time, vectorised across rows and states). Not measured timing."""
    B,T,_=log_probs.shape
    S=2*max(len(t) for t in targets)+1
    states=torch.zeros(B,S,dtype=torch.long);valid=torch.zeros(B,S,dtype=torch.bool)
    for b,t in enumerate(targets):
        for i,token in enumerate(t):states[b,2*i+1]=token
        valid[b,:2*len(t)+1]=True
    big=-1e30
    emission=log_probs.detach().gather(2,states[:,None,:].expand(B,T,S)).masked_fill(~valid[:,None,:],big)
    skip=torch.zeros(B,S,dtype=torch.bool)
    skip[:,2:]=(states[:,2:]!=0)&(states[:,2:]!=states[:,:-2])
    def shift(x,n):return torch.cat([torch.full((B,n),big),x[:,:-n]],1)
    def unshift(x,n):return torch.cat([x[:,n:],torch.full((B,n),big)],1)
    alpha=torch.full((B,T,S),big)
    alpha[:,0,0]=emission[:,0,0];alpha[:,0,1:2]=emission[:,0,1:2]
    for t in range(1,T):
        prev=alpha[:,t-1]
        alpha[:,t]=torch.logsumexp(torch.stack([prev,shift(prev,1),torch.where(skip,shift(prev,2),big)]),0)+emission[:,t]
    last=lengths-1
    final=torch.full((B,S),big)
    ends=torch.tensor([2*len(t) for t in targets])
    final[torch.arange(B),ends]=0
    has=torch.tensor([len(t)>0 for t in targets])
    final[torch.arange(B)[has],(ends-1)[has]]=0
    beta=torch.full((B,T,S),big)
    can_skip=torch.cat([skip[:,2:],torch.zeros(B,2,dtype=torch.bool)],1)
    for t in range(T-1,-1,-1):
        if t<T-1:
            following=beta[:,t+1]+emission[:,t+1]
            computed=torch.logsumexp(torch.stack([following,unshift(following,1),torch.where(can_skip,unshift(following,2),big)]),0)
        else:
            computed=torch.full((B,S),big)
        at_end=(last==t)[:,None]
        beta[:,t]=torch.where(at_end,final,torch.where((t<last)[:,None],computed,big))
    logZ=torch.logsumexp(alpha[torch.arange(B),last]+final,dim=1)
    posterior=(alpha+beta-logZ[:,None,None]).exp().clamp(0,1)
    posterior=posterior*(torch.arange(T)[None,:]<lengths[:,None])[:,:,None]
    successor=torch.zeros(B,S,vocabulary)
    for b,t in enumerate(targets):
        for i in range(len(t)-1):successor[b,2*i+1,t[i+1]]=1
    soft=torch.einsum("bts,bsv->btv",posterior,successor)
    mass=soft.sum(-1)
    return soft/mass.clamp_min(1e-8)[...,None],mass


def next_phone_targets(log_probs, target):
    soft,mass=next_phone_targets_batch(log_probs[None],torch.tensor([len(log_probs)]),[target],log_probs.shape[-1])
    return soft[0],mass[0]


def augment(raw, rng, minimum_frames=25):
    """Train-only perturbations of actual observations; no anatomy or generated speech."""
    raw=raw.copy()
    speed=rng.uniform(.85,1.15)
    grid=np.linspace(0,len(raw)-1,max(25,minimum_frames,round(len(raw)*speed)))
    raw=np.stack([np.interp(grid,np.arange(len(raw)),raw[:,k]) for k in range(9)],axis=1).astype(np.float32)
    # Apply the same small sensor rotation to acceleration, rotation and gravity.
    axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
    angle=rng.uniform(-math.pi/36,math.pi/36)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    rotation=np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
    for start in (0,3,6): raw[:,start:start+3]=raw[:,start:start+3]@rotation.T
    return raw


def standardise(raw, normalisation, source, trained_sources, rng=None,minimum_frames=25):
    value=features(augment(raw,rng,minimum_frames) if rng is not None else raw)
    z=(value-np.asarray(normalisation["mean"]))/np.asarray(normalisation["scale"])
    # Unseen or unspecified side: retain invariant features/derivatives; do not mirror axes.
    fallback=source not in trained_sources or source==-1
    if fallback or (rng is not None and rng.random()<.25):
        z[:,:9]=0;z[:,16:25]=0
    if rng is not None:
        z+=rng.normal(0,.02,size=z.shape)
        for _ in range(2):
            start=int(rng.integers(0,len(z)));z[start:start+int(rng.integers(1,6))]=0
        channels=rng.choice(32,size=int(rng.integers(0,5)),replace=False);z[:,channels]=0
    return torch.tensor(z,dtype=torch.float32),fallback


def batch(rows, norm, sources, rng=None):
    frames=[]
    for row in rows:
        possible=[len(p)+sum(a==b for a,b in zip(p,p[1:])) for p in row.get("label_paths",[[]])]
        minimum=max((n for n in possible if n<=len(row["raw"])),default=25)
        frames.append(standardise(np.asarray(row["raw"]),norm,row["quality"]["source"],sources,rng,minimum)[0])
    lengths=torch.tensor([len(x) for x in frames],dtype=torch.long)
    return nn.utils.rnn.pad_sequence(frames,batch_first=True),lengths


def greedy_phones(log_probs):
    """Best-path CTC collapse without the lexicon: what the motion model alone emits."""
    best=log_probs.argmax(-1).tolist();output=[];prior=None
    for token in best:
        if token!=prior and token!=0:output.append(token)
        prior=token
    return output


def evidence(log_probs):
    """Share of frames the model spends off blank: low for rest and head motion it rejects."""
    return float(1-np.exp(np.asarray(log_probs)[:,0]).mean())


def fit(training, validation, language, output, config):
    """With `validation`, keeps the epoch with the lowest development phoneme error (ties by CTC
    loss). Development CTC loss alone is lowest early, while the model still emits only blanks.
    Training runs at least `min_epochs`, and patience only counts once phoneme error has dropped
    below 1.0: while every epoch scores 1.0 there is no evidence that stopping is right.
    Without `validation` (the final refit), trains exactly `epochs` epochs."""
    output=Path(output)
    if output.exists(): raise ValueError("Model directory exists; frozen models are never overwritten")
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)
    training=[dict(r) for r in training if r["condition"]==config["condition"]]
    validation=[dict(r) for r in validation if r["condition"]==config["condition"]]
    if not training: raise ValueError("No training recordings for this condition")
    # No word-separator output: the lexical decoder places word boundaries.
    tokens=["<blank>"]+sorted({p["ipa"] for p in language["inventory"]})
    index={p:i for i,p in enumerate(tokens)}
    for row in training+validation: row["label_paths"]=[[index[t] for t in p if t!=SEPARATOR] for p in row["targets"]]
    all_frames=np.concatenate([features(np.asarray(r["raw"])) for r in training])
    norm=dict(mean=all_frames.mean(0).tolist(),scale=np.maximum(all_frames.std(0),1e-4).tolist(),fitted_recordings=[r["id"] for r in training])
    sources=sorted({r["quality"]["source"] for r in training})
    teacher=config.get("teacher_aux",False)
    model=MotionCTC(len(tokens),teacher=teacher)
    teacher_norm=None
    if teacher:
        values=np.concatenate([np.asarray(r["camera_teacher"])[np.asarray(r["camera_weight"])>0] for r in training if "camera_weight" in r])
        if len(values)<100:raise ValueError("Not enough training-only camera teacher frames")
        teacher_norm=dict(mean=values.mean(0).tolist(),scale=values.std(0).clip(.01).tolist())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    rng=np.random.default_rng(seed);history=[];best=(math.inf,math.inf);best_state=None;best_epoch=0
    patience=config.get("patience",15);min_epochs=config.get("min_epochs",config["epochs"]//2);size=config.get("batch_size",8)
    def objective(subset, train):
        total=0.0;count=0;phone_errors=0;phone_count=0;blank=0.0;frames=0;sample=None
        order=rng.permutation(len(subset)) if train else np.arange(len(subset))
        for begin in range(0,len(order),size):
            group=[subset[i] for i in order[begin:begin+size]]
            x,lengths=batch(group,norm,sources,rng if train and config["augmentation"] else None)
            logits,next_logits,activity,camera=model(x,lengths);lp=logits.log_softmax(-1)
            losses,variants=batch_ctc(lp,lengths,[r["label_paths"] for r in group])
            mask=(torch.arange(lp.shape[1])[None,:]<lengths[:,None]).float()
            if train and config["transitions"]=="next_phone":
                # Choose a variant by posterior CTC evidence, never word-duration timing.
                chosen=[paths[int(costs.argmin())] if row["activity"] else [] for row,(paths,costs) in zip(group,variants)]
                if any(chosen):
                    soft,mass=next_phone_targets_batch(lp,lengths,chosen,len(tokens))
                    auxiliary=-(soft*next_logits.log_softmax(-1)).sum(-1)
                    losses=losses+.1*(auxiliary*mass).sum(1)/mass.sum(1).clamp_min(1)
            if train and config["activity_aux"]:
                mean=(activity*mask).sum(1)/lengths
                losses=losses+.1*nn.functional.binary_cross_entropy_with_logits(mean,torch.tensor([float(r["activity"]) for r in group]),reduction="none")
            if train and teacher:
                if config["augmentation"]:raise ValueError("Teacher time warping is not implemented; teacher comparison must use matched original timelines")
                for i,row in enumerate(group):
                    if "camera_weight" not in row:continue
                    L=int(lengths[i]);weight=torch.tensor(row["camera_weight"],dtype=torch.float32)
                    target=(np.asarray(row["camera_teacher"])-teacher_norm["mean"])/teacher_norm["scale"]
                    errors=(camera[i,:L]-torch.tensor(target,dtype=torch.float32)).square().mean(-1)
                    losses=losses+.1*(errors*weight).sum()/weight.sum().clamp_min(1)*(torch.arange(len(group))==i)
            if not torch.isfinite(losses).all(): raise ValueError("Non-finite CTC loss; no invalid checkpoint saved")
            if not train:
                for i,row in enumerate(group):
                    if not row["activity"]:continue
                    L=int(lengths[i]);guess=greedy_phones(lp[i,:L])
                    errors,reference=min((edit_alignment(ref,guess)[0],ref) for ref in row["label_paths"])
                    phone_errors+=errors;phone_count+=len(reference)
                    blank+=float(lp[i,:L,0].exp().sum());frames+=L
                    if sample is None:sample=" ".join(tokens[t] for t in guess) or "(blank)"
            value=losses.mean()
            if train:
                optimizer.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(value.detach())*len(group);count+=len(group)
        return total/max(1,count),(phone_errors/phone_count if phone_count else math.inf),(blank/frames if frames else None),sample
    started=time.time();stopped_early=False
    for epoch in range(config["epochs"]):
        model.train();train_loss=objective(training,True)[0]
        record=dict(epoch=epoch+1,training_ctc=train_loss)
        if validation:
            model.eval()
            with torch.no_grad(): val_loss,per,blank,sample=objective(validation,False)
            record.update(development_ctc=val_loss,development_greedy_per=per,development_blank_fraction=blank)
            if (per,val_loss)<best:
                best=(per,val_loss);best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        history.append(record)
        if epoch==0 or (epoch+1)%10==0:
            detail=f', dev {val_loss:.3f}, greedy PER {per:.2f}, blank {blank or 0:.0%}, e.g. "{sample}"' if validation else ''
            print(f'{config["condition"]} {config["name"]} epoch {epoch+1}/{config["epochs"]}: train {train_loss:.3f}{detail}',flush=True)
        if validation and epoch+1>=min_epochs and best[0]<1 and epoch+1-best_epoch>=patience:
            stopped_early=True;break
    if not validation:
        best_state=copy.deepcopy(model.state_dict());best_epoch=len(history)
    # Epoch count a refit should use: the selected epoch only once it beat 1.0; otherwise
    # nothing was learned to select, so the full run is the honest count.
    refit_epochs=best_epoch if not validation or best[0]<1 else len(history)
    model.load_state_dict(best_state);model.eval();output.mkdir(parents=True)
    # Unaugmented training negatives include relaxed baselines and any explicitly
    # collected non-speech movements; never use validation/test statistics.
    with torch.no_grad():
        rest=[r for r in training if not r["activity"]]
        scores=[]
        for begin in range(0,len(rest),size):
            x,lengths=batch(rest[begin:begin+size],norm,sources,None)
            lp=model(x,lengths)[0].log_softmax(-1)
            scores+=[evidence(lp[i,:int(lengths[i])].numpy()) for i in range(len(lengths))]
    rejection=dict(evidence_threshold=float(np.quantile(scores,.95)),fitted_on=f"{len(scores)} training non-speech clips (95th percentile)") if len(scores)>=5 else None
    coverage=Counter();diphones=Counter()
    for row in training:
        for path in row["targets"]:
            phones=[p for p in path if p!=SEPARATOR]
            for phone in phones: coverage[phone]+=1/len(row["targets"])
            for a,b in zip(phones,phones[1:]):diphones[a+" → "+b]+=1/len(row["targets"])
    checkpoint=dict(state_dict=best_state,tokens=tokens,normalisation=norm,trained_sources=sources,config=config,rejection=rejection,
                    lm_training_text=[words(r["text"]) for r in training if r["activity"]],coverage=dict(coverage),diphones=dict(diphones),teacher_normalisation=teacher_norm)
    torch.save(checkpoint,output/"weights.pt")
    manifest=dict(schema=2,family="conv32-bidirectional-gru32-ctc",offline_context="Conv +/-60 ms then bidirectional GRU; completed clips only",
                  config=config,training_ids=[r["id"] for r in training],development_ids=[r["id"] for r in validation],
                  training_sessions=sorted({r["session"] for r in training}),development_sessions=sorted({r["session"] for r in validation}),
                  training_groups=sorted({g for r in training for g in r["groups"]}),development_groups=sorted({g for r in validation for g in r["groups"]}),
                  best_epoch=best_epoch,refit_epochs=refit_epochs,stopped_early=stopped_early,rejection=rejection,
                  checkpoint_selection=(f"lowest development greedy phoneme error, ties by CTC loss; at least {min_epochs} epochs; patience {patience} once below 1.0; never final-test results" if validation
                                        else "final refit on every fold's recordings for a fixed epoch count chosen by cross-validation"),history=history,
                  torch_version=str(torch.__version__),threads=THREADS,numpy_version=np.__version__,parameters=sum(p.numel() for p in model.parameters()),
                  fit_seconds=time.time()-started,created_unix=time.time(),feature_names=FEATURE_NAMES,
                  coverage=dict(coverage),diphone_coverage=dict(diphones),normalisation=norm,teacher_normalisation=teacher_norm,
                  reserved_final_groups=config.get("reserved_final_groups",[]))
    import hashlib
    manifest["weights_sha256"]=hashlib.sha256((output/"weights.pt").read_bytes()).hexdigest()
    manifest["language_hash"]=digest(language)
    atomic_json(output/"language.json",language);atomic_json(output/"model.json",manifest)
    return manifest


def load_model(path):
    import hashlib,json
    path=Path(path);manifest=json.loads((path/"model.json").read_text())
    if hashlib.sha256((path/"weights.pt").read_bytes()).hexdigest()!=manifest["weights_sha256"]:
        raise ValueError("Frozen weights fingerprint mismatch")
    language=json.loads((path/"language.json").read_text())
    if digest(language)!=manifest["language_hash"]:raise ValueError("Frozen pronunciation lexicon mismatch")
    checkpoint=torch.load(path/"weights.pt",map_location="cpu",weights_only=True)
    if manifest.get("feature_names")!=FEATURE_NAMES or checkpoint["config"]!=manifest["config"] or checkpoint["normalisation"]!=manifest["normalisation"]:
        raise ValueError("Incompatible feature schema or altered frozen configuration/normalisation")
    # Early local checkpoints predate embedding the reservation. Recover it from
    # their immutable dataset manifest in memory; never rewrite frozen artifacts.
    if not manifest.get("reserved_final_groups"):
        comparison_path=path.parent/"comparison.json"
        if comparison_path.exists():
            comparison=json.loads(comparison_path.read_text())
            dataset_manifest=json.loads((Path(comparison["dataset"])/"manifest.json").read_text())
            if dataset_manifest["examples_hash"]!=checkpoint["config"]["dataset_hash"]:
                raise ValueError("Frozen dataset identity mismatch")
            manifest={**manifest,"reserved_final_groups":dataset_manifest["reserved_final_groups"]}
    model=MotionCTC(len(checkpoint["tokens"]),teacher=checkpoint["config"].get("teacher_aux",False));model.load_state_dict(checkpoint["state_dict"]);model.eval()
    torch.set_num_threads(THREADS)
    return model,checkpoint,manifest,language


@torch.no_grad()
def emission(model,checkpoint,row):
    """Phone and next-phone log-probabilities with any legacy "|" column folded away."""
    x,fallback=standardise(np.asarray(row["raw"]),checkpoint["normalisation"],row["quality"]["source"],checkpoint["trained_sources"])
    phones,following,activity,_=model(x[None],torch.tensor([len(x)]))
    lp,tokens=without_separator(phones[0].log_softmax(-1).numpy(),checkpoint["tokens"])
    following=following[0].log_softmax(-1).numpy()
    if SEPARATOR in checkpoint["tokens"]:
        following=np.delete(following,checkpoint["tokens"].index(SEPARATOR),axis=1)
    return lp,tokens,following,float(activity[0].mean().sigmoid()),fallback
