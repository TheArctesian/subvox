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
from .data import features, words, digest, atomic_json, FEATURE_NAMES, FEATURE_WIDTH, DIRECTIONAL, quaternion_multiply
from . import camera as camera_input
from .decoding import edit_alignment, without_separator, SEPARATOR

# Results are deterministic for a given thread count; it's recorded with each model.
THREADS = min(4, os.cpu_count() or 1)


class LocalEncoder(nn.Module):
    """Residual dilated convolutions (dilation 1, 2, 4, 8): each output frame sees about ±0.36 s,
    enough for a syllable and its neighbours but not a whole sentence, so it can't memorise
    sentences by their global shape the way a bidirectional GRU can."""
    def __init__(self, inputs, width):
        super().__init__()
        self.project = nn.Conv1d(inputs,width,1)
        # Depthwise-separable blocks keep the parameter count near the GRU's.
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=d,dilation=d,groups=width),nn.Conv1d(width,width,1))
                                     for d in (1,2,4,8)])

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1])[None,:]<lengths[:,None]).float()[:,None,:]
        h = self.project(x.transpose(1,2))*mask
        for block in self.blocks:
            h = (h+nn.functional.gelu(block(h)))*mask
        return h.transpose(1,2)


class MotionCTC(nn.Module):
    def __init__(self, vocabulary, hidden=32, teacher=False, coarse=0, conditions=0, inputs=FEATURE_WIDTH, encoder="gru"):
        super().__init__()
        self.convolution = nn.Sequential(nn.Conv1d(inputs,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,3,padding=1),nn.GELU())
        # A shared spoken/silent model tells the GRU which condition it reads and gives each
        # condition its own blank bias: same jaw kinematics, different amplitude and timing.
        self.condition = nn.Embedding(conditions,4) if conditions else None
        self.blank_bias = nn.Parameter(torch.zeros(conditions)) if conditions else None
        self.local = encoder=="local"
        self.encoder = LocalEncoder(32+(4 if conditions else 0),hidden*2) if self.local else nn.GRU(32+(4 if conditions else 0),hidden,batch_first=True,bidirectional=True)
        self.phones = nn.Linear(hidden*2,vocabulary)
        self.next_phone = nn.Linear(hidden*2,vocabulary)
        self.activity = nn.Linear(hidden*2,1)
        self.camera = nn.Linear(hidden*2,3) if teacher else None
        # Coarse observable classes (+ blank) read from the conv layer, below the GRU: the
        # low layers learn the jaw/lip classes the sensor carries, the top refines them.
        self.coarse = nn.Linear(32,coarse+1) if coarse else None

    def forward(self, frames, lengths, with_coarse=False, conditions=None):
        # Zero padded frames after every stem layer, so a clip's edge frames don't depend on how
        # much padding its batch happened to add.
        mask = (torch.arange(frames.shape[1])[None,:]<lengths[:,None]).float()[:,None,:]
        x = frames.transpose(1,2)*mask
        for layer in self.convolution:
            x = layer(x)*mask if isinstance(layer,nn.GELU) else layer(x)
        x = x.transpose(1,2)
        inputs = x
        if self.condition is not None:
            inputs = torch.cat([x,self.condition(conditions)[:,None,:].expand(-1,x.shape[1],-1)],-1)
        if self.local:
            encoded = self.encoder(inputs,lengths)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(inputs,lengths.cpu(),batch_first=True,enforce_sorted=False)
            encoded,_ = self.encoder(packed)
            encoded,_ = nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=frames.shape[1])
        phones = self.phones(encoded)
        if self.blank_bias is not None:
            phones = torch.cat([phones[...,:1]+self.blank_bias[conditions][:,None,None],phones[...,1:]],-1)
        outputs = (phones, self.next_phone(encoded), self.activity(encoded).squeeze(-1),self.camera(encoded) if self.camera is not None else None)
        return (*outputs, self.coarse(x)) if with_coarse else outputs


CONTRAST_MARGIN = .5   # nats per phone
CONTRAST_WEIGHT = .2


def coarse_weight(epoch, epochs):
    """Coarse-CTC weight, fixed before training: 1.0 for the first quarter, annealed linearly
    to 0.3 by halfway, then held at 0.3."""
    progress=epoch/max(1,epochs)
    if progress<=.25:return 1.0
    if progress>=.5:return .3
    return 1.0-.7*(progress-.25)/.25


def coarse_units(tokens, language):
    """Class index per token (blank stays 0) under the pre-registered alphabet rule: 'observable'
    unless it leaves fewer than half the uniquely identifiable words 'phone' does."""
    from .articulatory import ALPHABETS, cohort_entropy
    alphabet="observable"
    if cohort_entropy(language,"observable")["unique_word_share"]<.5*cohort_entropy(language,"phone")["unique_word_share"]:
        alphabet="jaw_round_labial"
    labels=[str(ALPHABETS[alphabet](t)) for t in tokens[1:]]
    classes=sorted(set(labels))
    return alphabet,[0]+[classes.index(l)+1 for l in labels],len(classes)


CONDITIONS = ("Normal speech", "Silent mouthing")


def conditions_of(rows):
    return torch.tensor([CONDITIONS.index(r["condition"]) for r in rows],dtype=torch.long)


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


def augment(raw, rng, minimum_frames=25, extra=None):
    """Train-only perturbations of actual observations; no anatomy or generated speech.
    `extra` (camera channels on the same grid) gets the same time warp and is returned alongside."""
    raw=raw.copy()
    speed=rng.uniform(.85,1.15)
    grid=np.linspace(0,len(raw)-1,max(25,minimum_frames,round(len(raw)*speed)))
    raw=np.stack([np.interp(grid,np.arange(len(raw)),raw[:,k]) for k in range(raw.shape[1])],axis=1).astype(np.float32)
    if extra is not None:
        extra=np.stack([np.interp(grid,np.arange(len(extra)),extra[:,k]) for k in range(extra.shape[1])],axis=1).astype(np.float32)
    if raw.shape[1]>=13:
        raw[:,9:13]/=np.linalg.norm(raw[:,9:13],axis=1,keepdims=True).clip(1e-9)
    # Apply the same small sensor rotation to acceleration, rotation and gravity (and attitude).
    axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
    angle=rng.uniform(-math.pi/36,math.pi/36)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    rotation=np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
    for start in (0,3,6): raw[:,start:start+3]=raw[:,start:start+3]@rotation.T
    if raw.shape[1]>=13:
        # A sensor-frame rotation R turns attitude q into q ⊗ r⁻¹, so jaw-angle axes rotate with the rest.
        inverse=np.array([math.cos(angle/2),*(-math.sin(angle/2)*axis)])
        raw[:,9:13]=quaternion_multiply(raw[:,9:13],inverse).astype(np.float32)
    return raw if extra is None else (raw,extra)


BASELINE_CLIP = 50.0
GRAVITY = [6, 7, 8]


def attach_baselines(rows, pool):
    """Give each row its instructed relaxed window (the "<id>/rest" sibling) as `baseline_raw`;
    a rest clip is its own baseline. Rows without one fall back to their own statistics."""
    rest={r["id"]:r["raw"] for r in pool if not r["activity"]}
    return [dict(r,baseline_raw=rest.get(r["id"]+"/rest") if r["activity"] else r["raw"]) for r in rows]


def columns_of(values, columns):
    """A model trained on an older, narrower feature set reads only its own columns."""
    return values if columns is None else values[:,columns]


def within(indices, columns):
    return indices if columns is None else [j for j,c in enumerate(columns) if c in indices]


def baseline_stats(row, floor, columns=None):
    """Per-channel median and MAD of the row's rest window (or, lacking one, of the clip itself),
    the MAD floored so a quiet recording can't blow the scale up."""
    reference=row.get("baseline_raw")
    values=columns_of(features(np.asarray(reference if reference is not None else row["raw"])),columns)
    median=np.median(values,axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(values-median),axis=0),np.asarray(floor))
    # Gravity is posture, nearly constant at rest: centre it on this recording's orientation but
    # keep its global scale, or a few degrees of head tilt would read as huge motion.
    scale[within(GRAVITY,columns)]=1.0
    return median,scale,reference is None


def fit_length(values, frames):
    """Stretch per-frame channels to `frames` rows (a derived row may borrow another clip's motion)."""
    values=np.asarray(values,dtype=np.float32)
    if len(values)==frames:return values
    if len(values)<2:return np.zeros((frames,values.shape[1] if values.ndim==2 else camera_input.RAW_WIDTH),np.float32)
    grid=np.linspace(0,len(values)-1,frames)
    return np.stack([np.interp(grid,np.arange(len(values)),values[:,k]) for k in range(values.shape[1])],1).astype(np.float32)


def camera_block(camera, frames, normalisation, rng=None):
    """Standardised camera features; invalid frames read as 0 with their mask columns 0."""
    values=camera_input.features(fit_length(camera if camera is not None else np.zeros((frames,camera_input.RAW_WIDTH)),frames))
    valid=camera_input.valid_columns(values)
    stats=normalisation["camera"]
    z=(values-np.asarray(stats["mean"]))/np.asarray(stats["scale"])
    z[~valid]=0
    if rng is not None:
        noisy=valid.copy();noisy[:,-2:]=False
        z[noisy]+=rng.normal(0,.02,size=int(noisy.sum()))
        # Modality dropout: a fused model must still work when the face is lost or the camera is off.
        if normalisation.get("modalities")=="fused" and rng.random()<CAMERA_DROPOUT:z[:]=0
    return z


CAMERA_DROPOUT = .3


def standardise(raw, normalisation, source, trained_sources, rng=None,minimum_frames=25,baseline=None,camera=None):
    """`baseline` (median, scale) puts features in units of this recording's own resting noise
    before the global standardisation: insertion depth and seal change gain and offset per fit.
    `normalisation["modalities"]` picks the inputs: "imu" (default), "camera", or "fused" (both)."""
    columns=normalisation.get("columns")
    modalities=normalisation.get("modalities","imu")
    if modalities!="imu":
        camera=fit_length(camera if camera is not None else np.zeros((len(raw),camera_input.RAW_WIDTH)),len(raw))
        if rng is not None:raw,camera=augment(raw,rng,minimum_frames,extra=camera)
    elif rng is not None:
        raw=augment(raw,rng,minimum_frames)
    value=columns_of(features(raw),columns)
    if baseline is not None:
        # Clipped: a channel nearly flat at rest would otherwise swamp the rest at ±thousands.
        value=np.clip((value-baseline[0])/baseline[1],-BASELINE_CLIP,BASELINE_CLIP)
    z=(value-np.asarray(normalisation["mean"]))/np.asarray(normalisation["scale"])
    # Unseen or unspecified side: retain invariant features/derivatives; do not mirror axes.
    fallback=source not in trained_sources or source==-1
    if fallback or (rng is not None and rng.random()<.25):
        z[:,within(DIRECTIONAL,columns)]=0
    if rng is not None:
        z+=rng.normal(0,.02,size=z.shape)
        for _ in range(2):
            start=int(rng.integers(0,len(z)));z[start:start+int(rng.integers(1,6))]=0
        channels=rng.choice(z.shape[1],size=int(rng.integers(0,5)),replace=False);z[:,channels]=0
    if modalities!="imu":
        zc=camera_block(camera,len(z),normalisation,rng)
        z=zc if modalities=="camera" else np.concatenate([z,zc],1)
    return torch.tensor(z,dtype=torch.float32),fallback


def batch(rows, norm, sources, rng=None):
    frames=[]
    for row in rows:
        possible=[len(p)+sum(a==b for a,b in zip(p,p[1:])) for p in row.get("label_paths",[[]])]
        minimum=max((n for n in possible if n<=len(row["raw"])),default=25)
        baseline=baseline_stats(row,norm["baseline"]["floor"],norm.get("columns"))[:2] if norm.get("baseline") else None
        camera=np.asarray(row["camera"]) if norm.get("modalities","imu")!="imu" and row.get("camera") else None
        frames.append(standardise(np.asarray(row["raw"]),norm,row["quality"]["source"],sources,rng,minimum,baseline,camera)[0])
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


def input_width(imu_width, modalities):
    return (imu_width if modalities!="camera" else 0)+(camera_input.FEATURE_WIDTH if modalities!="imu" else 0)


def camera_normalisation(rows):
    """Training-only mean/scale per camera feature, over the frames where that feature is valid."""
    blocks=[camera_input.features(np.asarray(r["camera"])) for r in rows if r.get("camera")]
    if not blocks:raise ValueError("No training recordings have camera landmarks; record paired sentences with the camera on")
    values=np.concatenate(blocks);valid=camera_input.valid_columns(values)
    if not valid[:,0].any():raise ValueError("No valid camera mouth frames in training")
    count=valid.sum(0).clip(1)
    mean=np.where(valid,values,0).sum(0)/count
    scale=np.sqrt(np.where(valid,(values-mean)**2,0).sum(0)/count).clip(1e-3)
    mean[-2:]=0;scale[-2:]=1
    return dict(mean=mean.tolist(),scale=scale.tolist(),names=camera_input.FEATURE_NAMES,
                frames=int(valid[:,0].sum()),face_frames=int(valid[:,camera_input.FACE.start].sum()))


def fit(training, validation, language, output, config):
    """With `validation`, keeps the epoch with the lowest development phoneme error (ties by CTC
    loss). Development CTC loss alone is lowest early, while the model still emits only blanks.
    The last epoch is also saved (weights-final-epoch.pt): its budget was fixed before training,
    so its scores carry no selection bias, unlike the selected checkpoint's.
    Early stopping is off unless `patience` is set; it then waits at least `min_epochs` and only
    counts once phoneme error has dropped below 1.0.
    Without `validation` (the final refit), trains exactly `epochs` epochs."""
    output=Path(output)
    if output.exists(): raise ValueError("Model directory exists; frozen models are never overwritten")
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(THREADS);torch.use_deterministic_algorithms(True)
    shared=config["condition"]=="shared"
    training=[dict(r) for r in training if shared or r["condition"]==config["condition"]]
    validation=[dict(r) for r in validation if shared or r["condition"]==config["condition"]]
    if not training: raise ValueError("No training recordings for this condition")
    # No word-separator output: the lexical decoder places word boundaries.
    tokens=["<blank>"]+sorted({p["ipa"] for p in language["inventory"]})
    index={p:i for i,p in enumerate(tokens)}
    for row in training+validation: row["label_paths"]=[[index[t] for t in p if t!=SEPARATOR] for p in row["targets"]]
    # An ablation can drop feature groups; the model then reads only these columns, as a legacy model would.
    columns=[i for i,n in enumerate(FEATURE_NAMES) if "att_" not in n] if config.get("drop_attitude") else None
    all_frames=np.concatenate([columns_of(features(np.asarray(r["raw"])),columns) for r in training])
    baseline=None
    if config.get("normalisation")=="rest_baseline":
        training,validation=attach_baselines(training,training),attach_baselines(validation,validation)
        # Floor at a quarter of the training spread: shrinks a near-constant rest window toward the global scale.
        floor=(.25*np.maximum(all_frames.std(0),1e-4)).tolist()
        baseline=dict(floor=floor,method="rest-window median/MAD per recording; clip CMVN when no rest window")
        all_frames=np.concatenate([np.clip((columns_of(features(np.asarray(r["raw"])),columns)-m)/sc,-BASELINE_CLIP,BASELINE_CLIP) for r in training for m,sc,_ in [baseline_stats(r,floor,columns)]])
    norm=dict(mean=all_frames.mean(0).tolist(),scale=np.maximum(all_frames.std(0),1e-4).tolist(),fitted_recordings=[r["id"] for r in training],
              **({"baseline":baseline} if baseline else {}),**({"columns":columns} if columns else {}))
    modalities=config.get("modalities","imu")
    if modalities!="imu":
        norm["modalities"]=modalities;norm["camera"]=camera_normalisation(training)
    sources=sorted({r["quality"]["source"] for r in training})
    teacher=config.get("teacher_aux",False)
    coarse_alphabet,coarse_map,coarse_count=coarse_units(tokens,language) if config.get("coarse_aux") else (None,None,0)
    if coarse_map:
        for row in training+validation:
            row["coarse_paths"]=[list(q) for q in sorted({tuple(coarse_map[t] for t in p) for p in row["label_paths"]})]
    model=MotionCTC(len(tokens),teacher=teacher,coarse=coarse_count,conditions=len(CONDITIONS) if shared else 0,encoder=config.get("encoder","gru"),
                    inputs=input_width(len(columns) if columns else FEATURE_WIDTH,modalities))
    teacher_norm=None
    if teacher:
        values=np.concatenate([np.asarray(r["camera_teacher"])[np.asarray(r["camera_weight"])>0] for r in training if "camera_weight" in r])
        if len(values)<100:raise ValueError("Not enough training-only camera teacher frames")
        teacher_norm=dict(mean=values.mean(0).tolist(),scale=values.std(0).clip(.01).tolist())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    rng=np.random.default_rng(seed);history=[];best=(math.inf,math.inf);best_state=None;best_epoch=0
    patience=config.get("patience");min_epochs=config.get("min_epochs",config["epochs"]//2);size=config.get("batch_size",8)
    epoch_now=[0]
    rest_pool=[r for r in training if not r["activity"]] or training
    def objective(subset, train):
        total=0.0;count=0;phone_errors=0;phone_count=0;blank=0.0;frames=0;sample=None;coarse_errors=0;coarse_count_=0
        order=rng.permutation(len(subset)) if train else np.arange(len(subset))
        for begin in range(0,len(order),size):
            group=[subset[i] for i in order[begin:begin+size]]
            x,lengths=batch(group,norm,sources,rng if train and config["augmentation"] else None)
            outputs=model(x,lengths,with_coarse=bool(coarse_map),conditions=conditions_of(group))
            logits,next_logits,activity,camera=outputs[:4];lp=logits.log_softmax(-1)
            losses,variants=batch_ctc(lp,lengths,[r["label_paths"] for r in group])
            if coarse_map:
                coarse_lp=outputs[4].log_softmax(-1)
                # Repeated classes need a blank between them, so a clip can fit its phones but not its classes; skip it.
                usable=[i for i,r in enumerate(group) if any(fits(q,int(lengths[i])) for q in r["coarse_paths"])]
                if train and usable:
                    index=torch.tensor(usable)
                    coarse_losses=batch_ctc(coarse_lp.index_select(0,index),lengths[index],[group[i]["coarse_paths"] for i in usable])[0]
                    losses=losses+coarse_weight(epoch_now[0],config["epochs"])*torch.zeros(len(group)).index_add(0,index,coarse_losses)
                if not train:
                    for i in usable:
                        if not group[i]["activity"]:continue
                        guess=greedy_phones(coarse_lp[i,:int(lengths[i])])
                        errors,reference=min((edit_alignment(q,guess)[0],q) for q in group[i]["coarse_paths"])
                        coarse_errors+=errors;coarse_count_+=len(reference)
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
            if config.get("motion_contrast"):
                # The label must explain its own motion better than someone else's (a batch
                # partner resampled to this length) or a tiled rest clip: an input-ignoring
                # model pays the full margin. Margin and weight fixed before training.
                active=[i for i,r in enumerate(group) if r["activity"]]
                if active:
                    negatives=[]
                    for i in active:
                        own=np.asarray(group[i]["raw"])
                        partner=[group[j] for j in active if j!=i]
                        source=np.asarray(partner[0]["raw"]) if partner else np.asarray(rest_pool[i%len(rest_pool)]["raw"])
                        if partner:
                            grid=np.linspace(0,len(source)-1,len(own))
                            source=np.stack([np.interp(grid,np.arange(len(source)),source[:,k]) for k in range(source.shape[1])],1)
                        else:
                            source=np.resize(source,(len(own),source.shape[1]))
                        negatives.append(dict(group[i],raw=source.tolist()))
                    xn,ln=batch(negatives,norm,sources,None)
                    pos=-batch_ctc(model(*batch([dict(group[i],raw=group[i]["raw"]) for i in active],norm,sources,None),conditions=conditions_of([group[i] for i in active]))[0].log_softmax(-1),
                                   torch.tensor([len(group[i]["raw"]) for i in active]),[group[i]["label_paths"] for i in active])[0]
                    neg=-batch_ctc(model(xn,ln,conditions=conditions_of(negatives))[0].log_softmax(-1),ln,[r["label_paths"] for r in negatives])[0]
                    margin=pos-neg
                    if train:
                        hinge=CONTRAST_WEIGHT*torch.relu(CONTRAST_MARGIN-margin)
                        losses=losses+torch.zeros(len(group)).index_add(0,torch.tensor(active),hinge)
                    record_extra["training_margin" if train else "development_margin"]=float(margin.mean().detach())
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
        if not train and coarse_map:record_extra["development_coarse_error"]=coarse_errors/coarse_count_ if coarse_count_ else None
        return total/max(1,count),(phone_errors/phone_count if phone_count else math.inf),(blank/frames if frames else None),sample
    started=time.time();stopped_early=False
    record_extra={}
    for epoch in range(config["epochs"]):
        epoch_now[0]=epoch
        model.train();train_loss=objective(training,True)[0]
        record=dict(epoch=epoch+1,training_ctc=train_loss)
        if validation:
            model.eval()
            with torch.no_grad(): val_loss,per,blank,sample=objective(validation,False)
            record.update(development_ctc=val_loss,development_greedy_per=per,development_blank_fraction=blank,**record_extra)
            if (per,val_loss)<best:
                best=(per,val_loss);best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        history.append(record)
        if epoch==0 or (epoch+1)%10==0:
            detail=f', dev {val_loss:.3f}, greedy PER {per:.2f}, blank {blank or 0:.0%}, e.g. "{sample}"' if validation else ''
            print(f'{config["condition"]} {config["name"]} epoch {epoch+1}/{config["epochs"]}: train {train_loss:.3f}{detail}',flush=True)
        if validation and patience and epoch+1>=min_epochs and best[0]<1 and epoch+1-best_epoch>=patience:
            stopped_early=True;break
    final_state=copy.deepcopy(model.state_dict())
    if not validation:
        best_state=final_state;best_epoch=len(history)
    # Epoch count a refit should use: the selected epoch only once it beat 1.0; otherwise
    # nothing was learned to select, so the full run is the honest count.
    refit_epochs=best_epoch if not validation or best[0]<1 else len(history)
    output.mkdir(parents=True)
    def label_prior(state):
        """Mean posterior over unaugmented training utterances: what the model emits a priori."""
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            active=[r for r in training if r["activity"]];total=torch.zeros(len(tokens));count=0
            for begin in range(0,len(active),size):
                x,lengths=batch(active[begin:begin+size],norm,sources,None)
                lp=model(x,lengths,conditions=conditions_of(active[begin:begin+size]))[0].log_softmax(-1)
                for i in range(len(lengths)):
                    total+=lp[i,:int(lengths[i])].exp().sum(0);count+=int(lengths[i])
        return (total/max(1,count)).clamp_min(1e-6).log().tolist()
    final_prior=label_prior(final_state)
    prior=label_prior(best_state)
    # Unaugmented training negatives include relaxed baselines and any explicitly
    # collected non-speech movements; never use validation/test statistics.
    def rejection_for(state):
        """Evidence threshold (95th percentile over training rest clips) for one set of weights."""
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            rest=[r for r in training if not r["activity"]]
            scores=[]
            for begin in range(0,len(rest),size):
                x,lengths=batch(rest[begin:begin+size],norm,sources,None)
                lp=model(x,lengths,conditions=conditions_of(rest[begin:begin+size]))[0].log_softmax(-1)
                scores+=[evidence(lp[i,:int(lengths[i])].numpy()) for i in range(len(lengths))]
        if len(scores)<5:return None
        result=dict(evidence_threshold=float(np.quantile(scores,.95)),fitted_on=f"{len(scores)} training non-speech clips (95th percentile)")
        if shared:
            # One threshold per condition, from that condition's own rest clips.
            result["per_condition"]={c:float(np.quantile([v for v,r in zip(scores,rest) if r["condition"]==c],.95))
                                     for c in CONDITIONS if sum(r["condition"]==c for r in rest)>=5}
        return result
    # Each checkpoint gets its own threshold: a final-epoch model without one would accept everything.
    final_rejection=rejection_for(final_state)
    rejection=rejection_for(best_state)
    coverage=Counter();diphones=Counter()
    for row in training:
        for path in row["targets"]:
            phones=[p for p in path if p!=SEPARATOR]
            for phone in phones: coverage[phone]+=1/len(row["targets"])
            for a,b in zip(phones,phones[1:]):diphones[a+" → "+b]+=1/len(row["targets"])
    checkpoint=dict(state_dict=best_state,tokens=tokens,normalisation=norm,trained_sources=sources,config=config,rejection=rejection,
                    label_log_prior=prior,coarse_classes=coarse_count,coarse_alphabet=coarse_alphabet,coarse_map=coarse_map,lm_training_text=[words(r["text"]) for r in training if r["activity"]],coverage=dict(coverage),diphones=dict(diphones),teacher_normalisation=teacher_norm)
    torch.save(checkpoint,output/"weights.pt")
    if validation:
        torch.save(dict(checkpoint,state_dict=final_state,label_log_prior=final_prior,rejection=final_rejection),output/"weights-final-epoch.pt")
    manifest=dict(schema=2,family="conv32-bidirectional-gru32-ctc",offline_context="Conv +/-60 ms then bidirectional GRU; completed clips only",
                  config=config,training_ids=[r["id"] for r in training],development_ids=[r["id"] for r in validation],
                  training_sessions=sorted({r["session"] for r in training}),development_sessions=sorted({r["session"] for r in validation}),
                  training_groups=sorted({g for r in training for g in r["groups"]}),development_groups=sorted({g for r in validation for g in r["groups"]}),
                  best_epoch=best_epoch,refit_epochs=refit_epochs,stopped_early=stopped_early,rejection=rejection,
                  checkpoint_selection=(f"lowest development greedy phoneme error, ties by CTC loss; at least {min_epochs} epochs; patience {patience} once below 1.0; never final-test results" if validation
                                        else "final refit on every fold's recordings for a fixed epoch count chosen by cross-validation"),history=history,
                  torch_version=str(torch.__version__),threads=THREADS,numpy_version=np.__version__,parameters=sum(p.numel() for p in model.parameters()),
                  fit_seconds=time.time()-started,created_unix=time.time(),feature_names=[FEATURE_NAMES[i] for i in columns] if columns else FEATURE_NAMES,
                  coverage=dict(coverage),diphone_coverage=dict(diphones),normalisation=norm,teacher_normalisation=teacher_norm,
                  reserved_final_groups=config.get("reserved_final_groups",[]),
                  modalities=modalities,**({"camera_feature_names":camera_input.FEATURE_NAMES} if modalities!="imu" else {}))
    import hashlib
    manifest["weights_sha256"]=hashlib.sha256((output/"weights.pt").read_bytes()).hexdigest()
    if validation:
        manifest["final_epoch_weights_sha256"]=hashlib.sha256((output/"weights-final-epoch.pt").read_bytes()).hexdigest()
    manifest["language_hash"]=digest(language)
    atomic_json(output/"language.json",language);atomic_json(output/"model.json",manifest)
    return manifest


def load_model(path, which="selected"):
    """`which="final-epoch"` loads the fixed-budget last epoch a validated fit also saved."""
    import hashlib,json
    path=Path(path);manifest=json.loads((path/"model.json").read_text())
    name,key=("weights.pt","weights_sha256") if which=="selected" else ("weights-final-epoch.pt","final_epoch_weights_sha256")
    if key not in manifest:raise ValueError(f"This model has no {which} checkpoint")
    if hashlib.sha256((path/name).read_bytes()).hexdigest()!=manifest[key]:
        raise ValueError("Frozen weights fingerprint mismatch")
    language=json.loads((path/"language.json").read_text())
    if digest(language)!=manifest["language_hash"]:raise ValueError("Frozen pronunciation lexicon mismatch")
    checkpoint=torch.load(path/name,map_location="cpu",weights_only=True)
    names=manifest.get("feature_names") or []
    if any(n not in FEATURE_NAMES for n in names) or not names or checkpoint["config"]!=manifest["config"] or checkpoint["normalisation"]!=manifest["normalisation"]:
        raise ValueError("Incompatible feature schema or altered frozen configuration/normalisation")
    modalities=checkpoint["normalisation"].get("modalities","imu")
    if modalities!="imu" and manifest.get("camera_feature_names")!=camera_input.FEATURE_NAMES:
        raise ValueError("Camera feature schema changed since this model was frozen")
    if names!=FEATURE_NAMES:
        # Older models used a subset of today's features; select exactly those, in their order.
        checkpoint["normalisation"]=dict(checkpoint["normalisation"],columns=[FEATURE_NAMES.index(n) for n in names])
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
    model=MotionCTC(len(checkpoint["tokens"]),teacher=checkpoint["config"].get("teacher_aux",False),coarse=checkpoint.get("coarse_classes") or 0,inputs=input_width(len(names),modalities),encoder=checkpoint["config"].get("encoder","gru"),
                    conditions=len(CONDITIONS) if checkpoint["config"].get("condition")=="shared" else 0)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    torch.set_num_threads(THREADS)
    return model,checkpoint,manifest,language


@torch.no_grad()
def emission(model,checkpoint,row):
    """Phone and next-phone log-probabilities with any legacy "|" column folded away."""
    norm=checkpoint["normalisation"]
    baseline=baseline_stats(row,norm["baseline"]["floor"],norm.get("columns"))[:2] if norm.get("baseline") else None
    camera=np.asarray(row["camera"]) if norm.get("modalities","imu")!="imu" and row.get("camera") else None
    x,fallback=standardise(np.asarray(row["raw"]),norm,row["quality"]["source"],checkpoint["trained_sources"],baseline=baseline,camera=camera)
    shared=checkpoint.get("config",{}).get("condition")=="shared"
    phones,following,activity,_=model(x[None],torch.tensor([len(x)]),conditions=conditions_of([row]) if shared else None)
    lp,tokens=without_separator(phones[0].log_softmax(-1).numpy(),checkpoint["tokens"])
    following=following[0].log_softmax(-1).numpy()
    if SEPARATOR in checkpoint["tokens"]:
        following=np.delete(following,checkpoint["tokens"].index(SEPARATOR),axis=1)
    return lp,tokens,following,float(activity[0].mean().sigmoid()),fallback
