#!/usr/bin/env python3
"""Paired analysis of one completed 54-trial visit. Reads only the selected visit snapshot.
Requires numpy/scipy/matplotlib in an isolated analysis environment. No final-test input.
Usage: analyze_signal_visit.py VISIT_JSON OUTPUT_DIRECTORY
"""
import json, sys, math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
visit = json.loads(source.read_text())
takes = sorted(visit['takes'], key=lambda t: t['cueStartUptime'])
conditions = {'Signal check — mouth the word':'Mouth', 'Signal check — keep mouth still':'Think', 'Signal check — speak aloud':'Speak'}
assert len(takes) == 54, 'This paired analysis expects the complete three-round 54-trial protocol.'
labels = sorted({t['label'] for t in takes}); assert len(labels)==6
rng = np.random.default_rng(6151)
rows=[]; curves={c:[] for c in conditions.values()}; gaps=[]
for i,t in enumerate(takes):
    assert t['purpose'] in conditions, 'Unexpected non-diagnostic trial'
    s=t['samples']; times=np.array([x['t'] for x in s]); gaps.extend(np.diff(times).tolist())
    cue_start, cue_end=t['cueStartUptime'], t['cueEndUptime']
    baseline=(times>=times[0]+0.15)&(times<cue_start-0.15)
    cue=(times>=cue_start)&(times<=cue_end)
    assert baseline.sum()>10 and cue.sum()>10
    row={'takeID':t['id'],'word':t['label'],'round':i//18,'condition':conditions[t['purpose']], 'cueStart':cue_start}
    for channel,keys in [('rotation',['rx','ry','rz']),('acceleration',['ax','ay','az'])]:
        a=np.array([[x[k] for k in keys] for x in s]); energy=(a*a).sum(axis=1)
        base=math.sqrt(float(energy[baseline].mean())); active=math.sqrt(float(energy[cue].mean()))
        row[channel]={'baselineRMS':base,'cueRMS':active,'ratio':active/max(base,1e-12), 'logRatio':math.log(max(active,1e-12)/max(base,1e-12))}
        if channel=='rotation':
            trace=[]
            for start in np.arange(-1.5,5.0,0.1):
                mask=(times>=cue_start+start)&(times<cue_start+start+0.1)
                trace.append(math.sqrt(float(energy[mask].mean()))/max(base,1e-12) if mask.any() else np.nan)
            curves[row['condition']].append(trace)
    rows.append(row)
for r in range(3):
    assert len({(x['word'],x['condition']) for x in rows if x['round']==r})==18, 'Round is not balanced'

# Exact paired sign-flip test over 18 word×round pairs; one-sided: Mouth > Think.
# Rotation and acceleration are tested together with Holm correction.
def paired(channel):
    lookup={(x['word'],x['round'],x['condition']):x for x in rows}
    differences=np.array([lookup[(w,r,'Mouth')][channel]['logRatio']-lookup[(w,r,'Think')][channel]['logRatio'] for w in labels for r in range(3)])
    total=1<<len(differences); exceed=0; observed=float(differences.mean())
    for start in range(0,total,16384):
        integers=np.arange(start,min(total,start+16384),dtype=np.uint32)
        signs=2*((integers[:,None]>>np.arange(len(differences),dtype=np.uint32))&1).astype(float)-1
        stats=signs@differences/len(differences)
        exceed+=int((stats>=observed-1e-12).sum())
    boot=differences[rng.integers(0,len(differences),size=(20000,len(differences)))].mean(axis=1)
    return {'pairedCount':len(differences),'geometricMeanMouthToThinkRatio':math.exp(observed),
            'bootstrap95CI':np.exp(np.quantile(boot,[.025,.975])).tolist(),
            'oneSidedExactSignFlipP':exceed/total,'positivePairs':int((differences>0).sum()),
            'conditionMedianRatios':{c:float(np.median([x[channel]['ratio'] for x in rows if x['condition']==c])) for c in conditions.values()}}
stats={c:paired(c) for c in ['rotation','acceleration']}
ordered=sorted(stats,key=lambda c:stats[c]['oneSidedExactSignFlipP']); previous=0
for i,c in enumerate(ordered):
    previous=max(previous,min(1,(2-i)*stats[c]['oneSidedExactSignFlipP'])); stats[c]['holmAdjustedP']=previous

# Simple condition discrimination with thresholds fitted on other rounds only.
cv=[]
for held in range(3):
    train=[x for x in rows if x['condition'] in ('Mouth','Think') and x['round']!=held]
    test=[x for x in rows if x['condition'] in ('Mouth','Think') and x['round']==held]
    vals=sorted(x['rotation']['logRatio'] for x in train)
    thresholds=[vals[0]-1]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+1]
    threshold=max(thresholds,key=lambda cut:sum((x['rotation']['logRatio']>cut)==(x['condition']=='Mouth') for x in train))
    cv.extend({'actual':x['condition'],'predicted':'Mouth' if x['rotation']['logRatio']>threshold else 'Think','round':held} for x in test)

session_dir=Path(visit['visit']['recordingDirectory']); verification=json.loads((session_dir/'voice-verification.json').read_text())
whisper_files=list(session_dir.glob('local-whisper-*.json'))
recognized_by_condition={c:0 for c in conditions.values()}
if whisper_files:
    words=json.loads(whisper_files[-1].read_text())['words']; meta=json.loads((session_dir/'meta.json').read_text())
    for row,t in zip(rows,takes):
        lo=t['cueStartUptime']-meta['startUptime']-meta['audioStartOffsetSeconds']; hi=t['cueEndUptime']-meta['startUptime']-meta['audioStartOffsetSeconds']
        heard=[w for w in words if w['end']>=lo and w['start']<=hi]
        if heard: recognized_by_condition[row['condition']]+=1
else: meta={}
result={'visitID':visit['visit']['id'],'trials':54,'words':labels,'pairedMotion':stats,
        'conditionThresholdCV':{'correct':sum(x['actual']==x['predicted'] for x in cv),'total':len(cv),'trials':cv},
        'sampleRateMedianHz':float(1/np.median(gaps)),'largestSampleIntervalSeconds':float(max(gaps)),
        'intervalsOver120ms':int(sum(x>.12 for x in gaps)),
        'originalAppVerificationMatches':sum(x['matched'] for x in verification),'voicedReferences':len(verification),
        'whisperIntervalOverlapCountsNotSpeechDetection':recognized_by_condition,'audioDevice':meta.get('audioDevice'),
        'trialMetrics':rows,
        'limitations':['One visit, only three repetitions per word. Word/round pairs are treated as exchangeable for the exploratory sign-flip test; repeated measurements are not independent people or sessions.',
                       'A larger motion response establishes condition-related movement, not word identity or jaw-specific motion.',
                       'Bootstrap intervals are descriptive within this visit, not population confidence or independent-session validation.',
                       'Think means instructed stillness, not an objective measurement of thought. Silent trials require acoustic checks; Whisper detections may be alignment spillover or hallucination.']}
# Preserve the original app output as provenance, but do not mistake broad segment spans
# for word-level verification or speech in silent trials.
if whisper_files:
    spans=[w['end']-w['start'] for w in words]
    result['whisperMedianReportedWordSpanSeconds']=float(np.median(spans)) if spans else None
    result['originalVerificationTimingReliable']=bool(spans and np.median(spans)<2)
clip_check=out/'clips/verification.json'
if clip_check.exists():
    reviewed=json.loads(clip_check.read_text())
    voiced=[x for x in reviewed if x['purpose']=='Signal check — speak aloud']
    result['independentlyCroppedVoiceMatches']=sum(x['matched'] for x in voiced)
    result['independentlyCroppedVoiceCount']=len(voiced)
word_check=out/'word-identity-significance.json'
if word_check.exists(): result['wordIdentityRoundPermutationResults']=json.loads(word_check.read_text())
sensitivity=out/'without-loud-pair-audit.json'
if sensitivity.exists():
    checked=json.loads(sensitivity.read_text())
    result['sensitivityWithoutLoudPair']={
        'mouthTrials':checked['repeatedTakeCount'],
        'wholeWord':next(x for x in checked['results'] if x['name']=='all / whole')}
(out/'paired-analysis.json').write_text(json.dumps(result,indent=2))

colors={'Speak':'#19764c','Mouth':'#315fc8','Think':'#7b3eb5'}
fig,axes=plt.subplots(1,3,figsize=(15,5.1),gridspec_kw={'width_ratios':[1,1,1.6]})
for ax,channel,title in zip(axes[:2],['rotation','acceleration'],['Rotation','Acceleration']):
    for i,c in enumerate(['Think','Mouth','Speak']):
        vals=np.array([x[channel]['ratio'] for x in rows if x['condition']==c])
        jitter=rng.uniform(-.12,.12,len(vals)); ax.scatter(i+jitter,vals,color=colors[c],s=27,alpha=.75)
        med=np.median(vals);ax.plot([i-.22,i+.22],[med,med],color=colors[c],lw=3)
    ax.axhline(1,color='#966012',ls='--',lw=1,label='Blank baseline')
    ax.set_yscale('log');ax.set_xticks([0,1,2],['Think','Mouth','Speak']);ax.set_title(title,loc='left',fontweight='bold')
    ax.set_ylabel('Cue RMS / blank-baseline RMS'); ax.grid(axis='y',alpha=.14)
    ax.spines[['top','right']].set_visible(False)
ax=axes[2];x=np.arange(-1.5,5.0,.1)+.05
for c in ['Think','Mouth','Speak']:
    a=np.asarray(curves[c]);med=np.nanmedian(a,axis=0);q=np.nanquantile(a,[.25,.75],axis=0)
    ax.plot(x,med,color=colors[c],label=c,lw=2);ax.fill_between(x,q[0],q[1],color=colors[c],alpha=.12)
ax.axvline(0,color='#333333',ls=':',lw=1);ax.axhline(1,color='#966012',ls='--',lw=1)
ax.set_title('When does the motion appear?',loc='left',fontweight='bold');ax.set_xlabel('Seconds from word cue');ax.set_ylabel('Rotation / blank-baseline RMS');ax.set_yscale('log');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14)
fig.suptitle('Mouthing versus Think: your completed recording',x=.055,ha='left',fontweight='bold',fontsize=16)
fig.text(.055,.90,'54 randomized trials • six words • three rounds • 18 trials per condition',fontsize=10,color='#555555')
fig.tight_layout(rect=[0,0,1,.86]);fig.savefig(out/'motion-comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k not in ['trialMetrics','conditionThresholdCV','limitations']},indent=2))
print('Condition-only held-out-round accuracy',result['conditionThresholdCV']['correct'],'/',len(cv))
