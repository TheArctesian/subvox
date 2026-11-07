"""CTC prefix beam, pronunciation lexicon, optional source-only word bigram model.

The network emits phones only. Word boundaries are hypotheses of the lexical beam: wherever
the current word is complete it may end there (BOUNDARY, consuming no frame) or continue.
Connected speech has no articulatory gap between words, so the motion model isn't asked for one."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
import numpy as np
from .data import words

NEG = -float("inf")
BOUNDARY = -1
SEPARATOR = "|"


def logadd(*values):
    top = max(values)
    if not math.isfinite(top):
        return top
    return top + math.log(sum(math.exp(x-top) for x in values))


def without_separator(log_probs, tokens):
    """Checkpoints before boundary-free training emitted "|". Fold it into blank so one
    decoder serves both: an emitted boundary then costs nothing and places no word break."""
    if SEPARATOR not in tokens:
        return np.asarray(log_probs), list(tokens)
    s = tokens.index(SEPARATOR)
    lp = np.array(log_probs, dtype=np.float64)
    lp[:, 0] = np.logaddexp(lp[:, 0], lp[:, s])
    return np.delete(lp, s, axis=1), tokens[:s] + tokens[s+1:]


class Lexicon:
    def __init__(self, language, tokens):
        self.tokens = tokens
        self.index = {t:i for i,t in enumerate(tokens)}
        self.pronunciations = defaultdict(list)
        self.prefixes = set()
        for word, variants in sorted(language["entries"].items()):
            for variant in variants:
                if not variant["ipa"] or any(p not in self.index for p in variant["ipa"]):
                    continue
                path = tuple(self.index[p] for p in variant["ipa"])
                if word not in self.pronunciations[path]:
                    self.pronunciations[path].append(word)
                for i in range(len(path)+1):
                    self.prefixes.add(path[:i])
        self._allowed = {}

    @staticmethod
    def current_word(prefix):
        boundary = len(prefix)-1
        while boundary >= 0 and prefix[boundary] != BOUNDARY:
            boundary -= 1
        return prefix[boundary+1:]

    def allowed(self, prefix, token):
        # Prefix validation depends only on the current unfinished word, not expected text.
        key = (self.current_word(prefix), token)
        if key not in self._allowed:
            self._allowed[key] = key[0]+(token,) in self.prefixes
        return self._allowed[key]

    def complete(self, prefix):
        word = self.current_word(prefix)
        return not word or word in self.pronunciations

    def word_paths(self, prefix, width=8, lm=None):
        parts = [[]]
        for token in prefix:
            if token == BOUNDARY:
                parts.append([])
            else:
                parts[-1].append(token)
        if parts and not parts[-1]:
            parts.pop()
        if not parts or any(tuple(part) not in self.pronunciations for part in parts):
            return []
        hypotheses = [((), 0.0)]
        for part in parts:
            choices = self.pronunciations[tuple(part)]
            next_paths = []
            for prior, score in hypotheses:
                for word in choices:
                    penalty = lm.logprob(prior[-1] if prior else "<s>", word) if lm else 0
                    next_paths.append((prior+(word,), score+penalty))
            hypotheses = sorted(next_paths, key=lambda h:(-h[1],h[0]))[:width]
        return hypotheses


class BigramLM:
    def __init__(self, texts):
        self.counts = Counter()
        self.contexts = Counter()
        self.vocabulary = set()
        for text in texts:
            prior = "<s>"
            for word in text:
                self.counts[prior,word] += 1
                self.contexts[prior] += 1
                self.vocabulary.add(word)
                prior = word

    def logprob(self, prior, word):
        # Additive smoothing; unseen words retain finite probability. No held-out text.
        return math.log((self.counts[prior,word]+.1)/(self.contexts[prior]+.1*(len(self.vocabulary)+1)))


def prefix_beam(log_probs, width=16, topk=10, lexicon=None):
    """With a lexicon, only hypotheses ending on a complete word are returned (the
    end-of-utterance constraint); an unfinished last word is never silently dropped later."""
    beam = {(): (0.0, NEG)}
    def score(item):
        return -logadd(*item[1]), item[0]
    for frame in np.asarray(log_probs):
        candidates = np.argsort(frame)[-topk:].tolist()
        if 0 not in candidates:
            candidates.append(0)
        updated = {}
        def accumulate(prefix, blank=NEG, nonblank=NEG):
            b,n = updated.get(prefix,(NEG,NEG))
            updated[prefix] = (logadd(b,blank),logadd(n,nonblank))
        for prefix,(pb,pn) in beam.items():
            accumulate(prefix,blank=logadd(pb,pn)+float(frame[0]))
            # A boundary emits nothing, so CTC's repeat rule looks through it to the last phone.
            last = next((t for t in reversed(prefix) if t != BOUNDARY), None)
            for token in candidates:
                if token == 0:
                    continue
                probability = float(frame[token])
                if token == last:
                    accumulate(prefix,nonblank=pn+probability)
                    if lexicon is None or lexicon.allowed(prefix,token):
                        accumulate(prefix+(token,),nonblank=pb+probability)
                elif lexicon is None or lexicon.allowed(prefix,token):
                    accumulate(prefix+(token,),nonblank=logadd(pb,pn)+probability)
        if lexicon is not None:
            for prefix,value in list(updated.items()):
                if prefix and prefix[-1] != BOUNDARY and lexicon.complete(prefix):
                    ended = prefix+(BOUNDARY,)
                    if ended not in updated:
                        updated[ended] = value
        beam = dict(sorted(updated.items(),key=score)[:width*2 if lexicon is not None else width])
    if lexicon is None:
        return [(prefix,logadd(*p)) for prefix,p in sorted(beam.items(),key=score)]
    finished = {}
    for prefix,p in beam.items():
        if not lexicon.complete(prefix):
            continue
        key = prefix[:-1] if prefix and prefix[-1] == BOUNDARY else prefix
        finished[key] = max(finished.get(key,NEG),logadd(*p))
    return sorted(finished.items(),key=lambda h:(-h[1],h[0]))


def viterbi_positions(log_probs, phones):
    """Frame-to-phone-position assignment of the best CTC path for `phones` (-1 = blank)."""
    T = len(log_probs)
    states = [0]
    for token in phones:
        states += [token, 0]
    S = len(states)
    emission = np.asarray(log_probs)[:, states]
    skip = np.array([s > 1 and states[s] != 0 and states[s] != states[s-2] for s in range(S)])
    delta = np.full((T, S), NEG); back = np.zeros((T, S), dtype=np.int64)
    delta[0, 0] = emission[0, 0]
    if S > 1:
        delta[0, 1] = emission[0, 1]
    for t in range(1, T):
        stay = delta[t-1]
        one = np.concatenate([[NEG], delta[t-1, :-1]])
        two = np.where(skip, np.concatenate([[NEG, NEG], delta[t-1, :-2]]), NEG)
        choices = np.stack([stay, one, two])
        best = choices.argmax(0)
        delta[t] = choices[best, np.arange(S)] + emission[t]
        back[t] = np.arange(S) - best
    s = S-1 if S == 1 or delta[-1, S-1] >= delta[-1, S-2] else S-2
    if not np.isfinite(delta[-1, s]):
        return None
    path = []
    for t in range(T-1, -1, -1):
        path.append((s-1)//2 if s % 2 else -1)
        s = back[t, s]
    return path[::-1]


def transition_score(log_probs, next_log_probs, phones):
    """Mean log-probability the next-phone head gives each actual successor while the model
    is on the preceding phone, along the hypothesis's own best alignment."""
    if len(phones) < 2:
        return 0.0
    positions = viterbi_positions(log_probs, phones)
    if positions is None:
        return NEG
    positions = np.asarray(positions)
    total = 0.0
    for i in range(len(phones)-1):
        frames = np.flatnonzero(positions == i)
        total += float(np.asarray(next_log_probs)[frames, phones[i+1]].mean())
    return total / (len(phones)-1)


def decode(log_probs, tokens, lexicon, lm=None, lm_weight=.15, next_log_probs=None, transition_weight=1.0):
    """`next_log_probs` (the next-phone head) rescores word hypotheses by how well each phone
    predicts its successor. That is where diphone context enters decoding, without a diphone
    output vocabulary; reports keep it separate from plain decoding."""
    free = prefix_beam(log_probs)
    lexical = prefix_beam(log_probs,lexicon=lexicon)
    ranked = []
    for prefix, acoustic in lexical:
        phones = [t for t in prefix if t != BOUNDARY]
        transitions = transition_score(log_probs,next_log_probs,phones) if next_log_probs is not None else 0.0
        if not math.isfinite(transitions):
            continue
        for word_path, language in lexicon.word_paths(prefix,lm=lm):
            ranked.append(dict(text=" ".join(word_path),phones=[tokens[i] for i in phones],
                               acoustic_score=acoustic,lm_score=language,transition_score=transitions,
                               score=acoustic+lm_weight*language+transition_weight*transitions*len(phones)))
    unique = {}
    for row in ranked:
        if row["text"] not in unique or row["score"] > unique[row["text"]]["score"]:
            unique[row["text"]] = row
    ranked = sorted(unique.values(),key=lambda h:(-h["score"],h["text"]))
    margin = ranked[0]["score"]-ranked[1]["score"] if len(ranked)>1 else 0
    tied = [row for row in ranked if ranked[0]["score"]-row["score"]<1e-8]
    # Homophones (to/two, blue/blew) tie exactly without a language model, and motion can't
    # separate them any more than audio can. Keep the first alphabetically and list the rest;
    # only a tie between different phone strings is real motion ambiguity.
    ambiguous = len({tuple(row["phones"]) for row in tied})>1
    return dict(phones=[tokens[i] for i in (free[0][0] if free else ())],
                words=ranked[0]["text"] if ranked else "",ranked=ranked[:5],margin=margin,
                ambiguous=ambiguous,homophones=[row["text"] for row in tied[1:]] if not ambiguous else [],
                language_model=lm is not None,transitions=next_log_probs is not None)


# Fixed before any result was seen; 0.5 is reported as a sensitivity check, never selected.
PRIOR_ALPHA = .3


def prior_correct(log_probs, log_prior, alpha=PRIOR_ALPHA):
    """Divide emissions by the model's own training label prior (hybrid-ASR style), then
    renormalise per frame. A model that emits blank a priori then has to earn each blank."""
    adjusted = np.asarray(log_probs) - alpha*np.asarray(log_prior)
    return adjusted - np.logaddexp.reduce(adjusted, axis=1, keepdims=True)


def homophone_equal(language):
    """Word equality that also accepts words sharing a pronunciation variant."""
    sounds = defaultdict(set)
    for word, variants in language["entries"].items():
        for variant in variants:
            sounds[word].add(tuple(variant["ipa"]))
    return lambda a, b: a == b or bool(sounds.get(a, set()) & sounds.get(b, set()))


def edit_alignment(reference, hypothesis, same=None):
    differ=(lambda a,b:not same(a,b)) if same else (lambda a,b:a!=b)
    n,m=len(reference),len(hypothesis)
    dp=np.zeros((n+1,m+1),dtype=np.int32)
    dp[:,0]=np.arange(n+1);dp[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+differ(reference[i-1],hypothesis[j-1]))
    alignment=[];i,j=n,m
    while i or j:
        if i and j and dp[i,j]==dp[i-1,j-1]+differ(reference[i-1],hypothesis[j-1]):
            alignment.append((reference[i-1],hypothesis[j-1]));i-=1;j-=1
        elif i and dp[i,j]==dp[i-1,j]+1:
            alignment.append((reference[i-1],"<del>"));i-=1
        else:
            alignment.append(("<ins>",hypothesis[j-1]));j-=1
    return int(dp[n,m]),list(reversed(alignment))


def summary(rows, seed=42):
    positives=[r for r in rows if r["activity"]]
    negatives=[r for r in rows if not r["activity"]]
    def error_rate(key_num,key_den, subset):
        return sum(r[key_num] for r in subset)/max(1,sum(r[key_den] for r in subset))
    result=dict(utterances=len(positives),negative_examples=len(negatives),
                phoneme_error_rate=error_rate("phone_errors","phone_count",positives) if positives else None,
                word_error_rate_no_lm=error_rate("word_errors","word_count",positives) if positives else None,
                word_error_rate_lm=error_rate("lm_word_errors","word_count",positives) if positives else None,
                word_error_rate_transitions=error_rate("transition_word_errors","word_count",positives)
                    if positives and all("transition_word_errors" in r for r in positives) else None,
                rejection_rate=sum(not r["accepted"] for r in positives)/len(positives) if positives else None,
                false_accept_rate=sum(r["accepted"] for r in negatives)/len(negatives) if negatives else None)
    if positives and all("homophone_word_errors" in r for r in positives):
        # Counts blue for blew as correct: the share of strict errors that are only spelling.
        result["word_error_rate_no_lm_homophones_merged"]=error_rate("homophone_word_errors","word_count",positives)
    # Cluster bootstrap by whole sentence group, not independent frames/repetitions.
    groups=sorted({tuple(r["groups"]) for r in positives})
    result["sentence_groups"]=len(groups)
    if len(groups)>=2:
        rng=np.random.default_rng(seed);values=[]
        for _ in range(500):
            sampled=[r for i in rng.integers(0,len(groups),len(groups)) for r in positives if tuple(r["groups"])==groups[i]]
            values.append(error_rate("word_errors","word_count",sampled))
        result["wer_cluster_bootstrap_95_interval"]=np.quantile(values,[.025,.975]).tolist()
    else:
        result["wer_cluster_bootstrap_95_interval"]=None
        result["uncertainty_note"]="Only one sentence group; no defensible group-bootstrap interval. Repetitions are not independent vocabulary examples."
    result["phone_confusions"]=dict(Counter(f"{a} → {b}" for r in positives for a,b in r["phone_alignment"] if a!=b).most_common(30))
    result["word_confusions"]=dict(Counter(f"{a} → {b}" for r in positives
        for a,b in edit_alignment(words(r["reference_text"]),words(r["text"]))[1] if a!=b).most_common(30)) if positives and all("reference_text" in r and "text" in r for r in positives) else {}
    return result
