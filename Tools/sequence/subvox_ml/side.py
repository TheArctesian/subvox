"""Infer which earbud (left 1, right 2) recorded takes whose sensor side was never saved.

The two buds sit mirrored, so their mean gravity direction differs in sign along the ear axis.
A small logistic model on that direction is trusted only if it classifies every held-out
known-side session almost perfectly, and a session's side is filled in only when every one of
its takes agrees confidently. Otherwise the takes stay masked: a wrong side would feed mirrored
axes to the model, which is worse than masking them."""
from __future__ import annotations
import numpy as np

LOSO_ACCURACY = .97   # every held-out known-side session must reach this take accuracy
TAKE_CONFIDENCE = .95  # every take of an unknown session must reach this probability, same side


def side_features(row):
    """Mean gravity unit vector over the take (the rest window when available)."""
    raw = np.asarray(row.get("baseline_raw") or row["raw"])[:, 6:9]
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-6)
    return unit.mean(0)


def fit_logistic(x, y, l2=.01, iterations=50):
    """L2-regularised logistic regression by Newton's method; returns weights with bias last."""
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ w))
        gradient = x.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        hessian = (x.T * (p * (1 - p))) @ x + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(hessian + 1e-9 * np.eye(len(w)), gradient)
    return w


def probability_right(w, x):
    return 1 / (1 + np.exp(-(np.hstack([x, np.ones((len(x), 1))]) @ w)))


def infer_sides(rows):
    """Returns ({row id: inferred side}, report). Known sides are never changed."""
    known = [r for r in rows if r["quality"]["source"] in (1, 2)]
    unknown = [r for r in rows if r["quality"]["source"] not in (1, 2)]
    sessions = sorted({r["session"] for r in known})
    report = dict(known_sessions=len(sessions), unknown_takes=len(unknown), loso={}, accepted=False, assigned_sessions={})
    if len(sessions) < 2 or not unknown or len({r["quality"]["source"] for r in known}) < 2:
        report["reason"] = "need known-side sessions of both sides and some unknown takes"
        return {}, report
    x = np.array([side_features(r) for r in known])
    y = np.array([r["quality"]["source"] == 2 for r in known], dtype=float)
    for session in sessions:
        held = np.array([r["session"] == session for r in known])
        if len(set(y[~held])) < 2:
            report["loso"][session] = None
            continue
        w = fit_logistic(x[~held], y[~held])
        report["loso"][session] = float(((probability_right(w, x[held]) > .5) == y[held]).mean())
    scores = [v for v in report["loso"].values() if v is not None]
    if not scores or min(scores) < LOSO_ACCURACY:
        report["reason"] = f"leave-one-session-out accuracy {min(scores) if scores else None} below {LOSO_ACCURACY}"
        return {}, report
    report["accepted"] = True
    w = fit_logistic(x, y)
    inferred = {}
    for session in sorted({r["session"] for r in unknown}):
        takes = [r for r in unknown if r["session"] == session]
        p = probability_right(w, np.array([side_features(r) for r in takes]))
        if (p > TAKE_CONFIDENCE).all() or (p < 1 - TAKE_CONFIDENCE).all():
            side = 2 if p[0] > .5 else 1
            inferred.update({r["id"]: side for r in takes})
            report["assigned_sessions"][session] = dict(side=side, takes=len(takes), min_confidence=float(np.minimum(p, 1 - p).max() if side == 1 else p.min()))
        else:
            report["assigned_sessions"][session] = dict(side=None, takes=len(takes), reason="takes disagree or below confidence")
    return inferred, report
