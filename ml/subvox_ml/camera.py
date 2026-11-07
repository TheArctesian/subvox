"""Numeric camera landmarks as a model input, on the motion timeline.

Only Vision points/measurements saved by the app are used; there are no images. Points are put
in an eye frame (origin between the eyes, x along the eye line, units of inter-eye distance), so
head position and roll in the image do not read as mouth movement. Yaw/pitch remain uncorrected.
Contours are resampled by arc length to fixed counts so a Vision revision with a different
point count still maps onto the same channels.
"""
import numpy as np
from .data import RATE

SCALARS = ("opening", "width", "innerArea", "outerAspect", "centreX", "centreY")
OUTER, INNER, CONTOUR, NOSE = 12, 8, 11, 3
MAX_GAP = .12  # s; camera frames further apart are not interpolated across

MOUTH_NAMES = (list(SCALARS) + [f"outer{i}_{a}" for i in range(OUTER) for a in "xy"]
               + [f"inner{i}_{a}" for i in range(INNER) for a in "xy"])
FACE_NAMES = [f"contour{i}_{a}" for i in range(CONTOUR) for a in "xy"] + [f"nose{i}_{a}" for i in range(NOSE) for a in "xy"]
RAW_NAMES = MOUTH_NAMES + FACE_NAMES + ["mouth_valid", "face_valid"]
RAW_WIDTH = len(RAW_NAMES)
MOUTH = slice(0, len(MOUTH_NAMES))
FACE = slice(len(MOUTH_NAMES), len(MOUTH_NAMES) + len(FACE_NAMES))
MOUTH_MASK, FACE_MASK = RAW_WIDTH - 2, RAW_WIDTH - 1
FEATURE_NAMES = MOUTH_NAMES + FACE_NAMES + ["d_" + s for s in SCALARS] + ["mouth_valid", "face_valid"]
FEATURE_WIDTH = len(FEATURE_NAMES)
VALUE_WIDTH = len(MOUTH_NAMES) + len(FACE_NAMES)


def _pixels(points, width, height):
    return np.asarray([[p["x"] * width, p["y"] * height] for p in points], dtype=np.float64)


def _contour(points, count, closed):
    """`count` points evenly spaced by arc length along the contour."""
    if len(points) < 2:
        return None
    path = np.vstack([points, points[:1]]) if closed else points
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if lengths[-1] <= 0:
        return None
    targets = np.linspace(0, lengths[-1], count, endpoint=not closed)
    return np.stack([np.interp(targets, lengths, path[:, k]) for k in range(2)], axis=1)


def frame_values(reading):
    """(mouth vector or None, face vector or None) for one camera reading."""
    mouth = reading.get("mouth")
    width, height = reading.get("imageWidth"), reading.get("imageHeight")
    if not mouth or not width or not height or mouth.get("confidence", 0) < .5 or reading.get("usedReceiptTime"):
        return None, None
    left, right = _pixels([mouth["leftEye"]], width, height)[0], _pixels([mouth["rightEye"]], width, height)[0]
    span = np.linalg.norm(right - left)
    if not np.isfinite(span) or span <= 0:
        return None, None
    u = (right - left) / span
    v = np.array([-u[1], u[0]])
    origin = (left + right) / 2

    def frame(points):
        return np.stack([(points - origin) @ u, (points - origin) @ v], axis=1) / span

    outer = _contour(frame(_pixels(mouth.get("outerLips", []), width, height)), OUTER, True)
    inner = _contour(frame(_pixels(mouth.get("innerLips", []), width, height)), INNER, True)
    scalars = [mouth.get(k, np.nan) for k in SCALARS]
    mouth_vector = None
    if outer is not None and inner is not None:
        mouth_vector = np.concatenate([scalars, outer.ravel(), inner.ravel()])
        if not np.isfinite(mouth_vector).all():
            mouth_vector = None
    face = reading.get("face") or {}
    contour = _contour(frame(_pixels(face.get("faceContour", []), width, height)), CONTOUR, False) if face else None
    nose = _contour(frame(_pixels(face.get("noseCrest", []), width, height)), NOSE, False) if face else None
    face_vector = None
    if contour is not None and nose is not None:
        face_vector = np.concatenate([contour.ravel(), nose.ravel()])
        if not np.isfinite(face_vector).all():
            face_vector = None
    return mouth_vector, face_vector


def grid(readings, start, frames):
    """Camera channels interpolated onto the motion grid `start + i/RATE`. Frames without a
    valid camera reading on both sides within `MAX_GAP` stay zero with their mask column 0.
    One camera/estimator identity per recording: a switch is not interpolated across."""
    out = np.zeros((frames, RAW_WIDTH), dtype=np.float32)
    usable = [r for r in readings or [] if r.get("mouth")]
    if not usable or frames <= 0:
        return out
    identities = {}
    for r in usable:
        key = (r.get("cameraID"), r.get("estimator"))
        identities[key] = identities.get(key, 0) + 1
    chosen = max(identities, key=identities.get)
    times = np.asarray(start + np.arange(frames) / RATE)
    mouth_rows, face_rows = [], []
    for r in sorted((r for r in usable if (r.get("cameraID"), r.get("estimator")) == chosen), key=lambda r: r["t"]):
        m, f = frame_values(r)
        if m is not None:
            mouth_rows.append((r["t"], m))
        if f is not None:
            face_rows.append((r["t"], f))
    for rows, columns, mask in ((mouth_rows, MOUTH, MOUTH_MASK), (face_rows, FACE, FACE_MASK)):
        if len(rows) < 2:
            continue
        t = np.array([x[0] for x in rows])
        values = np.stack([x[1] for x in rows])
        j = np.searchsorted(t, times)
        ok = (j > 0) & (j < len(t))
        jj = np.clip(j, 1, len(t) - 1)
        ok &= (t[jj] - t[jj - 1]) <= MAX_GAP
        fraction = np.clip((times - t[jj - 1]) / np.maximum(t[jj] - t[jj - 1], 1e-9), 0, 1)[:, None]
        interpolated = values[jj - 1] * (1 - fraction) + values[jj] * fraction
        out[ok, columns] = interpolated[ok]
        out[ok, mask] = 1
    return out


def coverage(camera):
    camera = np.asarray(camera)
    return float(camera[:, MOUTH_MASK].mean()) if len(camera) else 0.0


def features(camera):
    """Values (zeroed where invalid), scalar derivatives where both frames are valid, and masks.
    Masks are re-binarised: time-warp augmentation interpolates them."""
    camera = np.asarray(camera, dtype=np.float32)
    mouth_ok = camera[:, MOUTH_MASK] >= .999
    face_ok = camera[:, FACE_MASK] >= .999
    values = camera[:, :VALUE_WIDTH].copy()
    values[~mouth_ok, MOUTH] = 0
    values[~face_ok, FACE] = 0
    scalars = values[:, :len(SCALARS)]
    derivative = np.zeros_like(scalars)
    both = mouth_ok[1:] & mouth_ok[:-1]
    derivative[1:][both] = (scalars[1:] - scalars[:-1])[both] * RATE
    return np.concatenate([values, derivative, mouth_ok[:, None], face_ok[:, None]], axis=1).astype(np.float32)


def valid_columns(feature_rows):
    """Per-frame validity for each feature column (masks themselves are always 'valid')."""
    mouth_ok = feature_rows[:, -2] > .5
    face_ok = feature_rows[:, -1] > .5
    mask = np.ones(feature_rows.shape, dtype=bool)
    mask[:, MOUTH] = mouth_ok[:, None]
    mask[:, FACE] = face_ok[:, None]
    mask[:, VALUE_WIDTH:VALUE_WIDTH + len(SCALARS)] = mouth_ok[:, None]
    return mask
