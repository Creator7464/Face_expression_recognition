
from .config import DEFAULT

EMOTIONS = ["anger", "sad", "happy", "fear", "neutral"]


def ramp(x, lo, hi):
    """Linear 0..1 ramp between lo and hi."""
    if hi == lo:
        return float(x >= hi)
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def classify_features(f, t=DEFAULT):
    smile = ramp(f["curvature"], *t.smile_curv)
    frown = ramp(-f["curvature"], *t.frown_curv)
    wide = ramp(f["mouth_width"], *t.wide_mouth)
    narrow = ramp(t.narrow_mouth_ref - f["mouth_width"], 0.0, t.narrow_mouth_span)
    teeth = ramp(f["teeth"], *t.teeth)
    opened = ramp(f["mouth_open"], *t.mouth_open)
    raised = ramp(f["brow_gap"], *t.brow_raised)
    lowered = ramp(t.brow_lowered_ref - f["brow_gap"], 0.0, t.brow_lowered_span)
    furrow = ramp(f["furrow"], *t.furrow)
    sclera = ramp(f["sclera"], *t.sclera)
    cheeks = ramp(f["cheek_red"], *t.cheek_red)
    eyes_red = ramp(f["eye_red"], *t.eye_red)

    scores = {
        # smiling mouth, wide, teeth showing, slightly flushed cheeks
        "happy": 0.45 * smile + 0.25 * wide + 0.20 * teeth + 0.10 * cheeks * (smile > 0),
        # downturned mouth, narrow, reddened under-eye (crying)
        "sad": 0.50 * frown + 0.20 * narrow + 0.15 * eyes_red + 0.15 * (1 - opened) * frown,
        # lowered/furrowed brows, flushed face, tight lips
        "anger": 0.30 * furrow + 0.25 * lowered + 0.20 * cheeks
                 + 0.15 * (1 - opened) * (1 - smile) + 0.10 * narrow,
        # wide eyes, raised brows, open mouth without a smile
        "fear": 0.35 * sclera + 0.30 * raised + 0.35 * opened * (1 - smile),
    }
    scores["neutral"] = t.neutral_base + t.neutral_gain * (1 - max(scores.values()))
    label = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    return label, scores[label] / total, {k: round(v, 3) for k, v in scores.items()}
