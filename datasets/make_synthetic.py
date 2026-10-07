"""Generate a small synthetic cartoon-face set (smoke test, NOT a benchmark).

    python datasets/make_synthetic.py --out datasets/synthetic --per-class 20

Faces are drawn at the positions used by the whole-image fallback, so evaluate
with --no-mtcnn (MTCNN will not detect cartoon faces).
"""
import argparse
import os

import cv2
import numpy as np

S = 256
SKIN = (110, 140, 190)          # BGR
EYE_X = (int(.30 * S), int(.70 * S))
EYE_Y, MOUTH_Y = int(.38 * S), int(.78 * S)
U = (EYE_X[1] - EYE_X[0]) / 64.0  # pixels per canonical unit


def face(emotion, rng):
    img = np.full((S, S, 3), SKIN, np.uint8)
    cv2.ellipse(img, (S // 2, int(.52 * S)), (int(.40 * S), int(.47 * S)), 0, 0, 360,
                (120, 150, 200), -1)
    j = lambda a: a * rng.uniform(.85, 1.15)
    eye_h = {"fear": 11, "anger": 5, "sad": 7}.get(emotion, 7)
    brow_gap = {"fear": 36, "anger": 18}.get(emotion, 26)
    for i, x in enumerate(EYE_X):
        cv2.ellipse(img, (x, EYE_Y), (int(19 * U / 1.6 * 1.6), int(j(eye_h) * U)), 0, 0, 360,
                    (255, 255, 255), -1)
        cv2.circle(img, (x, EYE_Y), int(5 * U), (40, 30, 30), -1)
        by = EYE_Y - int(j(brow_gap) * U)
        dy = int(5 * U) if emotion == "anger" else 0
        sgn = 1 if i == 0 else -1
        cv2.line(img, (x - 14 * int(U), by - sgn * dy), (x + 14 * int(U), by + sgn * dy),
                 (30, 30, 40), max(2, int(2.5 * U)))
    if emotion == "anger":                       # furrow lines between brows
        for dx in (-3, 3):
            cv2.line(img, (S // 2 + int(dx * U), EYE_Y - int(30 * U)),
                     (S // 2 + int(dx * U), EYE_Y - int(10 * U)), (40, 40, 60), 2)
    cv2.line(img, (S // 2, int(.45 * S)), (S // 2 - 6, int(.60 * S)), (90, 120, 170), 2)
    # mouth
    mx0, mx1 = int(.36 * S), int(.64 * S)
    sag = {"happy": 7, "sad": -7}.get(emotion, 0)
    sag = int(j(abs(sag)) * U) * (1 if sag >= 0 else -1) if sag else 0
    mid = (S // 2, MOUTH_Y + sag)
    pts = np.array([(mx0, MOUTH_Y), ((mx0 + S // 2) // 2, (MOUTH_Y + mid[1]) // 2 + sag // 4),
                    mid, ((mx1 + S // 2) // 2, (MOUTH_Y + mid[1]) // 2 + sag // 4),
                    (mx1, MOUTH_Y)], np.int32)
    if emotion == "fear":
        cv2.ellipse(img, (S // 2, MOUTH_Y), (int(.09 * S), int(j(9) * U)), 0, 0, 360, (20, 10, 40), -1)
    elif emotion == "happy":
        cv2.fillPoly(img, [np.vstack([pts, [(mx1 - 6, MOUTH_Y + 2), (mx0 + 6, MOUTH_Y + 2)]])],
                     (255, 255, 255))
        cv2.polylines(img, [pts], False, (40, 20, 120), max(2, int(2 * U)))
    else:
        cv2.polylines(img, [pts], False, (40, 20, 120), max(2, int(2 * U)))
    # redness: cheeks (anger, happy) and under-eyes (sad)
    ov = img.copy()
    for i, x in enumerate(EYE_X):
        out = -6 if i == 0 else 6
        if emotion in ("anger", "happy"):
            cv2.circle(ov, (x + int(out * U), EYE_Y + int(38 * U)), int(15 * U), (60, 60, 230), -1)
        if emotion == "sad":
            cv2.ellipse(ov, (x, EYE_Y + int(16 * U)), (int(16 * U), int(7 * U)), 0, 0, 360, (70, 70, 230), -1)
    alpha = 0.55 if emotion == "anger" else 0.4
    img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for emo in ["anger", "sad", "happy", "fear", "neutral"]:
        os.makedirs(os.path.join(a.out, emo), exist_ok=True)
        for k in range(a.per_class):
            cv2.imwrite(os.path.join(a.out, emo, f"{emo}_{k:03d}.png"), face(emo, rng))
    print(f"Wrote {5 * a.per_class} images to {a.out}")
