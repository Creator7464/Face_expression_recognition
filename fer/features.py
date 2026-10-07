
import math

import cv2
import numpy as np

IED = 64.0
CANON = (256, 256)
EYE_MID = (128.0, 100.0)


def _finite(x, default=0.0):
    return float(x) if x is not None and np.isfinite(x) else default


def align_face(bgr, lm):
    """Rotate/scale so eyes are horizontal, 64 px apart, centred at EYE_MID."""
    le, re = sorted([lm["left_eye"], lm["right_eye"]], key=lambda p: p[0])
    ml, mr = sorted([lm["mouth_left"], lm["mouth_right"]], key=lambda p: p[0])
    cx, cy = (le[0] + re[0]) / 2, (le[1] + re[1]) / 2
    dx, dy = re[0] - le[0], re[1] - le[1]
    dist = max(math.hypot(dx, dy), 1e-6)
    angle = math.degrees(math.atan2(dy, dx))
    M = cv2.getRotationMatrix2D((cx, cy), angle, IED / dist)
    M[0, 2] += EYE_MID[0] - cx
    M[1, 2] += EYE_MID[1] - cy
    face = cv2.warpAffine(bgr, M, CANON, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    def tf(p):
        v = M @ np.array([p[0], p[1], 1.0])
        return float(v[0]), float(v[1])

    pts = {"le": tf(le), "re": tf(re), "nose": tf(lm["nose"]),
           "ml": tf(ml), "mr": tf(mr)}
    return face, pts


def _box(cx, cy, hw, hh):
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _crop(img, box):
    h, w = img.shape[:2]
    x0, y0 = int(max(0, round(box[0]))), int(max(0, round(box[1])))
    x1, y1 = int(min(w, round(box[2]))), int(min(h, round(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def _darkest_row(gray, box):
   
    roi = _crop(gray, box)
    if roi is None:
        return None, None
    prof = roi.mean(axis=1)
    if len(prof) >= 3:
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    return int(max(0, round(box[1]))) + int(np.argmin(prof)), prof


def extract_features(bgr, landmarks):
    face, p = align_face(bgr, landmarks)
    gray = cv2.GaussianBlur(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    # reference statistics over an elliptical face mask
    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, (128, 127), (62, 80), 0, 0, 360, 255, -1)
    m = mask > 0
    face_mean = float(gray[m].mean())
    a_ref = float(np.median(lab[..., 1][m]))
    v_ref = float(hsv[..., 2][m].mean())

    ex = [p["le"][0], p["re"][0]]
    ey = (p["le"][1] + p["re"][1]) / 2
    mid_x = (p["ml"][0] + p["mr"][0]) / 2
    mcy = (p["ml"][1] + p["mr"][1]) / 2
    mouth_w = p["mr"][0] - p["ml"][0]
    rois = {}

    # ---- mouth: lip line, curvature, opening, teeth -------------------
    lip_box = _box(mid_x, mcy, max(mouth_w * 0.3, 4), 16)
    lip_y, prof = _darkest_row(gray, lip_box)
    rois["mouth_band"] = lip_box
    if lip_y is None:
        curvature, mouth_open, teeth = 0.0, 0.0, 0.0
    else:
        curvature = (lip_y - mcy) / IED          # >0: corners above lip line
        thr = prof.min() + 0.5 * (np.median(prof) - prof.min())
        mouth_open = float((prof < thr).sum()) / IED
        tb = _box(mid_x, lip_y, max(mouth_w * 0.3, 4), 10)
        rois["teeth"] = tb
        t_hsv = _crop(hsv, tb)
        teeth = 0.0 if t_hsv is None else float(
            ((t_hsv[..., 1] < 70) & (t_hsv[..., 2] > max(165, 1.25 * v_ref))).mean())

    # ---- brows: furrow energy + brow-eye distance -----------------------
    gb = _box(128, ey - 20, 10, 14)
    rois["glabella"] = gb
    g = _crop(gray, gb)
    furrow = 0.0 if g is None else float(
        np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean() / 8.0 / (face_mean + 1))

    gaps = []
    for i, x in enumerate(ex):
        bb = _box(x, ey - 27, 16, 15)            # rows ey-42 .. ey-12
        rois[f"brow{i}"] = bb
        by, _ = _darkest_row(gray, bb)
        if by is not None:
            gaps.append((ey - by) / IED)
    brow_gap = float(np.mean(gaps)) if gaps else 0.40

    # ---- eyes: visible sclera ------------------------------------------
    scl = []
    for i, x in enumerate(ex):
        eb = _box(x, ey, 18, 9)
        rois[f"eye{i}"] = eb
        e = _crop(hsv, eb)
        if e is not None:
            scl.append(float(((e[..., 1] < 70) & (e[..., 2] > max(150, 1.2 * v_ref))).mean()))
    sclera = float(np.mean(scl)) if scl else 0.0

    # ---- redness: cheeks and under-eye (Lab a*) -------------------------
    a_ch = lab[..., 1]
    cheek, bags = [], []
    for i, x in enumerate(ex):
        out = -6 if i == 0 else 6
        cb = _box(x + out, ey + 38, 11, 9)
        rois[f"cheek{i}"] = cb
        c = _crop(a_ch, cb)
        if c is not None:
            cheek.append(float(c.mean()))
        ub = _box(x, ey + 16, 12, 4)
        rois[f"undereye{i}"] = ub
        c = _crop(a_ch, ub)
        if c is not None:
            bags.append(float(c.mean()))
    cheek_red = (float(np.mean(cheek)) - a_ref) if cheek else 0.0
    eye_red = (float(np.mean(bags)) - a_ref) if bags else 0.0

    feats = {
        "mouth_width": mouth_w / IED,
        "curvature": curvature,
        "mouth_open": mouth_open,
        "teeth": teeth,
        "furrow": furrow,
        "brow_gap": brow_gap,
        "sclera": sclera,
        "cheek_red": cheek_red,
        "eye_red": eye_red,
    }
    feats = {k: round(_finite(v), 4) for k, v in feats.items()}
    return feats, face, rois
