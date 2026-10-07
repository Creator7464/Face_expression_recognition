
import os

import cv2

LABELS = ["anger", "sad", "happy", "fear", "neutral"]
# folder-name aliases used by common datasets (FER-2013, CK+, JAFFE, RAF-DB ...)
ALIASES = {"angry": "anger", "anger": "anger", "sad": "sad", "sadness": "sad",
           "happy": "happy", "happiness": "happy", "fear": "fear", "fearful": "fear",
           "neutral": "neutral"}
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_dataset(root, max_per_class=None):
    
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        label = ALIASES.get(sub.lower())
        if label is None:               # maybe a split folder (train/test)
            if sub.lower() in ("train", "test", "val", "validation"):
                yield from iter_dataset(d, max_per_class)
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
        for f in files[:max_per_class]:
            yield os.path.join(d, f), label


def compute_metrics(y_true, y_pred):
    conf = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    per_class = {}
    for c in LABELS:
        tp = conf[c][c]
        support = sum(conf[c].values())
        predicted = sum(conf[t][c] for t in LABELS)
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "support": support}
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0
    return {"accuracy": round(acc, 4), "n": n, "confusion": conf, "per_class": per_class}


def evaluate(recognizer, root, max_per_class=None, assume_face=True, progress=None):
    items = list(iter_dataset(root, max_per_class))
    y_true, y_pred, skipped = [], [], 0
    for i, (path, label) in enumerate(items):
        img = cv2.imread(path)
        res = recognizer.analyze(img, assume_face=assume_face) if img is not None else []
        if not res:
            skipped += 1
        else:
            best = max(res, key=lambda r: r["box"][2] * r["box"][3])  # largest face
            y_true.append(label)
            y_pred.append(best["emotion"])
        if progress:
            progress((i + 1) / len(items))
    out = compute_metrics(y_true, y_pred)
    out["skipped_no_face"] = skipped
    return out


def format_report(m):
    lines = [f"Accuracy: {m['accuracy']:.1%} on {m['n']} images "
             f"(skipped {m.get('skipped_no_face', 0)})", "",
             "Confusion matrix (rows = true, cols = predicted)",
             " " * 9 + "".join(f"{c:>9}" for c in LABELS)]
    for t in LABELS:
        lines.append(f"{t:<9}" + "".join(f"{m['confusion'][t][p]:>9}" for p in LABELS))
    lines += ["", f"{'class':<9}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>6}"]
    for c, v in m["per_class"].items():
        lines.append(f"{c:<9}{v['precision']:>8.2f}{v['recall']:>8.2f}{v['f1']:>8.2f}{v['support']:>6}")
    return "\n".join(lines)
