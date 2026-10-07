
import argparse
import csv
import json

import cv2

from fer.evaluate import compute_metrics, format_report, iter_dataset
from fer.recognizer import FaceExpressionRecognizer

FEATURES = ["mouth_width", "curvature", "mouth_open", "teeth", "furrow",
            "brow_gap", "sclera", "cheek_red", "eye_red"]
SCORES = ["anger", "sad", "happy", "fear", "neutral"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with one sub-folder per emotion")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--no-mtcnn", action="store_true",
                    help="skip MTCNN; treat every image as one cropped face")
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip images where MTCNN finds no face (default: use whole image)")
    ap.add_argument("--json", help="save metrics to this JSON file")
    ap.add_argument("--dump-features", metavar="CSV",
                    help="save one row per image: label, prediction, features, scores")
    a = ap.parse_args()

    rec = FaceExpressionRecognizer(use_mtcnn=not a.no_mtcnn)
    items = list(iter_dataset(a.data, a.max_per_class))
    if not items:
        raise SystemExit(f"No images found under {a.data} (expected <emotion>/<image> folders)")

    y_true, y_pred, rows, skipped = [], [], [], 0
    for i, (path, label) in enumerate(items, 1):
        img = cv2.imread(path)
        res = rec.analyze(img, assume_face=not a.no_fallback) if img is not None else []
        if not res:
            skipped += 1
        else:
            best = max(res, key=lambda r: r["box"][2] * r["box"][3])   # largest face
            y_true.append(label)
            y_pred.append(best["emotion"])
            rows.append([path, label, best["emotion"],
                         int(best["detection_confidence"] > 0)]       # 1 = MTCNN, 0 = fallback
                        + [best["features"][k] for k in FEATURES]
                        + [best["scores"][k] for k in SCORES])
        if i % 100 == 0 or i == len(items):
            print(f"  processed {i}/{len(items)}", flush=True)

    m = compute_metrics(y_true, y_pred)
    m["skipped_no_face"] = skipped
    print(format_report(m))
    n_det = sum(r[3] for r in rows)
    print(f"\nMTCNN detected a face in {n_det}/{len(rows)} images; "
          f"the rest used the whole-image fallback.")

    if a.json:
        json.dump(m, open(a.json, "w"), indent=2)
    if a.dump_features:
        with open(a.dump_features, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_label", "predicted", "mtcnn_detected"]
                       + FEATURES + [f"score_{k}" for k in SCORES])
            w.writerows(rows)
        print(f"Saved {len(rows)} feature rows to {a.dump_features}")


if __name__ == "__main__":
    main()