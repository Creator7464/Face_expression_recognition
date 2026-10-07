
import argparse
import json
import os

import cv2

from fer import FaceExpressionRecognizer
from fer.visualize import draw_debug_face, draw_results

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(EXTS))
    return [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--output", default="output", help="output folder")
    ap.add_argument("--min-confidence", type=float, default=0.90)
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--debug", action="store_true", help="save aligned faces with ROIs")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.output, exist_ok=True)
    rec = FaceExpressionRecognizer(min_confidence=a.min_confidence)
    report = {}
    for path in collect(a.image):
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue
        res = rec.analyze(img, keep_aligned=a.debug)
        name = os.path.splitext(os.path.basename(path))[0]
        if a.debug:
            for r in res:
                cv2.imwrite(os.path.join(a.output, f"{name}_face{r['id']}_debug.png"),
                            draw_debug_face(r.pop("_aligned"), r.pop("_rois")))
        out = draw_results(img, res, landmarks=not a.no_landmarks)
        cv2.imwrite(os.path.join(a.output, f"{name}_result.jpg"), out)
        report[os.path.basename(path)] = res
        print(f"{os.path.basename(path)}: {len(res)} face(s)")
        for r in res:
            print(f"  face {r['id']}: {r['emotion']} ({r['emotion_confidence']:.2f})  {r['scores']}")
        if a.show:
            cv2.imshow(name, out)
            cv2.waitKey(0)
    with open(os.path.join(a.output, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved results to {a.output}/")


if __name__ == "__main__":
    main()
