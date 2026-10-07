import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fer.classifier import classify_features as _classify  # noqa: E402
from fer.config import GENERIC  # noqa: E402
from fer.features import IED, align_face, extract_features  # noqa: E402

def classify_features(f):
    return _classify(f, GENERIC)


BASE = dict(mouth_width=0.80, curvature=0.0, mouth_open=0.06, teeth=0.0,
            furrow=0.05, brow_gap=0.41, sclera=0.05, cheek_red=0.0, eye_red=0.0)


def feats(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


class ClassifierTests(unittest.TestCase):
    def test_neutral(self):
        self.assertEqual(classify_features(feats())[0], "neutral")

    def test_happy(self):
        f = feats(curvature=0.10, mouth_width=1.0, teeth=0.12, cheek_red=4)
        self.assertEqual(classify_features(f)[0], "happy")

    def test_sad(self):
        f = feats(curvature=-0.07, mouth_width=0.70, eye_red=4)
        self.assertEqual(classify_features(f)[0], "sad")

    def test_anger(self):
        f = feats(furrow=0.35, brow_gap=0.30, cheek_red=7)
        self.assertEqual(classify_features(f)[0], "anger")

    def test_fear(self):
        f = feats(sclera=0.30, brow_gap=0.58, mouth_open=0.20)
        self.assertEqual(classify_features(f)[0], "fear")


class GeometryTests(unittest.TestCase):
    LM = {"left_eye": (100, 120), "right_eye": (160, 120), "nose": (130, 160),
          "mouth_left": (108, 200), "mouth_right": (152, 200)}

    def test_alignment_levels_eyes(self):
        lm = dict(self.LM, left_eye=(100, 100), right_eye=(160, 130))  # tilted
        _, p = align_face(np.full((300, 300, 3), 128, np.uint8), lm)
        self.assertAlmostEqual(p["le"][1], p["re"][1], places=3)
        self.assertAlmostEqual(p["re"][0] - p["le"][0], IED, places=3)

    def test_feature_extraction_runs(self):
        img = np.random.randint(0, 255, (300, 300, 3), np.uint8)
        f, face, rois = extract_features(img, self.LM)
        self.assertEqual(set(f), set(BASE))
        self.assertEqual(face.shape[:2], (256, 256))


class EvaluateTests(unittest.TestCase):
    def test_metrics(self):
        from fer.evaluate import compute_metrics
        m = compute_metrics(["happy", "happy", "sad", "fear"], ["happy", "sad", "sad", "fear"])
        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(m["confusion"]["happy"]["sad"], 1)
        self.assertEqual(m["per_class"]["sad"]["recall"], 1.0)

    def test_synthetic_end_to_end(self):
        import tempfile
        from fer.evaluate import evaluate
        from fer.recognizer import FaceExpressionRecognizer
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets"))
        from ..dashboard import make_synthetic
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            import cv2
            for emo in ["happy", "sad"]:
                os.makedirs(os.path.join(d, emo))
                for k in range(3):
                    cv2.imwrite(os.path.join(d, emo, f"{k}.png"), make_synthetic.face(emo, rng))
            m = evaluate(FaceExpressionRecognizer(use_mtcnn=False, thresholds=GENERIC), d)
            self.assertEqual(m["n"], 6)
            self.assertGreaterEqual(m["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()