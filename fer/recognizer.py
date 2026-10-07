from .classifier import classify_features
from .detector import FaceDetector, whole_image_face
from .features import extract_features


class FaceExpressionRecognizer:
    
    def __init__(self, min_confidence=0.90, thresholds=None, use_mtcnn=True):
        self.detector = FaceDetector(min_confidence=min_confidence) if use_mtcnn else None
        self.thresholds = thresholds

    def analyze(self, bgr, keep_aligned=False, assume_face=False):
        """assume_face: if MTCNN finds nothing, classify the whole image as a face."""
        faces = self.detector.detect(bgr) if self.detector else []
        if not faces and (assume_face or self.detector is None):
            faces = [whole_image_face(bgr)]
        results = []
        for i, face in enumerate(faces):
            feats, aligned, rois = extract_features(bgr, face.landmarks)
            args = (feats,) if self.thresholds is None else (feats, self.thresholds)
            label, conf, scores = classify_features(*args)
            r = {"id": i, "box": [int(v) for v in face.box],
                 "detection_confidence": round(face.confidence, 3),
                 "emotion": label, "emotion_confidence": round(conf, 3),
                 "scores": scores, "features": feats,
                 "landmarks": face.landmarks}
            if keep_aligned:
                r["_aligned"], r["_rois"] = aligned, rois
            results.append(r)
        return results
