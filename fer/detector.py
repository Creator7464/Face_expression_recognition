"""Multi-face detection with MTCNN."""
from dataclasses import dataclass, field

import cv2


@dataclass
class Face:
    box: tuple                      # (x, y, w, h)
    confidence: float
    landmarks: dict = field(default_factory=dict)  # name -> (x, y)


class FaceDetector:
    def __init__(self, min_confidence=0.90, min_face_size=24):
        self._mtcnn = None          # loaded on first use (TensorFlow is heavy)
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size

    def detect(self, bgr):
        if self._mtcnn is None:
            from mtcnn import MTCNN
            self._mtcnn = MTCNN()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = []
        for d in self._mtcnn.detect_faces(rgb):
            if d["confidence"] < self.min_confidence:
                continue
            x, y, w, h = d["box"]
            if min(w, h) < self.min_face_size:
                continue
            faces.append(Face(
                box=(max(x, 0), max(y, 0), w, h),
                confidence=float(d["confidence"]),
                landmarks={k: (float(v[0]), float(v[1]))
                           for k, v in d["keypoints"].items()},
            ))
        return faces


def whole_image_face(bgr):
    """Treat the whole image as one cropped, frontal face (e.g. FER-2013 48x48
    images, which MTCNN often cannot detect). Landmarks are placed at typical
    relative positions, so width-based features carry little information."""
    h, w = bgr.shape[:2]
    lm = {"left_eye": (0.30 * w, 0.38 * h), "right_eye": (0.70 * w, 0.38 * h),
          "nose": (0.50 * w, 0.58 * h),
          "mouth_left": (0.36 * w, 0.78 * h), "mouth_right": (0.64 * w, 0.78 * h)}
    return Face(box=(0, 0, w, h), confidence=0.0, landmarks=lm)
