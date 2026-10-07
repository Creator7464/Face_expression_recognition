
from dataclasses import dataclass


@dataclass
class Thresholds:
    # smile / frown from mouth-corner curvature (positive = corners above lip line)
    smile_curv: tuple = (0.02, 0.10)
    frown_curv: tuple = (0.00, 0.06)
    # mouth width / IED
    wide_mouth: tuple = (0.85, 1.05)
    narrow_mouth_ref: float = 0.80      # width below this counts as "narrow"
    narrow_mouth_span: float = 0.12
    # visible teeth (fraction of mouth ROI pixels)
    teeth: tuple = (0.02, 0.15)
    # mouth opening (dark band thickness / IED)
    mouth_open: tuple = (0.08, 0.22)
    # brow-to-eye distance / IED
    brow_raised: tuple = (0.45, 0.60)
    brow_lowered_ref: float = 0.38
    brow_lowered_span: float = 0.08
    # furrow energy between eyebrows
    furrow: tuple = (0.10, 0.35)
    # visible white of the eyes
    sclera: tuple = (0.10, 0.35)
    # redness (Lab a* difference vs. face median)
    cheek_red: tuple = (2.0, 8.0)
    eye_red: tuple = (1.5, 6.0)
    # neutral baseline
    neutral_base: float = 0.25
    neutral_gain: float = 0.35


# Original thresholds, tuned for sharp, colour, full-size frontal faces.
GENERIC = Thresholds()


FER2013 = Thresholds(
    smile_curv=(0.30, 0.40),
    frown_curv=(0.30, 0.40),
    wide_mouth=(0.84, 0.98),
    narrow_mouth_ref=0.80,
    narrow_mouth_span=0.12,
    teeth=(0.02, 0.15),
    mouth_open=(0.06, 0.14),
    brow_raised=(0.90, 1.00),
    brow_lowered_ref=0.0,
    brow_lowered_span=0.08,
    furrow=(0.01, 0.04),
    sclera=(0.01, 0.10),
    neutral_base=0.0,
    neutral_gain=0.30,
)

DEFAULT = FER2013   