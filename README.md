# Face Expression Recognition

Detects the faces of **all people** in an image with **MTCNN** and classifies each
face as **anger, sad, happy, fear or neutral** using a **heuristic** approach that
combines **landmark geometry** and **pixel intensity / colour** (e.g. red cheeks).

## Install & run
```bash
pip install -r requirements.txt
python main.py --image samples/group.jpg            # one image
python main.py --image samples/ --output output     # a folder
python main.py --image samples/group.jpg --debug    # also save aligned faces + ROIs
python -m unittest discover tests                   # run tests
streamlit run dashboard/app.py                      # interactive dashboard
python evaluate.py --data datasets/synthetic --no-mtcnn   # dataset evaluation
```
See `datasets/README.md` for the synthetic set and how to get FER-2013, CK+, JAFFE etc.

## Dashboard
`streamlit run dashboard/app.py` gives two tabs: **Analyze image** (upload, annotated
faces, per-face score charts, measured features, aligned-face ROI view) and
**Evaluate on dataset** (accuracy, confusion matrix, per-class precision/recall/F1).
Outputs: annotated `*_result.jpg` per image and `results.json` (boxes, landmarks,
scores, raw features).

## How it works
1. **MTCNN** returns box, confidence and 5 landmarks per face.
2. **Alignment**: each face is rotated/scaled so the eyes are level and 64 px apart
   (inter-eye distance, IED), making every measurement pose- and size-normalised.
3. **Features** (`fer/features.py`)
   | Feature | Source | Signals |
   |---|---|---|
   | mouth width / IED | landmarks | happy (wide), sad (narrow) |
   | mouth-corner curvature | corners vs. darkest lip-line row (intensity) | smile / frown |
   | mouth opening, teeth ratio | intensity + HSV in mouth ROI | fear, happy |
   | brow-to-eye distance | darkest row above each eye | raised (fear) / lowered (anger) |
   | furrow energy | Sobel-x between brows | anger |
   | sclera ratio | low-saturation bright pixels in eye ROI | fear (wide eyes) |
   | cheek redness | Lab a* of cheeks minus face median | anger / happy (flush) |
   | under-eye redness | Lab a* below eyes | sad (crying) |
4. **Classifier** (`fer/classifier.py`): each emotion gets a weighted score from
   0-1 ramps over those features; neutral is the baseline when nothing is strong.

## Tuning
All thresholds are in `fer/config.py`. Heuristics vary with lighting, skin tone,
glasses/beards and camera quality, so calibrate the ranges on your own images
(use `--debug` to check the ROIs land on the right facial regions).
Note: MTCNN gives only 5 landmarks, so brows and the lip line are estimated from
intensity profiles, which is less precise than a 68-point landmark model.
