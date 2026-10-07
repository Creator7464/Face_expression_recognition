# Datasets for testing

## 1. Synthetic (included, generated)
```bash
python datasets/make_synthetic.py --out datasets/synthetic --per-class 20
python evaluate.py --data datasets/synthetic --no-mtcnn
```
Cartoon faces: a **pipeline smoke test only**, not a benchmark (MTCNN will not
detect them, hence `--no-mtcnn`). Real accuracy must be measured on real faces.

## 2. Real datasets (download yourself, free)
| Dataset | Notes | Link |
|---|---|---|
| **FER-2013** | 48x48 grayscale, ~35k images, folders `train/angry,fear,happy,neutral,sad,...` | kaggle.com/datasets/msambare/fer2013 |
| **CK+ (48x48)** | Posed lab expressions, easy and clean | kaggle.com/datasets/shawon10/ckplus |
| **JAFFE** | 213 images, 10 Japanese women | zenodo.org/record/3451524 |
| **RAF-DB** | In-the-wild, 30k images (request access) | whdeng.cn/RAF/model1.html |
| **AffectNet** | Largest in-the-wild set (request access) | mohammadmahoor.com/affectnet |

Unpack so each emotion has its own folder, for example:
```
datasets/fer2013/test/angry/*.jpg   datasets/fer2013/test/happy/*.jpg ...
```
`angry -> anger` and similar aliases are mapped automatically; classes this
project doesn't have (disgust, surprise, contempt) are skipped.

```bash
python evaluate.py --data datasets/fer2013/test --max-per-class 200
```
Tiny cropped sets like FER-2013 and CK+ are usually not detected by MTCNN, so
those images fall back to "whole image = one face" (landmarks at typical
positions, so mouth-width features are uninformative). For full-size photos
MTCNN is used automatically. Check dataset licences before redistributing.
