import cv2

COLORS = {"anger": (0, 0, 220), "sad": (220, 120, 0), "happy": (0, 200, 0),
          "fear": (200, 0, 200), "neutral": (160, 160, 160)}


def draw_results(bgr, results, landmarks=True):
    out = bgr.copy()
    scale = max(out.shape[:2]) / 900.0
    th = max(1, int(round(2 * scale)))
    for r in results:
        x, y, w, h = r["box"]
        c = COLORS[r["emotion"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, th)
        text = f'{r["emotion"]} {r["emotion_confidence"]:.2f}'
        fs = max(0.4, 0.6 * scale)
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        top = max(y - tht - 8, 0)
        cv2.rectangle(out, (x, top), (x + tw + 6, top + tht + 8), c, -1)
        cv2.putText(out, text, (x + 3, top + tht + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), th, cv2.LINE_AA)
        if landmarks:
            for px, py in r["landmarks"].values():
                cv2.circle(out, (int(px), int(py)), max(2, int(3 * scale)), (0, 255, 255), -1)
    return out


def draw_debug_face(aligned, rois):
    img = aligned.copy()
    for box in rois.values():
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      (0, 255, 255), 1)
    return img
