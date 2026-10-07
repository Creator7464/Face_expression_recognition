
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fer.evaluate import LABELS, evaluate  # noqa: E402
from fer.recognizer import FaceExpressionRecognizer  # noqa: E402
from fer.visualize import draw_debug_face, draw_results  # noqa: E402

st.set_page_config(page_title="Face Expression Recognition", layout="wide")
st.title("Face Expression Recognition")

with st.sidebar:
    st.header("Settings")
    use_mtcnn = st.toggle("Use MTCNN detection", value=True,
                          help="Off = treat each image as one cropped face (for tiny datasets).")
    min_conf = st.slider("Min detection confidence", 0.5, 0.99, 0.90, 0.01)
    show_lm = st.checkbox("Show landmarks", value=True)
    show_debug = st.checkbox("Show aligned faces + regions", value=False)


@st.cache_resource(show_spinner="Loading MTCNN...")
def get_recognizer(use_mtcnn, min_conf):
    return FaceExpressionRecognizer(min_confidence=min_conf, use_mtcnn=use_mtcnn)


rec = get_recognizer(use_mtcnn, min_conf)
tab_img, tab_eval = st.tabs(["Analyze image", "Evaluate on dataset"])

with tab_img:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        res = rec.analyze(bgr, keep_aligned=show_debug)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(cv2.cvtColor(draw_results(bgr, res, show_lm), cv2.COLOR_BGR2RGB),
                     caption=f"{len(res)} face(s) detected", use_container_width=True)
        with c2:
            if not res:
                st.warning("No faces found. Try a lower confidence or turn MTCNN off for cropped faces.")
            counts = pd.Series([r["emotion"] for r in res]).value_counts()
            if len(counts):
                st.subheader("Emotions in image")
                st.bar_chart(counts)
        for r in res:
            with st.expander(f"Face {r['id']}: {r['emotion']} ({r['emotion_confidence']:.2f})",
                             expanded=len(res) <= 3):
                a, b, c = st.columns(3)
                a.write("**Emotion scores**")
                a.bar_chart(pd.Series(r["scores"]))
                b.write("**Measured features**")
                b.dataframe(pd.Series(r["features"], name="value"))
                if show_debug:
                    c.image(cv2.cvtColor(draw_debug_face(r["_aligned"], r["_rois"]),
                                         cv2.COLOR_BGR2RGB), caption="Aligned face + ROIs")

with tab_eval:
    st.write("Folder layout: `root/<emotion>/image.jpg` "
             "(anger/angry, sad, happy, fear, neutral). See `datasets/README.md`.")
    root = st.text_input("Dataset folder", "datasets/synthetic")
    n = st.number_input("Max images per class", 1, 5000, 100)
    if st.button("Run evaluation", type="primary"):
        if not os.path.isdir(root):
            st.error("Folder not found.")
        else:
            bar = st.progress(0.0)
            m = evaluate(rec, root, int(n), assume_face=True, progress=bar.progress)
            st.metric("Accuracy", f"{m['accuracy']:.1%}", f"{m['n']} images, "
                      f"{m['skipped_no_face']} skipped")
            cm = pd.DataFrame(m["confusion"]).T[LABELS]
            l, r_ = st.columns(2)
            l.write("**Confusion matrix** (rows = true, cols = predicted)")
            l.dataframe(cm.style.background_gradient(cmap="Blues"))
            r_.write("**Per-class F1**")
            r_.bar_chart(pd.Series({k: v["f1"] for k, v in m["per_class"].items()}))
            st.write("**Per-class metrics**")
            st.dataframe(pd.DataFrame(m["per_class"]).T)
