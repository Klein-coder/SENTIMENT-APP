# streamlit_app.py
from preprocessing import TextCleaner, LemmaTokenizer
import streamlit as st
import joblib
import os
import numpy as np
import nltk
import re

# download required datasets quietly (only if not present)
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# --- CONFIG ---
PIPELINE_PATH = "sentiment_pipeline_raw_bigram.joblib"
MAX_TEXT_LENGTH = 2000

st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("Sentiment Analysis App 😊")
st.write("Type some text and click **Predict**. (Model loaded with joblib)")

# --- Negation override (quick inference-time fix)
_CONTRACTIONS_NO_APOST = {
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt",
    "can't": "cant", "won't": "wont", "isn't": "isnt",
    "aren't": "arent", "wasn't": "wasnt", "weren't": "werent",
    "couldn't": "couldnt", "shouldn't": "shouldnt", "wouldn't": "wouldnt",
    "haven't": "havent", "hasn't": "hasnt", "hadn't": "hadnt",
    "i'm": "im", "i've": "ive", "i'll": "ill"
}
def normalize_contractions_to_training(text: str) -> str:
    txt = text.lower()
    for k, v in _CONTRACTIONS_NO_APOST.items():
        txt = re.sub(rf"\b{re.escape(k)}\b", v, txt)
    return txt

NEG_OVERRIDE = {
    "dont like", "do not like", "didnt like", "cannot stand",
    "not worth", "not good", "not recommend", "not happy"
}
def postprocess(text: str, pred: int, prob_positive: float):
    txt = normalize_contractions_to_training(text)
    for phrase in NEG_OVERRIDE:
        if phrase in txt:
            return 0, max(0.0, min(1.0, 1.0 - prob_positive))
    return pred, prob_positive

# --- Load pipeline (cached so it doesn't reload on every interaction)
@st.cache_resource(show_spinner=False)
def load_pipeline(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipeline = load_pipeline(PIPELINE_PATH)

if pipeline is None:
    st.warning(
        f"Pipeline file not found at `{PIPELINE_PATH}`. "
        "You can upload it below or place it in the app folder."
    )
    uploaded = st.file_uploader("Upload `sentiment_pipeline_raw_bigram.joblib`", type=["joblib"])
    if uploaded:
        with open(PIPELINE_PATH, "wb") as f:
            f.write(uploaded.getbuffer())
        pipeline = load_pipeline(PIPELINE_PATH)
        st.success("Model uploaded and loaded ✅")

st.markdown("----")

# Input: single text or multiple (one per line)
mode = st.radio("Input mode", ("Single text", "Batch (one per line)"))

if mode == "Single text":
    text = st.text_area("Enter text to classify", height=150, placeholder="I love this product...")
    texts_list = [text.strip()] if text.strip() else []
else:
    batch = st.text_area("Enter multiple texts (one per line)", height=200,
                         placeholder="I love it\nThis is terrible\nSo-so product")
    texts_list = [t.strip() for t in batch.splitlines() if t.strip()]

st.write("")  # spacing

if st.button("Predict"):

    if not pipeline:
        st.error("Model not loaded. Upload the joblib file or ensure the file exists.")
    elif not texts_list:
        st.error("Please enter at least one text to classify.")
    else:
        for t in texts_list:
            if len(t) > MAX_TEXT_LENGTH:
                st.error(f"Each text must be <= {MAX_TEXT_LENGTH} characters. One item is too long.")
                st.stop()

        try:
            preds = pipeline.predict(texts_list)
            probs = pipeline.predict_proba(texts_list)[:, 1]
        except Exception as e:
            st.exception(f"Prediction failed: {e}")
            st.stop()

        # --- Apply postprocess override
        preds_adj, probs_adj = [], []
        for txt, p, pr in zip(texts_list, preds, probs):
            p_adj, pr_adj = postprocess(txt, int(p), float(pr))
            preds_adj.append(p_adj)
            probs_adj.append(pr_adj)

        # Show results
        import pandas as pd
        df = pd.DataFrame({
            "text": texts_list,
            "prediction": preds_adj,
            "prob_positive": probs_adj
        })
        df["label"] = df["prediction"].apply(lambda x: "Positive 😀" if x == 1 else "Negative 😡")
        df = df[["text", "label", "prob_positive"]]

        st.success("Predictions ready ✅")
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name="predictions.csv", mime="text/csv")

