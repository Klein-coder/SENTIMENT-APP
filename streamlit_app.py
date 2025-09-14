# streamlit_app.py
from preprocessing import TextCleaner, LemmaTokenizer
import streamlit as st
import joblib
import os
import numpy as np

# --- CONFIG ---
PIPELINE_PATH = "sentiment_pipeline_raw_bigram.joblib"
MAX_TEXT_LENGTH = 2000

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

st.title("Sentiment Analysis App 😊")
st.write("Type some text and click **Predict**. (Model loaded with joblib)")

# Load pipeline (cache so it doesn't reload on every interaction)
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
        # save uploaded file and load
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
        # validate length
        for t in texts_list:
            if len(t) > MAX_TEXT_LENGTH:
                st.error(f"Each text must be <= {MAX_TEXT_LENGTH} characters. One item is too long.")
                st.stop()

        # run predictions
        try:
            preds = pipeline.predict(texts_list)
            probs = pipeline.predict_proba(texts_list)[:, 1]
        except Exception as e:
            st.exception(f"Prediction failed: {e}")
            st.stop()

        # show results
        import pandas as pd
        df = pd.DataFrame({
            "text": texts_list,
            "prediction": preds.astype(int),
            "prob_positive": probs
        })
        # pretty prediction label
        df["label"] = df["prediction"].apply(lambda x: "Positive 😀" if x == 1 else "Negative 😡")
        # reorder columns
        df = df[["text", "label", "prob_positive"]]

        st.success("Predictions ready ✅")
        st.dataframe(df, use_container_width=True)

        # Show a download button for results as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name="predictions.csv", mime="text/csv")

