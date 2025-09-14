# app.py

 
from preprocessing import TextCleaner, LemmaTokenizer
import joblib

from flask import Flask, request, jsonify
import joblib
import logging
import traceback

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)



# Load pipeline once at startup
PIPELINE_PATH = "sentiment_pipeline_raw_bigram.joblib"  # <- make sure this file is in the same folder
pipeline = joblib.load(PIPELINE_PATH)

# Optional: set a maximum input length (characters) to avoid huge requests
MAX_TEXT_LENGTH = 2000

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        texts_list = []

        # Case 1: JSON input
        if request.is_json:
            data = request.get_json()
            if data is None or "text" not in data:
                return jsonify({"error": "JSON body must contain 'text' field"}), 400
            texts = data["text"]
            texts_list = [texts] if isinstance(texts, str) else texts

        # Case 2: Form input (from browser form)
        elif "text" in request.form:
            texts = request.form.get("text")
            texts_list = [texts]

        else:
            return jsonify({"error": "Invalid input. Send JSON {'text': ...} or form-data with 'text'"}), 400

        # Validate length
        for t in texts_list:
            if not isinstance(t, str):
                return jsonify({"error": "each item must be a string"}), 400
            if len(t) > MAX_TEXT_LENGTH:
                return jsonify({"error": f"each text must be <= {MAX_TEXT_LENGTH} characters"}), 400

        # Predict
        preds = pipeline.predict(texts_list)
        probs = pipeline.predict_proba(texts_list)[:, 1].tolist()

        results = []
        for txt, p, pr in zip(texts_list, preds, probs):
            results.append({
                "text": txt,
                "prediction": int(p),
                "prob_positive": float(pr)
            })

        # If request was JSON → return JSON
        if request.is_json:
            return jsonify({"results": results}), 200
        # If request was form → render HTML
        else:
            html_out = f"""
            <h2>Prediction Result</h2>
            <p><b>Text:</b> {results[0]['text']}</p>
            <p><b>Prediction:</b> {"Positive 😀" if results[0]['prediction']==1 else "Negative 😡"}</p>
            <p><b>Probability (Positive):</b> {results[0]['prob_positive']:.3f}</p>
            <a href="/">Back</a>
            """
            return html_out, 200

    except Exception as e:
        logging.error("Prediction error: %s", traceback.format_exc())
        return jsonify({"error": "internal error", "details": str(e)}), 500

    
# ✅ INSERT THIS NEW ROUTE HERE
@app.route("/", methods=["GET"])
def index():
    return """
    <h1>Sentiment Analysis App</h1>
    <p>Send a POST request to <code>/predict</code> with JSON like:</p>
    <pre>{ "text": "I love this phone" }</pre>
    <p>Or try the form below:</p>
    <form action="/predict" method="post">

      <textarea name="text" rows="4" cols="50" placeholder="Type a review..."></textarea><br><br>
      <button type="submit">Predict</button>
    </form>
    """

    
for rule in app.url_map.iter_rules():
    print(rule, list(rule.methods) if rule.methods is not None else [])

if __name__ == "__main__":
    # For development only. In production use Gunicorn/Uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
    

