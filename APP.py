import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.models import load_model
import pickle
import torch

# Load Models
@st.cache_resource
def load_lstm():
    return keras.models.load_model("best_tuned_toxicity_model.keras")

@st.cache_resource
def load_cnn():
    return keras.models.load_model("best_tuned_cnn_toxicity_model.keras")

@st.cache_resource
def load_distilbert():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert_toxicity_model/")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert_toxicity_model/")
    return model, tokenizer

# Sidebar
st.sidebar.title("Toxicity Detection Models")
model_option = st.sidebar.selectbox("Choose Model:", ["LSTM", "CNN", "DistilBERT"])

st.title("Comment Toxicity Detector")

input_text = st.text_area("Enter a comment for toxicity prediction:")

def predict_lstm(text):
    # Tokenizer parameters (update as per training)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
    # Here you should load the original fitted tokenizer
    # For demo: fit on single text
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=128, padding='post')
    model = load_lstm()
    pred = model.predict(padded)
    return float(pred[0][0])

def predict_cnn(text):
    # Tokenizer parameters (should match training)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=128, padding='post')
    model = load_cnn()
    pred = model.predict(padded)
    return float(pred[0][0])

def predict_distilbert(text):
    model, tokenizer = load_distilbert()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="tf", max_length=128)
    output = model(inputs)
    score = tf.math.sigmoid(output.logits)[0][0].numpy()
    return float(score)

if st.button("Predict Toxicity"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        if model_option == "LSTM":
            score = predict_lstm(input_text)
        elif model_option == "CNN":
            score = predict_cnn(input_text)
        else:
            score = predict_distilbert(input_text)
        st.markdown(f"### Toxicity Score: `{score:.3f}`")
        if score > 0.5:
            st.error("⚠️ This comment is likely TOXIC.")
        else:
            st.success("✅ This comment is likely NON-TOXIC.")

st.markdown("---")
st.write("Or upload a CSV file with a `comment_text` column for bulk prediction.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment_text" not in df.columns:
        st.error("CSV must have a 'comment_text' column.")
    else:
        st.write("First 5 comments preview:")
        st.write(df['comment_text'].head())
        result = []
        for text in df['comment_text']:
            if model_option == "LSTM":
                score = predict_lstm(text)
            elif model_option == "CNN":
                score = predict_cnn(text)
            else:
                score = predict_distilbert(text)
            result.append({"comment_text": text, "toxicity_score": score})
        res_df = pd.DataFrame(result)
        res_df['toxic'] = res_df['toxicity_score'] > 0.5
        st.dataframe(res_df)
        csv = res_df.to_csv(index=False).encode()
        st.download_button("Download Results CSV", csv, "toxicity_results.csv", "text/csv")

st.markdown("---")
st.caption("Place this file (app.py) in the same folder with your model files as shown in your screenshot.")

