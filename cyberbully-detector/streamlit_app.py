import streamlit as st
import sys
from pathlib import Path

# Add the 'app' folder to Python's path
sys.path.append(str(Path(__file__).resolve().parent / "app"))

from model import load_model, predict_text  # model.py inside app/

@st.cache_resource
def get_model():
    return load_model()  # this uses the default: unitary/toxic-bert

model, tokenizer = get_model()

st.title("Cyberbullying Detection Dashboard")
text = st.text_area("Enter a comment or message:")

if st.button("Analyze"):
    label, confidence = predict_text(text, model, tokenizer)
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
