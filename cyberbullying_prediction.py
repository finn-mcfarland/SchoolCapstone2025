import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# === Load model and tokenizer ===
model = load_model("cyberbullying_model.h5")

# Load the tokenizer (must be saved during training)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 100  # Must match training

print("Enter text to classify (type 'exit' to quit):\n")

while True:
    text = input(">> ")
    if text.lower() == "exit":
        break

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]

    label = "Cyberbullying" if pred > 0.5 else "Not Cyberbullying"
    print(f"Prediction: {label} ({pred:.2f} confidence)\n")
