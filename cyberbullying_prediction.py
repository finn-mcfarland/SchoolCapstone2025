import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

#load the model
model = load_model("cyberbullying_model.h5")

#load the tokeniser for decoding
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 100 #must match the value in training.py

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
