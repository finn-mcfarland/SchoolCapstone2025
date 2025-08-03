import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
import re
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#load the csv
df = pd.read_csv("Collated.csv", names=["Comment", "Classification"], encoding="ISO-8859-1")
df = df.dropna() #drop empty rows

#convert my nice labels to binary
label_map = {"not_cyberbullying": 0, "other_cyberbullying": 1}
df["Classification"] = df["Classification"].map(label_map)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove @mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-z\s]", "", text) # Remove punctuation/numbers
    return text.strip()

df["Comment"] = df["Comment"].astype(str).apply(clean_text)

#get the two columns seperatly
texts = df["Comment"].astype(str).tolist()
labels = df["Classification"].values


#tokenise the text and pad the sequences for efficient computation 
max_features = 20000
maxlen = 100
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# === Use sklearn's train_test_split ===
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

#build gru model - code stolen from website
model = Sequential([
    Embedding(max_features, 128, input_length=maxlen),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

#compile, train, and save
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32, callbacks=[early_stop])
model.save("cyberbullying_model.h5")

#save for decoding
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
