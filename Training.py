import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

#load the csv
df = pd.read_csv("Collated.csv", names=["Comment", "Classification"], encoding="ISO-8859-1")
df = df.dropna() #drop empty rows

#convert my nice labels to binary
label_map = {"not_cyberbullying": 0, "other_cyberbullying": 1}
df["Classification"] = df["Classification"].map(label_map)

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

#build gru model - code stolen from website
model = Sequential([
    Embedding(max_features, 128, input_length=maxlen),
    GRU(64, return_sequences=True),
    GRU(32),
    Dense(1, activation='sigmoid')
])

#compile, train, and save
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2)
model.save("cyberbullying_model.h5")

#save for decoding
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
