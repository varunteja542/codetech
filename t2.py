import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

df = pd.read_csv("cleaned_people_data.csv")

df["Text"] = df["First Name"] + " " + df["Job Title"]
texts = df["Text"].values
labels = df["Sex"].values.astype(np.float32)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)

with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

model = Sequential([
    Embedding(5000, 32, input_length=10),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, labels, epochs=10, batch_size=32, validation_split=0.2)

model.save("gender_classifier.h5")
print("✅ Model training complete. Saved as 'gender_classifier.h5'.")
