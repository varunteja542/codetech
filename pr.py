import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("gender_classifier.h5")

# Reinitialize tokenizer (Ensure it matches training tokenizer)
tokenizer = Tokenizer(num_words=5000)

# Define a function for prediction
def predict_gender(name):
    # Convert name to sequence
    seq = tokenizer.texts_to_sequences([name])
    seq_padded = pad_sequences(seq, maxlen=10)

    # Make prediction
    prediction = model.predict(seq_padded)
    
    # Interpret result
    gender = "Male" if prediction[0][0] > 0.5 else "Female"
    print(f"Predicted Gender for '{name}': {gender}")

# Test the function with example names
predict_gender("John")
predict_gender("Emily")
predict_gender("Alex")
predict_gender("Samantha")
