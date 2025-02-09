from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("gender_classifier.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Gender Prediction API! Use POST /predict with {'name': 'John'}"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    name = data.get("name", "")
    seq = tokenizer.texts_to_sequences([name])
    seq_padded = pad_sequences(seq, maxlen=10)
    prediction = model.predict(seq_padded)
    gender = "Male" if prediction[0][0] > 0.5 else "Female"
    return jsonify({"name": name, "predicted_gender": gender})

if __name__ == "__main__":
    app.run(debug=True)
