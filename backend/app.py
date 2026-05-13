from flask import Flask, request, jsonify
from flask_cors import CORS

from model.model import LSTM
from model.predict import load_model, predict
from model.train import train

import os
import sys
import threading


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "model/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_file_path = None

model, token_to_idx, idx_to_token, tokenize = load_model("model/model.pt")

@app.route("/", methods = ["GET"])
def home():
    return "Ok"

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_file_path

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    uploaded_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(uploaded_file_path)
    return jsonify({"message": "File uploaded successfully", "path": uploaded_file_path})


@app.route("/train", methods=["POST"])
def start_train():
    global uploaded_file_path

    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({"error": "No file uploaded yet"}), 400

    def run_training():
        train(uploaded_file_path)

    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Training started"})


@app.route("/message", methods = ["POST"])
def message():
    data = request.get_json()
    user_text = data.get("text", "")
    
    out = predict(model, token_to_idx, idx_to_token, tokenize, user_text)

    return jsonify({"text": out})



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)