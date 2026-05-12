from flask import Flask, request, jsonify
from flask_cors import CORS

from model.model import LSTM
from model.predict import load_model, predict

app = Flask(__name__)
CORS(app)

model, token_to_idx, idx_to_token, tokenize = load_model("model/model.pt")

@app.route("/", methods = ["GET"])
def home():
    return "Ok"


@app.route("/message", methods = ["POST"])
def message():
    data = request.get_json()
    user_text = data.get("text", "")
    
    out = predict(model, token_to_idx, idx_to_token, tokenize, user_text)

    return jsonify({"text": out})

@app.route("/train", methods=["POST"])
def train_model():
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)