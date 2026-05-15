from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os

from model.train import train
from model.predict import load_model, predict

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "model/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_file_path = None
training_status = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "done": False,
    "error": None,
}

try:
    model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len = load_model(
        'model/model.pt')
except FileNotFoundError:
    model = token_to_idx = idx_to_token = tokenize = tokenization_type = seq_len = None


@app.route("/", methods=["GET"])
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
    return jsonify({"message": "File uploaded successfully"})


@app.route("/train", methods=["POST"])
def start_train():
    global uploaded_file_path, training_status

    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({"error": "No file uploaded yet"}), 400

    if training_status["running"]:
        return jsonify({"error": "Training already in progress"}), 400

    data = request.get_json()

    # normalize "lstm" -> "LSTM"
    model_type = data.get("model", "lstm").upper()
    if model_type not in ("LSTM", "RNN"):
        return jsonify({"error": f"Invalid model type: {model_type}"}), 400

    hyperparams = {
        "model_type": model_type,
        "epochs": int(data.get("epochs", 20)),
        "lr": float(data.get("lr", 0.01)),
        "batch_size": int(data.get("batch_size", 64)),
        "emb_dim": int(data.get("emb_dim", 64)),
        "hidden_dim": int(data.get("hidden_dim", 128)),
        "num_layers": int(data.get("num_layers", 2)),
        "seq_len": int(data.get("seq_len", 100)),
        "tokenization_type": data.get("tokenization_type", "char"),
    }

    training_status = {
        "running": True,
        "epoch": 0,
        "total_epochs": hyperparams["epochs"],
        "loss": None,
        "done": False,
        "error": None,
    }

    def run_training():
        global training_status, model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len
        try:
            train(
                uploaded_file_path,
                model_type=hyperparams["model_type"],
                tokenization_type=hyperparams["tokenization_type"],
                seq_len=hyperparams["seq_len"],
                batch_size=hyperparams["batch_size"],
                emb_dim=hyperparams["emb_dim"],
                hidden_dim=hyperparams["hidden_dim"],
                num_layers=hyperparams["num_layers"],
                lr=hyperparams["lr"],
                epochs=hyperparams["epochs"],
                on_epoch_end=lambda epoch, loss: training_status.update({
                    "epoch": epoch,
                    "loss": round(loss, 4),
                }),
            )
            model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len = load_model(
                'model/model.pt', model_type=hyperparams["model_type"]
            )
            training_status.update({"running": False, "done": True})
        except Exception as e:
            training_status.update({"running": False, "error": str(e)})

    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Training started"})


@app.route("/train/status", methods=["GET"])
def train_status():
    return jsonify(training_status)


@app.route("/message", methods=["POST"])
def message():
    if model is None:
        return jsonify({"error": "No model loaded. Train a model first."}), 503

    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    out = predict(model, token_to_idx, idx_to_token, tokenize,
                  tokenization_type, seq_len, user_text)
    return jsonify({"text": out})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
