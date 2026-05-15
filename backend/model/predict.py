import os
import torch
import argparse
from model.model import LSTM, RNN
from model.utils import load_tokenizer


def load_model(path="model.pt", model_type="LSTM"):
    if not os.path.exists(path):
        backend_path = os.path.join("backend", path)
        if os.path.exists(backend_path):
            path = backend_path
        else:
            raise FileNotFoundError(f"Model not found at {path} or {backend_path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if model_type == "LSTM":
        model = LSTM(
            checkpoint["vocab_size"],
            checkpoint["emb_dim"],
            checkpoint["hidden_dim"],
            checkpoint["num_layers"],
        )
    elif model_type == "RNN":
        model = RNN(
            checkpoint["vocab_size"],
            checkpoint["emb_dim"],
            checkpoint["hidden_dim"],
            checkpoint["num_layers"],
        )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenization_type = checkpoint["tokenization_type"]
    tokenizer_path = checkpoint.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(path)), tokenizer_path)

    tokenize = load_tokenizer(tokenization_type, tokenizer_path)
    seq_len = checkpoint.get("seq_len", 100)

    return model, checkpoint["token_to_idx"], checkpoint["idx_to_token"], tokenize, tokenization_type, seq_len


def predict(model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len, seed_text, length=200, temperature=0.8):
    tokens = [token_to_idx.get(t) for t in tokenize(seed_text)]
    tokens = [t for t in tokens if t is not None]

    context = torch.tensor([tokens], dtype=torch.long)
    sep = "" if tokenization_type == "char" else " "
    generated = seed_text
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(context, hidden)
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = idx_to_token[next_idx]
            generated += sep + next_token
            context = torch.cat([context, torch.tensor([[next_idx]])], dim=1)[:, -seq_len:]

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_text", type=str)
    parser.add_argument("--model_type", choices=["LSTM", "RNN"], default="LSTM")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--path", type=str, default="model.pt")
    args = parser.parse_args()

    model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len = load_model(args.path, args.model_type)
    print(predict(model, token_to_idx, idx_to_token, tokenize, tokenization_type, seq_len, args.seed_text, args.length, args.temperature))