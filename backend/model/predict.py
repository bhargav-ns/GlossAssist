import os
import torch
import argparse
from model.model import LSTM
from model.utils import load_tokenizer


def load_model(path="model.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    model = LSTM(
        checkpoint["vocab_size"],
        checkpoint["emb_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_layers"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenization_type = checkpoint["tokenization_type"]
    tokenizer_path = checkpoint.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(path)), tokenizer_path)

    tokenize = load_tokenizer(tokenization_type, tokenizer_path)
    return model, checkpoint["token_to_idx"], checkpoint["idx_to_token"], tokenize


def predict(model, token_to_idx, idx_to_token, tokenize, seed_text, length=200, temperature=0.8):
    tokens = [token_to_idx.get(t) for t in tokenize(seed_text)]
    tokens = [t for t in tokens if t is not None]
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    generated = seed_text
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = idx_to_token[next_idx]
            generated += "" if tokenize == list else " "
            generated += next_token
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long)

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_text", type=str)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    model, token_to_idx, idx_to_token, tokenize = load_model()
    print(predict(model, token_to_idx, idx_to_token, tokenize, args.seed_text, args.length, args.temperature))