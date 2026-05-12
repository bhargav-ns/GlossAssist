import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from model import LSTM
from utils import build_tokenizer, save_tokenizer


def train(data_path, tokenization_type="char"):
    with open(data_path) as f:
        text = f.read()

    tokenize, tokenizer = build_tokenizer(tokenization_type, text)
    tokens = tokenize(text)
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)

    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    data = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long)

    seq_len = 100
    batch_size = 64
    emb_dim = 64
    hidden_dim = 128
    num_layers = 2
    lr = 0.01
    epochs = 20

    sequences, targets = [], []
    for i in range(0, len(data) - seq_len - 1, seq_len):
        sequences.append(data[i:i + seq_len])
        targets.append(data[i + 1:i + seq_len + 1])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.stack(sequences), torch.stack(targets)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LSTM(vocab_size, emb_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x, y in bar:
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}/{epochs} — Avg Loss: {total_loss / len(loader):.4f}")

    save_tokenizer(tokenizer, "tokenizer.json")
    torch.save({
        "model_state": model.state_dict(),
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "tokenization_type": tokenization_type,
        "tokenizer_path": "tokenizer.json" if tokenizer else None,
    }, "model.pt")
    print("Saved model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--tokenization_type", choices=["char", "whitespace", "bpe"], default="char")
    args = parser.parse_args()
    train(args.data_path, args.tokenization_type)