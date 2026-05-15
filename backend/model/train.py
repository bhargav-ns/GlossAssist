import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from model.model import LSTM, RNN
from model.utils import build_tokenizer, save_tokenizer


def train(
    data_path,
    model_type="LSTM",
    tokenization_type="char",
    seq_len=100,
    batch_size=64,
    emb_dim=64,
    hidden_dim=128,
    num_layers=2,
    lr=0.01,
    epochs=20,
    on_epoch_end=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(data_path) as f:
        text = f.read()

    tokenize, tokenizer = build_tokenizer(tokenization_type, text)
    tokens = tokenize(text)
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)

    token_to_idx = {t: i for i, t in enumerate(vocab)}
    idx_to_token = {i: t for t, i in token_to_idx.items()}

    data = torch.tensor([token_to_idx[t] for t in tokens], dtype=torch.long)

    sequences, targets = [], []
    for i in range(0, len(data) - seq_len, 1):
        sequences.append(data[i:i + seq_len])
        targets.append(data[i + 1:i + seq_len + 1])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.stack(sequences), torch.stack(targets)),
        batch_size=batch_size,
        shuffle=True,
    )

    if model_type == "LSTM":
        model = LSTM(vocab_size, emb_dim, hidden_dim, num_layers).to(device)
    elif model_type == "RNN":
        model = RNN(vocab_size, emb_dim, hidden_dim, num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        for epoch in range(epochs):
            total_loss = 0
            bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for x, y in bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output, _ = model(x)
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                bar.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} — Avg Loss: {avg_loss:.4f}")
            if on_epoch_end:
                on_epoch_end(epoch + 1, avg_loss)
    finally:
        if tokenizer:
            save_tokenizer(tokenizer, "tokenizer.json")
        torch.save({
            "model_state": model.state_dict(),
            "token_to_idx": token_to_idx,
            "idx_to_token": idx_to_token,
            "vocab_size": vocab_size,
            "model_type": model_type,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "tokenization_type": tokenization_type,
            "tokenizer_path": "tokenizer.json" if tokenizer else None,
        }, "model.pt")
        print("Saved model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--model_type", choices=["LSTM", "RNN"], default="LSTM")
    parser.add_argument("--tokenization_type", choices=["char", "whitespace", "bpe"], default="char")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(
        args.data_path,
        model_type=args.model_type,
        tokenization_type=args.tokenization_type,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        epochs=args.epochs,
    )