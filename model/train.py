import torch
import torch.nn as nn
from model import CharLSTM
import argparse
from tqdm import tqdm

def train(data_path):
    
    # Read the file
    with open(data_path, 'r') as f:
        text = f.read()

    # Create the vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {c:i for i, c in enumerate(chars)}
    idx_to_char = {i:c for c, i in char_to_idx.items()}

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    # Set hyperparams
    seq_len = 100
    batch_size = 64
    embed_dim = 64
    hidden = 128
    lr = 0.01
    num_layers = 2
    epochs = 20

    # Build sequences for training
    sequences, targets = [], []
    for i in range(0, len(data) - seq_len - 1, seq_len):
        sequences.append(data[i:i+seq_len])
        targets.append(data[i+1:i+seq_len+1])

    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    dataset = torch.utils.data.TensorDataset(sequences, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Set model, loss, optimizer
    model = CharLSTM(vocab_size, embed_dim, hidden, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in progress_bar:
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {total_loss/len(loader):.4f}")


    # save model

    torch.save({
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden,
        "num_layers": num_layers,
    }, "model.pt")

    print("Saved model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model with source data")
    parser.add_argument("data_path", type=str, help="Path to the training data file")
    args = parser.parse_args()
    train(args.data_path)