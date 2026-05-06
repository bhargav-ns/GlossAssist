import torch
import torch.nn as nn
from model import CharLSTM

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
    for epoch in range(len(epochs)):
        total_loss = 0
        for x,y in loader:
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")


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