import torch
from model import CharLSTM
import argparse

def load_model(path = "model.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    model = CharLSTM(
        checkpoint["vocab_size"],
        checkpoint["embed_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_layers"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["char_to_idx"], checkpoint["idx_to_char"]
    

def predict(model, char_to_idx, idx_to_char, seed_text, length = 200, temperature = 0.8):
    chars = [char_to_idx.get(c) for c in seed_text]
    chars = [c for c in chars if c is not None]
    input_tensor = torch.tensor([chars], dtype = torch.long)

    generated = seed_text
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim = 0)
            next_idx = torch.multinomial(probs, 1).item()
            generated += idx_to_char[next_idx]
            input_tensor = torch.tensor([[next_idx]], dtype = torch.long)

    return generated

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Input sentence")
    parser.add_argument("seed_text", type = str, help = "Input text to be used for generation")
    args = parser.parse_args()
    model, char_to_idx, idx_to_char = load_model()
    print(predict(model, char_to_idx, idx_to_char, args.seed_text))