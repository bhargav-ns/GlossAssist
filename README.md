# torchscribe

Train a character- or word-level language model on any text file, then chat with it through a web UI. Built with PyTorch (LSTM / RNN), Flask, and React.

---

## Features

- **Custom training** — upload any `.txt` file and train a model on it in the browser
- **Hyperparameter control** — tune epochs, learning rate, batch size, embedding/hidden dims, layers, sequence length, and tokenization
- **Live progress** — real-time epoch and loss tracking during training
- **Model types** — choose between LSTM and RNN architectures
- **Tokenization options** — character-level, whitespace, or BPE
- **Chat interface** — talk to your trained model directly in the browser

---

## Project Structure

```
torchscribe/
├── backend/
│   ├── app.py               # Flask API server
│   ├── model/
│   │   ├── model.py         # LSTM and RNN definitions
│   │   ├── train.py         # Training loop
│   │   ├── predict.py       # Inference and model loading
│   │   └── utils.py         # Tokenizer build/load/save
│   └── model/data/          # Uploaded training files (auto-created)
└── frontend/
    ├── src/
    │   ├── App.js
    │   └── App.css
    └── .env
```

---

## Quickstart

### 1. Backend

```bash
cd backend
pip install torch flask flask-cors tqdm
python app.py
```

The server starts on `http://localhost:5001`.

### 2. Frontend

```bash
cd frontend
echo "REACT_APP_API_URL=http://localhost:5001" > .env
npm install
npm start
```

The app opens at `http://localhost:3000`.

---

## Usage

1. **Upload** a `.txt` file using the file picker
2. **Configure** hyperparameters (or leave defaults)
3. **Start training** and watch the progress bar
4. Once training completes, **type a message** in the chat box and the model will continue your text

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| Model type | LSTM | Architecture: LSTM or RNN |
| Epochs | 20 | Number of training passes |
| Learning rate | 0.01 | Adam optimizer step size |
| Batch size | 64 | Sequences per gradient update |
| Embedding dim | 64 | Token embedding size |
| Hidden dim | 128 | Recurrent hidden state size |
| Layers | 2 | Number of stacked recurrent layers |
| Seq length | 100 | Context window length |
| Tokenization | char | `char`, `whitespace`, or `bpe` |

---

## Training from the CLI

```bash
cd backend
python -m model.train path/to/file.txt \
  --model_type LSTM \
  --epochs 30 \
  --lr 0.005 \
  --tokenization_type char
```

## Inference from the CLI

```bash
cd backend
python -m model.predict "Once upon a time" \
  --length 300 \
  --temperature 0.8 \
  --model_type LSTM \
  --path model.pt
```

---

