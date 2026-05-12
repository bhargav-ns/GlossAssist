# LSTM Text Generation

A character- and word-level LSTM language model that generates text from a seed prompt. Supports three tokenization strategies: character, whitespace, and BPE.

## Files

| File | Description |
|---|---|
| `model.py` | LSTM model definition |
| `utils.py` | Tokenizer build/save/load helpers |
| `train.py` | Training script |
| `generate.py` | Text generation script |

## Setup

```bash
pip install torch tokenizers tqdm
```

## Training

```bash
python train.py <data_path> --tokenization_type <type>
```

**Arguments**

| Argument | Options | Default | Description |
|---|---|---|---|
| `data_path` | — | required | Path to a plain text file |
| `--tokenization_type` | `char`, `whitespace`, `bpe` | `char` | Tokenization strategy |

**Examples**

```bash
# Character-level (good for stylistic/creative text)
python train.py data.txt --tokenization_type char

# Whitespace (word-level, no training required)
python train.py data.txt --tokenization_type whitespace

# BPE (subword, best for general text)
python train.py data.txt --tokenization_type bpe
```

Training saves two files:
- `model.pt` — model weights and vocabulary
- `tokenizer.json` — BPE tokenizer (only for `bpe` mode)

## Generation

```bash
python generate.py "<seed text>" --length <n> --temperature <t>
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `seed_text` | required | Text to seed the generation |
| `--length` | `200` | Number of tokens to generate |
| `--temperature` | `0.8` | Sampling temperature — lower is more conservative, higher is more creative |

**Examples**

```bash
python generate.py "Once upon a time"
python generate.py "The quick brown" --length 500 --temperature 0.6
```

## Tokenization strategies

**`char`** — splits text into individual characters. Best for learning fine-grained style and works well on small datasets. Generated output is joined without spaces.

**`whitespace`** — splits on whitespace. Simple word-level tokenization with no training step. Vocabulary can get large on diverse text.

**`bpe`** — byte-pair encoding via the `tokenizers` library. Learns a subword vocabulary of 2000 tokens from the training data. Best general-purpose choice for larger datasets.