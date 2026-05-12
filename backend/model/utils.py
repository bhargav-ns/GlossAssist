import os
import tempfile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def build_tokenizer(tokenization_type, text):
    if tokenization_type == "char":
        return list, None

    if tokenization_type == "whitespace":
        return str.split, None

    if tokenization_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=2000, special_tokens=["[UNK]"])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp = f.name
        tokenizer.train([tmp], trainer)
        os.unlink(tmp)
        return lambda t: tokenizer.encode(t).tokens, tokenizer

    raise ValueError(f"Unknown tokenization type: {tokenization_type!r}. Choose: char, whitespace, bpe")


def save_tokenizer(tokenizer, path):
    if tokenizer is not None:
        tokenizer.save(path)


def load_tokenizer(tokenization_type, tokenizer_path=None):
    if tokenization_type == "char":
        return list

    if tokenization_type == "whitespace":
        return str.split

    if tokenization_type == "bpe":
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return lambda t: tokenizer.encode(t).tokens

    raise ValueError(f"Unknown tokenization type: {tokenization_type!r}")