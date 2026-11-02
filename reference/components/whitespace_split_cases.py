"""Reference cases for the WhitespaceSplit pre-tokenizer, from huggingface/tokenizers alone.

Builds two small Unigram tokenizers in memory (WhitespaceSplit alone, and Sequence[WhitespaceSplit,
Metaspace] as in sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) and prints, for each
input, the ids and Unicode-scalar offsets the reference produces, plus ``pre_tokenize_str`` splits.
``tests/Starlings.Tokenizers.Tests/WhitespaceSplitComponentTests.cs`` embeds the output. Run with the
environment of ``reference/tokenizer-json/requirements.txt`` (tokenizers 0.23.3):

    python -I reference/components/whitespace_split_cases.py
"""

from __future__ import annotations

import json

import tokenizers
from tokenizers import Tokenizer

PIECES = ["<unk>", "a", "b", "c", "ab", "é", "中", "文", "​", "▁", "▁a", "▁b", "▁ab", "▁c"]

INPUTS = [
    "",
    "a",
    "a b",
    "   ",
    "  ab\tc\n",
    "a\r\nb",
    "a\u000bb\u000cc",
    "a\u0085b",
    "a b",
    " a",
    "a  b",
    "a b c",
    "a b c",
    "a　b",
    "a​b",
    "a᠎b",
    "a﻿b",
    "é 中文",
    "😀 a",
]


def build(pre_tokenizer: dict) -> Tokenizer:
    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "decoder": None,
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [[p, -1.0 - i * 0.1] for i, p in enumerate(PIECES)]},
    }
    return Tokenizer.from_str(json.dumps(spec))


def main() -> None:
    variants = {
        "whitespace_split": {"type": "WhitespaceSplit"},
        "whitespace_split_metaspace": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
            ],
        },
    }
    out = {"tokenizers": tokenizers.__version__, "pieces": PIECES, "variants": {}}
    for name, pre in variants.items():
        tok = build(pre)
        cases = []
        for text in INPUTS:
            enc = tok.encode(text)
            cases.append(
                {
                    "text": text,
                    "splits": tok.pre_tokenizer.pre_tokenize_str(text),
                    "ids": enc.ids,
                    "offsets": [list(o) for o in enc.offsets],
                }
            )
        out["variants"][name] = {"pre_tokenizer": pre, "cases": cases}
    print(json.dumps(out, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
