"""Reference cases for the WhitespaceSplit pre-tokenizer, from huggingface/tokenizers alone.

Builds two small Unigram tokenizers in memory (WhitespaceSplit alone, and Sequence[WhitespaceSplit,
Metaspace] as in sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) and prints, for each
input, the ids and Unicode-scalar offsets the reference produces, plus ``pre_tokenize_str`` splits.
Non-ASCII inputs are written as escapes so that no invisible character hides in the source.
``tests/Starlings.Tokenizers.Tests/WhitespaceSplitComponentTests.cs`` embeds the output. Run with the
environment of ``reference/tokenizer-json/requirements.txt`` (tokenizers 0.23.3):

    python -I reference/components/whitespace_split_cases.py
"""

from __future__ import annotations

import json

import tokenizers
from tokenizers import Tokenizer

PIECES = [
    "<unk>",
    "a",
    "b",
    "c",
    "ab",
    "\u00e9",
    "\u4e2d",
    "\u6587",
    "\u200b",
    "\u2581",
    "\u2581a",
    "\u2581b",
    "\u2581ab",
    "\u2581c",
]

INPUTS = [
    "",
    "a",
    "a b",
    "   ",
    "  ab\tc\n",
    "a\r\nb",
    "a\u000bb\u000cc",
    "a\u0085b",
    "a\u00a0b",
    "\u1680a",
    "a\u2000\u200ab",
    "a\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009b",
    "a\u2028b\u2029c",
    "a\u202fb\u205fc",
    "a\u3000b",
    "a\u200bb",
    "a\u180eb",
    "a\ufeffb",
    "\u00e9 \u4e2d\u6587",
    "\U0001f600 a",
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
                {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": True},
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
