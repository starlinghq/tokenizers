"""Reference cases for the o200k pre-tokenizer regex and the Phi-4-mini tokenizer, from huggingface/tokenizers.

The o200k split regex (tiktoken's ``o200k_base`` pattern, as microsoft/Phi-4-mini-instruct declares it in
``tokenizer.json``) is matched in .NET by a hand-written scanner (``O200kSplitPattern.cs``). This runner
records what the reference produces, so the scanner is checked against the reference rather than against
a reading of the regex. Requirements: ``../tokenizer-json/requirements.txt`` (tokenizers 0.23.3, the
version the ``phi4-bpe-chat`` profile names). No model weights, no transformers.

Outputs (``--out``, normally ``tests/Starlings.Tokenizers.Tests/TestData/o200k-bpe/``):

- ``unicode-classes.json``: the code points the reference regex engine (Oniguruma) matches with
  ``\\p{Lu}``, ``\\p{Ll}``, ``\\p{Lt}``, ``\\p{Lm}``, ``\\p{Lo}`` (and ``\\p{L}``, ``\\p{N}``, ``\\p{M}``,
  ``\\s`` again, to show they equal the earlier 0.23.2 probe), probed over every Unicode scalar value.
- ``split.jsonl.gz``: ``Split(Regex(o200k), "removed", invert=True).pre_tokenize_str`` (the Phi-4-mini
  form) on every string of one to three characters drawn from one representative per character class,
  and on seeded random strings. Each row is ``{"t": text, "s": [[start, end], ...]}`` in Unicode scalar
  values. The runner also checks that ``Split(Regex(o200k), "isolated", invert=False)`` gives the same
  pieces for every row (it fails otherwise), so one record serves both forms.
- ``phi4-differential.jsonl.gz`` (with ``--asset``): seeded random strings mixing the Phi-4-mini added
  tokens (special and not, all ``rstrip`` true except two), fragments of them, whitespace runs and text,
  encoded by ``Tokenizer.from_file(asset)``: ids, tokens, scalar offsets, decode with and without
  special tokens.
- ``environment.json``: runner, its repository commit, options, interpreter, packages, asset digest.

Usage:
    python -I run_o200k_reference.py --out ../../tests/Starlings.Tokenizers.Tests/TestData/o200k-bpe \\
        --asset ../../.models/microsoft/Phi-4-mini-instruct/<commit>/tokenizer.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import itertools
import json
import random
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reference/fixtures.py
import fixtures  # noqa: E402

PACKAGES = ["tokenizers"]
SEED = 20261102
RANDOM_SPLIT_STRINGS = 4000
RANDOM_PHI4_STRINGS = 1500

O200K = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# One representative per class the regex distinguishes, plus the characters its literals name.
REPRESENTATIVES = [
    "A",  # Lu
    "a",  # Ll
    "ǅ",  # Lt (Dz with caron)
    "ʰ",  # Lm (modifier small h)
    "日",  # Lo (CJK)
    "́",  # Mn (combining acute)
    "ः",  # Mc (Devanagari visarga)
    "1",  # Nd
    "½",  # No
    " ",
    "\t",
    "\n",
    "\r",
    "　",  # \s, not a blank
    "/",
    "'",
    "s",
    "!",
    "\U0001f600",  # So, astral
    "\U0001d400",  # Lu, astral
]

WHITE_SPACE = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)]
WHITE_SPACE += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]

PHI4_ADDED = [
    "<|endoftext|>",
    "<|endofprompt|>",
    "<|assistant|>",
    "<|end|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<|/tool|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_response|>",
    "<|tag|>",
]


def _scalars(lo: int, hi: int) -> list[int]:
    return [c for c in range(lo, hi + 1) if not 0xD800 <= c <= 0xDFFF]


def probe_unicode_classes() -> dict:
    scalars = _scalars(0, 0x10FFFF)
    text = "".join(chr(c) for c in scalars)

    def ranges(pattern: str) -> list[list[int]]:
        removed = Split(Regex(pattern), "removed").pre_tokenize_str(text)
        outside = bytearray(len(scalars))
        for _, (s, e) in removed:
            outside[s:e] = b"\x01" * (e - s)
        out: list[list[int]] = []
        for i, c in enumerate(scalars):
            if outside[i]:
                continue
            if out and out[-1][1] == c - 1:
                out[-1][1] = c
            else:
                out.append([c, c])
        return out

    names = ["Lu", "Ll", "Lt", "Lm", "Lo", "L", "N", "M"]
    classes = {name: ranges(rf"\p{{{name}}}") for name in names}
    classes["s"] = ranges(r"\s")
    return {
        "engine": f"tokenizers {fixtures.environment(PACKAGES)['packages']['tokenizers']} Split(Regex) (Oniguruma)",
        "classes": classes,
    }


def random_text(rng: random.Random, pieces: list[str], max_len: int) -> str:
    out = []
    for _ in range(rng.randint(1, max_len)):
        r = rng.random()
        if r < 0.55:
            out.append(rng.choice(pieces))
        elif r < 0.7:
            out.append(chr(rng.choice(WHITE_SPACE)))
        elif r < 0.75:
            c = rng.randint(0, 0x10FFFF)
            out.append(chr(c) if not 0xD800 <= c <= 0xDFFF else "?")
        else:
            out.append(rng.choice(TEXT_PIECES))
    return "".join(out)


TEXT_PIECES = [
    "Hello",
    "HELLO",
    "hello",
    "HeLLo",
    "camelCase",
    "XMLHttp",
    "it's",
    "IT'S",
    "we'll",
    "they'RE",
    "I'm",
    "she'd",
    "you've",
    "don't",
    "heſ",
    "'s",
    "'ll",
    "12345",
    "3.14",
    "٣٤٥٦",
    "日本語",
    "中文",
    "नमस्ते",
    "مرحبا",
    "École",
    "é",
    "ʰa",
    "ǅungla",
    "\U0001f468‍\U0001f469‍\U0001f467",
    "\U0001d400\U0001d41a",
    "a/b",
    "//",
    "\n/",
    "!\n/",
    "...",
    "--",
    "<|",
    "|>",
    "<|end",
    "tool|>",
    "def f(x):\n    return x\n",
    "\r\n",
]


def split_rows(rng: random.Random) -> list[dict]:
    removed_inverted = Split(Regex(O200K), "removed", invert=True)
    isolated = Split(Regex(O200K), "isolated", invert=False)
    texts = []
    for n in (1, 2, 3):
        texts.extend("".join(p) for p in itertools.product(REPRESENTATIVES, repeat=n))
    pieces = REPRESENTATIVES + ["S", "t", "T", "r", "e", "v", "m", "l", "L", "d", "ſ"]
    texts.extend(random_text(rng, pieces, 24) for _ in range(RANDOM_SPLIT_STRINGS))
    rows = []
    for text in texts:
        a = [list(o) for _, o in removed_inverted.pre_tokenize_str(text)]
        b = [list(o) for _, o in isolated.pre_tokenize_str(text)]
        if a != b:
            raise SystemExit(f"Removed+invert and Isolated differ on {text!r}: {a} vs {b}")
        rows.append({"t": text, "s": a})
    return rows


def phi4_rows(tok: Tokenizer, rng: random.Random) -> list[dict]:
    pieces = PHI4_ADDED + ["  ", "\n", "\n\n", " \n ", "\t", "　", " "]
    rows = []
    for _ in range(RANDOM_PHI4_STRINGS):
        text = random_text(rng, pieces, 16)
        enc = tok.encode(text)
        rows.append(
            {
                "t": text,
                "ids": enc.ids,
                "tokens": enc.tokens,
                "o": [v for o in enc.offsets for v in o],
                "skip": tok.decode(enc.ids, skip_special_tokens=True),
                "keep": tok.decode(enc.ids, skip_special_tokens=False),
            }
        )
    return rows


def write_gz_jsonl(path: Path, rows: list[dict]) -> None:
    buffer = io.BytesIO()
    # mtime 0 and no file name keep the gzip bytes reproducible.
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path, help="normally tests/.../TestData/o200k-bpe")
    parser.add_argument("--asset", type=Path, help="the pinned microsoft/Phi-4-mini-instruct tokenizer.json")
    args = parser.parse_args()

    fixtures.write_json(args.out / "unicode-classes.json", probe_unicode_classes())
    split = split_rows(random.Random(SEED))
    write_gz_jsonl(args.out / "split.jsonl.gz", split)
    record = {
        "runner": "reference/o200k-bpe/run_o200k_reference.py",
        "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
        "reference": "tokenizers.pre_tokenizers.Split(tokenizers.Regex(o200k), 'removed', invert=True)",
        "regex": O200K,
        "seed": SEED,
        "split_rows": len(split),
        "split_random_strings": RANDOM_SPLIT_STRINGS,
        "representatives": REPRESENTATIVES,
    }
    if args.asset is not None:
        tok = Tokenizer.from_file(str(args.asset))
        phi = phi4_rows(tok, random.Random(SEED + 1))
        write_gz_jsonl(args.out / "phi4-differential.jsonl.gz", phi)
        record["asset"] = fixtures.file_record(args.asset)
        record["phi4_rows"] = len(phi)
    record["environment"] = fixtures.environment(PACKAGES)
    fixtures.write_json(args.out / "environment.json", record)
    print({k: v for k, v in record.items() if k in ("split_rows", "phi4_rows")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
