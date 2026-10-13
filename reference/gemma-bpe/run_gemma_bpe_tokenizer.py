"""Differential and component references for the Gemma BPE tokenizer profile ``gemma-bpe-embeddinggemma2``.

The profile asset is the ``google/embeddinggemma-2`` repository ``tokenizer.json`` at ``914f7f89`` (SHA-256
``4d777ef5...``): SentencePiece-style BPE with byte fallback, ``Replace(" " -> "▁")``, ``Split(String " ",
MergedWithPrevious)``, ``TemplateProcessing`` and the ``Sequence[Replace, ByteFallback, Fuse]`` decoder. Its
conformance fixtures are the accepted ``conformance/gemma-bpe-embeddinggemma2/tokenizer*.json*``
(produced by ``model-pipeline/reference/embeddinggemma2/run_embeddinggemma2.py``). This runner adds two
non-protected references for the owned tokenizer, both computed with ``tokenizers`` 0.23.3 (the version the
fixtures were generated with; ``requirements.txt``):

- ``--differential-out <file.jsonl.gz>``: a seeded randomized differential dump. Generated strings cover
  scripts the vocabulary lacks (byte fallback), astral planes, private use, controls, emoji sequences,
  whitespace and newline runs, literal ``▁`` and byte-token look-alikes, and every added-token literal and
  fragments; the reference ids, Unicode-scalar offsets and decode are recorded on the asset and on option
  variants that reach interpreter branches the asset leaves unused (unknown token with and without fusion,
  ``ignore_merges``, every ``Split`` behavior and ``invert`` on a ``String`` pattern, multi-character
  ``Replace`` patterns and contents); plus random id sequences for decode (byte tokens, invalid UTF-8,
  added tokens, ids outside the vocabulary). Variants are text edits of the asset, listed in the header
  row, so the .NET test applies exactly the same edits.
- ``--components-out <file.json>``: a small synthetic tokenizer (byte fallback with some byte tokens
  missing, unknown token) and variants, with reference encodings and decodes, for unit tests that do not
  need the asset.

Usage:
    python run_gemma_bpe_tokenizer.py --asset <google/embeddinggemma-2@914f7f89/tokenizer.json> \\
        --differential-out <file> [--differential-cases 2000 --variant-cases 300 --decode-cases 1000 --seed 20261011] \\
        [--components-out <file>]
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

PROFILE = "gemma-bpe-embeddinggemma2"
ASSET = {
    "repo": "google/embeddinggemma-2",
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "file": "tokenizer.json",
    "sha256": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
}
TOKENIZERS_VERSION = "0.23.3"


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------------------------
# Asset variants as text edits (the .NET test applies the same edits to the same bytes)


def apply_edits(text: str, edits: list[dict]) -> str:
    for edit in edits:
        if edit["op"] == "replace_once":
            if text.count(edit["from"]) != 1:
                raise ValueError(f"expected exactly one {edit['from']!r}")
            text = text.replace(edit["from"], edit["to"])
        elif edit["op"] == "section":
            # Replaces the value of a top-level key of the pretty-printed document: from after
            # '\n  "<key>": ' to the next top-level key ('",\n  "' at indent 2).
            marker = f'\n  "{edit["key"]}": '
            if text.count(marker) != 1:
                raise ValueError(f"expected exactly one top-level {edit['key']!r}")
            start = text.index(marker) + len(marker)
            end = text.index(',\n  "', start)
            text = text[:start] + json.dumps(edit["value"], ensure_ascii=False) + text[end:]
        else:
            raise ValueError(edit["op"])
    return text


def split(pattern: str, behavior: str, invert: bool = False) -> dict:
    return {"type": "Split", "pattern": {"String": pattern}, "behavior": behavior, "invert": invert}


def replace(pattern: str, content: str) -> dict:
    return {"type": "Replace", "pattern": {"String": pattern}, "content": content}


VARIANTS: dict[str, list[dict]] = {
    "asset": [],
    "ignore-merges": [{"op": "replace_once", "from": '"ignore_merges": false,', "to": '"ignore_merges": true,'}],
    "unk-fused": [{"op": "replace_once", "from": '"byte_fallback": true,', "to": '"byte_fallback": false,'}],
    "unk-unfused": [
        {"op": "replace_once", "from": '"byte_fallback": true,', "to": '"byte_fallback": false,'},
        {"op": "replace_once", "from": '"fuse_unk": true,', "to": '"fuse_unk": false,'},
    ],
    "split-removed": [{"op": "section", "key": "pre_tokenizer", "value": split("▁", "Removed")}],
    "split-isolated": [{"op": "section", "key": "pre_tokenizer", "value": split("▁", "Isolated")}],
    "split-merged-previous": [{"op": "section", "key": "pre_tokenizer", "value": split("▁", "MergedWithPrevious")}],
    "split-merged-next": [{"op": "section", "key": "pre_tokenizer", "value": split("▁", "MergedWithNext")}],
    "split-contiguous": [{"op": "section", "key": "pre_tokenizer", "value": split("▁▁", "Contiguous")}],
    "split-invert": [{"op": "section", "key": "pre_tokenizer", "value": split("▁", "Removed", invert=True)}],
    "replace-multi": [
        {
            "op": "section",
            "key": "normalizer",
            "value": {
                "type": "Sequence",
                "normalizers": [replace(" ", "▁"), replace("▁▁", "→"), replace("ab", ""), replace("é", "é")],
            },
        }
    ],
}


# ---------------------------------------------------------------------------------------------
# Randomized differential inputs


def _scalars(lo: int, hi: int) -> list[int]:
    return [c for c in range(lo, hi + 1) if not 0xD800 <= c <= 0xDFFF]


WHITE_SPACE = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)]
WHITE_SPACE += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
ADDED_LITERALS = [
    "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
    "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>", "<|turn>",
    "<turn|>", "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>",
]  # fmt: skip
LOOKALIKES = ["▁", "▁▁", "<0x41>", "<0xE2>", "<0x0A>", "<0xZZ>", "<0x", "0x41>", "<0x+A>", "<0xe9>", "[multimodal]"]
LOOKALIKES += ["<unused0>", "<unused99>", "<start_of_turn>", "<s>", "</s>", "<|", "|>", "<<", ">>"]
RARE = [*_scalars(0x10000, 0x1007F), *_scalars(0x10300, 0x1032F), *_scalars(0x12000, 0x123FF)]
RARE += [*_scalars(0x13000, 0x1342F), *_scalars(0x17000, 0x187F7)]
RARE += [*_scalars(0x1680, 0x169F), *_scalars(0x1B000, 0x1B0FF)]
RARE += [*_scalars(0x10A00, 0x10A5F), *_scalars(0x11000, 0x1107F), *_scalars(0x1E800, 0x1E8DF)]
PRIVATE = [*_scalars(0xE000, 0xF8FF), *_scalars(0xF0000, 0xFFFFD), *_scalars(0x100000, 0x10FFFD)]
PRIVATE += [0xFFFE, 0xFFFF, 0x1FFFE, 0x10FFFF, *_scalars(0xE0000, 0xE007F), 0xFDD0, 0xFDEF]
ASTRAL_LETTERS = [*_scalars(0x1D400, 0x1D7CB), *_scalars(0x10330, 0x1034A), *_scalars(0x1E900, 0x1E943)]
ASTRAL_LETTERS += _scalars(0x20000, 0x2A6DF)
EMOJI = [*_scalars(0x1F300, 0x1FAFF), 0x200D, 0xFE0F, *range(0x1F3FB, 0x1F400), 0x20E3, *range(0x1F1E6, 0x1F200)]
# (name, weight, code point list or None, string list or None)
POOLS: list[tuple[str, int, list[int] | None, list[str] | None]] = [
    ("ascii-letters", 14, [*range(0x41, 0x5B), *range(0x61, 0x7B)], None),
    ("ascii-digits", 5, list(range(0x30, 0x3A)), None),
    ("ascii-punct", 6, [c for c in range(0x21, 0x7F) if not chr(c).isalnum()], None),
    ("space", 10, [0x20], None),
    ("white-space", 5, WHITE_SPACE, None),
    ("newline-tab", 4, [0x0A, 0x0D, 0x09], None),
    ("controls", 3, [*range(0x00, 0x20), *range(0x7F, 0xA0)], None),
    ("latin", 4, _scalars(0xA0, 0x24F), None),
    ("combining", 4, _scalars(0x300, 0x36F), None),
    ("indic-thai-tamil", 3, [*_scalars(0x900, 0x97F), *_scalars(0xE00, 0xE7F), *_scalars(0xB80, 0xBFF)], None),
    ("arabic-hebrew", 3, [*_scalars(0x590, 0x5FF), *_scalars(0x600, 0x6FF)], None),
    ("hangul", 2, [*_scalars(0x1100, 0x11FF), *_scalars(0xAC00, 0xD7A3)], None),
    ("cjk-kana", 4, [*_scalars(0x3040, 0x30FF), *_scalars(0x4E00, 0x9FFF)], None),
    ("fullwidth", 1, _scalars(0xFF00, 0xFFEF), None),
    ("zero-width", 2, [*range(0x200B, 0x2010), *range(0x2060, 0x2065), 0xFEFF, 0x034F, 0x180E], None),
    ("emoji", 5, EMOJI, None),
    ("rare-scripts", 6, RARE, None),
    ("private-noncharacters", 3, PRIVATE, None),
    ("astral-letters", 3, ASTRAL_LETTERS, None),
    ("any-bmp", 3, None, None),
    ("any-scalar", 4, None, None),
    ("whitespace-run", 4, None, None),
    ("added-token", 4, None, ADDED_LITERALS),
    ("added-fragment", 2, None, [t[: max(1, len(t) // 2)] for t in ADDED_LITERALS] + [t[1:] for t in ADDED_LITERALS]),
    ("lookalike", 3, None, LOOKALIKES),
]  # fmt: skip


def random_text(rng: random.Random) -> str:
    names = list(POOLS)
    weights = [p[1] for p in POOLS]
    out = []
    segments = rng.randint(20, 60) if rng.random() < 0.03 else rng.randint(1, 10)
    for _ in range(segments):
        name, _w, points, strings = rng.choices(names, weights)[0]
        for _ in range(rng.randint(1, 6)):
            if strings is not None:
                out.append(rng.choice(strings))
            elif points is not None:
                out.append(chr(rng.choice(points)))
            elif name == "whitespace-run":
                out.append(rng.choice([" ", "\n", "\t", "  ", "\r\n", "　"]) * rng.randint(2, 40))
            elif name == "any-bmp":
                c = rng.randrange(0, 0xF800)
                out.append(chr(c if c < 0xD800 else c + 0x800))
            else:
                c = rng.randrange(0, 0x10F800)
                out.append(chr(c if c < 0xD800 else c + 0x800))
    return "".join(out)


def compact_encoding(tok: Tokenizer, text: str, pair: str | None) -> dict:
    enc = tok.encode(text, pair) if pair is not None else tok.encode(text)
    row = {"ids": enc.ids, "o": [x for se in enc.offsets for x in se]}
    if pair is not None:
        row["seq"] = enc.sequence_ids
    row["dk"] = tok.decode(enc.ids, skip_special_tokens=False)
    return row


def differential(asset_text: str, cases: int, decode_cases: int, variant_cases: int, seed: int) -> list[dict]:
    tokenizers = {name: Tokenizer.from_str(apply_edits(asset_text, edits)) for name, edits in VARIANTS.items()}
    rng = random.Random(seed)
    rows = []
    for i in range(cases):
        text = random_text(rng)
        pair = random_text(rng) if i % 25 == 0 else None
        for name, tok in tokenizers.items():
            if name != "asset" and i >= variant_cases:
                continue
            row = {"v": name, "t": text}
            if pair is not None:
                row["p"] = pair
            row.update(compact_encoding(tok, text, pair))
            rows.append(row)
    tok = tokenizers["asset"]
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    byte_ids = [tok.token_to_id(f"<0x{b:02X}>") for b in range(256)]
    added_ids = sorted(tok.get_added_tokens_decoder())
    special = [tok.token_to_id(t) for t in ["▁", "▁▁", "<unk>", "<bos>", "<eos>"]]
    for _ in range(decode_cases):
        ids = []
        for _ in range(rng.randint(1, 16)):
            r = rng.random()
            if r < 0.45:
                ids.append(rng.choice(byte_ids))
            elif r < 0.55:
                # A valid multi-byte sequence as byte tokens, sometimes cut short.
                encoded = chr(rng.choice([*RARE, *EMOJI, *ASTRAL_LETTERS])).encode("utf-8")
                cut = len(encoded) if rng.random() < 0.6 else rng.randint(1, len(encoded))
                ids.extend(byte_ids[b] for b in encoded[:cut])
            elif r < 0.80:
                ids.append(rng.randrange(0, vocab_size))
            elif r < 0.90:
                ids.append(rng.choice(added_ids))
            elif r < 0.97:
                ids.append(rng.choice(special))
            else:
                ids.append(rng.randrange(vocab_size, vocab_size + 1000))
        rows.append(
            {
                "v": "decode",
                "dec": ids,
                "dk": tok.decode(ids, skip_special_tokens=False),
                "ds": tok.decode(ids, skip_special_tokens=True),
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Synthetic component references


def added(token_id: int, content: str, special: bool) -> dict:
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": special,
    }


def components() -> dict:
    vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "▁": 4, "▁a": 5, "<0xC3>": 6, "<0xA9>": 7, "<0xF0>": 8}
    vocab |= {"<0x9F>": 9, "<0x98>": 10, "<0x80>": 11, "<0xE2>": 12, "<0x41>": 13, "<bos>": 14, "<eos>": 15}
    base = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [added(14, "<bos>", True), added(15, "<eos>", True)],
        "normalizer": replace(" ", "▁"),
        "pre_tokenizer": split(" ", "MergedWithPrevious"),
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [replace("▁", " "), {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [["a", "b"], ["▁", "a"]],
        },
    }

    def variant(change) -> dict:
        doc = copy.deepcopy(base)
        change(doc)
        return doc

    docs = {
        "base": base,
        "unk-unfused": variant(lambda d: d["model"].update(fuse_unk=False)),
        "no-byte-fallback": variant(lambda d: d["model"].update(byte_fallback=False)),
        "no-unk": variant(lambda d: d["model"].update(unk_token=None)),
        "split-invert": variant(lambda d: d.update(pre_tokenizer=split("a", "Removed", invert=True))),
    }
    for behavior in ["Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous"]:
        docs["split-" + behavior] = variant(lambda d, b=behavior: d.update(pre_tokenizer=split("▁", b)))
    odd = variant(lambda d: None)
    for token_id, content in [(16, "<0x+A>"), (17, "<0xe9>"), (18, "<0xc3>"), (19, "<0x-1>"), (20, "<0xé>")]:
        odd["added_tokens"].append(added(token_id, content, False))
    odd["added_tokens"].append(added(21, "<0xA9 >", False))
    docs["byte-token-lookalikes"] = odd

    texts = ["ab é😀€€a", "€é€ b", " a  ab ", "xéy", "a<bos>é", "", "€", "😀😀 €"]
    decodes = [[6], [6, 7], [8, 9, 10], [8, 9, 10, 11], [4, 6, 1, 7, 4], [13, 6, 7, 13], [0, 6, 14, 7], [99, 6, 7]]
    decodes += [[16], [18, 7], [6, 17], [19], [20], [6, 21], [6, 16], [8, 9, 10, 11, 8, 9]]
    cases = []
    for name, doc in docs.items():
        tok = Tokenizer.from_str(json.dumps(doc))
        for text in texts:
            enc = tok.encode(text)
            cases.append(
                {
                    "tokenizer": name,
                    "text": text,
                    "ids": enc.ids,
                    "tokens": enc.tokens,
                    "offsets": [list(o) for o in enc.offsets],
                    "decode": tok.decode(enc.ids, skip_special_tokens=False),
                }
            )
        for ids in decodes:
            cases.append(
                {
                    "tokenizer": name,
                    "decode_ids": ids,
                    "keep": tok.decode(ids, skip_special_tokens=False),
                    "skip": tok.decode(ids, skip_special_tokens=True),
                }
            )
    return {
        "reference": f"tokenizers {version('tokenizers')}",
        "generator": "reference/gemma-bpe/run_gemma_bpe_tokenizer.py --components-out",
        "offsets": "Unicode scalar values",
        "tokenizers": docs,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, help="the profile's pinned tokenizer.json")
    parser.add_argument("--differential-out", type=Path)
    parser.add_argument("--differential-cases", type=int, default=2000)
    parser.add_argument("--decode-cases", type=int, default=1000)
    parser.add_argument("--variant-cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--components-out", type=Path)
    args = parser.parse_args()

    if version("tokenizers") != TOKENIZERS_VERSION:
        print(f"environment mismatch: tokenizers {version('tokenizers')}, expected {TOKENIZERS_VERSION}")
        return 2

    if args.components_out:
        args.components_out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(components(), ensure_ascii=False, indent=1) + "\n"
        args.components_out.write_text(text, encoding="utf-8")
        print(f"components -> {args.components_out}")

    if args.differential_out:
        if args.asset is None:
            parser.error("--differential-out needs --asset")
        asset_bytes = args.asset.read_bytes()
        asset_sha = hashlib.sha256(asset_bytes).hexdigest()
        if asset_sha != ASSET["sha256"]:
            print(f"{args.asset}: sha256 {asset_sha}, expected {ASSET['sha256']}")
            return 1
        asset_text = asset_bytes.decode("utf-8")
        rows = differential(asset_text, args.differential_cases, args.decode_cases, args.variant_cases, args.seed)
        header = {
            "v": "header",
            "profile": PROFILE,
            "asset": ASSET,
            "asset_sha256": asset_sha,
            "seed": args.seed,
            "cases": args.differential_cases,
            "decode_cases": args.decode_cases,
            "variant_cases": args.variant_cases,
            "variants": VARIANTS,
            "generator": "reference/gemma-bpe/run_gemma_bpe_tokenizer.py random_text",
            "reference": f"tokenizers {version('tokenizers')}",
            "environment": {"python": platform.python_version(), "machine": platform.machine()},
            "offsets": "o: flattened [start, end] pairs in Unicode scalar values",
        }
        args.differential_out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(args.differential_out, "wb", mtime=0) as raw:
            for row in [header, *rows]:
                raw.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        size = args.differential_out.stat().st_size
        print(f"differential: {len(rows)} rows -> {args.differential_out} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
