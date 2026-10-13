"""Reference fixtures for the Qwen byte-level BPE tokenizer profiles (P-024 follow-up).

Two declared profiles share one vocabulary and merge list and differ in the pre-tokenizer regex:

- ``kev`` (profile ``qwen2-bpe-kev``): what Kev-0.8B runs. Kev loads ``Qwen/Qwen3.5-0.8B-Base`` at
  ``dc7cdfe2`` with transformers 5.17.0, which builds ``Qwen2Tokenizer``; its live pipeline equals the
  Kev repository ``tokenizer.json`` (``06b95093...``): Qwen2/Qwen3 regex without ``\\p{M}``.
- ``chat`` (profile ``qwen35-bpe-chat``): the pilot LLM ``Qwen/Qwen3.5-0.8B`` at ``2fc06364`` loaded by
  transformers 5.18.0, which builds ``Qwen3_5Tokenizer`` from the class (not from the Hub
  ``tokenizer.json``); its live pipeline, serialized with ``to_str(pretty=True)``, is the profile asset
  (``a5cd9732...``): Qwen3.5 regex with ``\\p{M}``.

The runner loads the snapshot with the pinned transformers (``requirements-<profile>.txt``), checks that
the live ``backend_tokenizer`` serializes to exactly the profile asset, and then records the encodings of
``tokenizers.Tokenizer.from_file(asset)`` (tokenizers 0.23.2, the engine both transformers versions use).
Every case is also encoded by the transformers tokenizer (``add_special_tokens=False``, as Kev calls it)
and any id difference is reported.

Outputs (``--out``):

- ``tokenizer.jsonl``: per corpus case, ids, tokens, type ids, attention and special-token masks,
  offsets in three units, sequence ids for pairs, decode with and without special tokens (fixture
  format version 1, ``reference/README.md``).
- ``live-pipeline.json``: the transformers class, the serialized live pipeline digest, whether it equals
  the asset, and the transformers cross-check.
- ``environment.json``: interpreter, packages, CPU, asset and corpus digests.

Optional:

- ``--differential-out <file.jsonl.gz>``: a seeded randomized differential dump (generated strings across
  scripts, astral planes, whitespace, combining marks, contractions and added-token fragments), with the
  ids, scalar offsets and decode of the reference, on the asset and on option variants that exercise
  interpreter branches the asset leaves unused; plus random id sequences for decode.
- ``--unicode-classes <file.json>``: the code points the reference regex engine (Oniguruma in tokenizers
  0.23.2) puts in ``\\p{L}``, ``\\p{N}``, ``\\p{M}`` and ``\\s``, and the case-insensitive matches of the
  contraction alternatives, probed over every Unicode scalar value.
- ``--write-live <file>``: write the serialized live pipeline (how the chat asset is produced).

Usage:
    python run_qwen_bpe_tokenizer.py --profile kev --snapshot <Qwen3.5-0.8B-Base@dc7cdfe2> \\
        --asset <jaredpalmer/kev-0.8b@9a45d25e/tokenizer.json> \\
        --corpus ../corpus/tokenizer-corpus-v1.jsonl --corpus ../corpus/tokenizer-corpus-qwen-bpe-v1.jsonl \\
        --out <dir> [--differential-out <file> --differential-cases 1500 --seed 20261010]
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reference/fixtures.py
import fixtures  # noqa: E402

PROFILES = {
    "kev": {
        "profile": "qwen2-bpe-kev",
        "transformers": "5.17.0",
        "class": "Qwen2Tokenizer",
        "load": {"repo": "Qwen/Qwen3.5-0.8B-Base", "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"},
        "asset": {
            "repo": "jaredpalmer/kev-0.8b",
            "revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
            "file": "tokenizer.json",
            "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        },
    },
    "chat": {
        "profile": "qwen35-bpe-chat",
        "transformers": "5.18.0",
        "class": "Qwen3_5Tokenizer",
        "load": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
        "asset": {
            "repo": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "file": "transformers-5.18.0/tokenizer.json",
            "derivation": "backend_tokenizer.to_str(pretty=True) under transformers 5.18.0",
            "sha256": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
        },
    },
}
PACKAGES = ["transformers", "tokenizers", "huggingface-hub"]
SNAPSHOT_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def encoding_record(text: str, enc, pair: str | None) -> dict:
    record = {
        "ids": enc.ids,
        "tokens": enc.tokens,
        "type_ids": enc.type_ids,
        "attention_mask": enc.attention_mask,
        "special_tokens_mask": enc.special_tokens_mask,
    }
    if pair is None:
        record["offsets"] = fixtures.all_offset_units(text, enc.offsets)
    else:
        seq = enc.sequence_ids
        record["sequence_ids"] = seq
        per_unit = {unit: [] for unit in fixtures.OFFSET_UNITS}
        for (s, e), sid in zip(enc.offsets, seq, strict=True):
            source = pair if sid == 1 else text
            for unit in fixtures.OFFSET_UNITS:
                per_unit[unit].append(fixtures.convert_offsets(source, [(s, e)], unit)[0])
        record["offsets"] = per_unit
    return record


def encode(tok: Tokenizer, text: str, pair: str | None):
    return tok.encode(text, pair) if pair is not None else tok.encode(text)


def tokenizer_rows(tok: Tokenizer, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        enc = encode(tok, text, pair)
        row = {"id": case["id"], "category": case["category"], "text": text}
        if pair is not None:
            row["pair"] = pair
        row["encoding"] = encoding_record(text, enc, pair)
        row["decode"] = {
            "skip_special_tokens": tok.decode(enc.ids, skip_special_tokens=True),
            "keep_special_tokens": tok.decode(enc.ids, skip_special_tokens=False),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# Randomized differential inputs


def _scalars(lo: int, hi: int) -> list[int]:
    return [c for c in range(lo, hi + 1) if not 0xD800 <= c <= 0xDFFF]


WHITE_SPACE = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)]
WHITE_SPACE += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
ADDED_LITERALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|box_start|>", "<|vision_pad|>",
    "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_pad|>", "<|file_sep|>", "<tool_response>",
    "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<tts_pad>", "<tts_text_bos_single>",
    "<|audio_pad|>",
]  # fmt: skip
CONTRACTIONS = ["'s", "'S", "'t", "'T", "'re", "'RE", "'rE", "'ve", "'Ve", "'m", "'M", "'ll", "'LL", "'lL"]
CONTRACTIONS += ["'d", "'D", "'ſ", "'", "''", "'r", "'v", "'l", "’s"]

# (name, weight, code point list or None, string list or None)
DIGITS_OTHER = [*range(0x660, 0x66A), *range(0x966, 0x970), *range(0x2460, 0x249C), *range(0x1D7CE, 0x1D800)]
ASTRAL_LETTERS = [*_scalars(0x1D400, 0x1D7CB), *_scalars(0x10330, 0x1034A), *_scalars(0x1E900, 0x1E943)]
ASTRAL_LETTERS += _scalars(0x20000, 0x2A6DF)
ASTRAL_MARKS = [*_scalars(0x1D165, 0x1D169), *_scalars(0x11300, 0x11303), *_scalars(0xE0100, 0xE01EF)]
ASTRAL_MARKS += _scalars(0x1E944, 0x1E94A)
POOLS: list[tuple[str, int, list[int] | None, list[str] | None]] = [
    ("ascii-letters", 14, [*range(0x41, 0x5B), *range(0x61, 0x7B)], None),
    ("ascii-digits", 6, list(range(0x30, 0x3A)), None),
    ("ascii-punct", 6, [c for c in range(0x21, 0x7F) if not chr(c).isalnum()], None),
    ("space", 8, [0x20], None),
    ("white-space", 5, WHITE_SPACE, None),
    ("newline", 4, [0x0A, 0x0D], None),
    ("controls", 1, [*range(0x00, 0x20), *range(0x7F, 0xA0)], None),
    ("latin", 4, _scalars(0xA0, 0x24F), None),
    ("combining", 6, _scalars(0x300, 0x36F), None),
    ("nfc-bases", 3, [ord(c) for c in "aeiouyAEIOUYnNcCsSzZgGhHkKwW"], None),
    ("indic-thai-tamil", 4, [*_scalars(0x900, 0x97F), *_scalars(0xE00, 0xE7F), *_scalars(0xB80, 0xBFF)], None),
    ("arabic-hebrew", 3, [*_scalars(0x590, 0x5FF), *_scalars(0x600, 0x6FF)], None),
    ("hangul", 3, [*_scalars(0x1100, 0x11FF), *_scalars(0xAC00, 0xD7A3)], None),
    ("cjk-kana", 4, [*_scalars(0x3040, 0x30FF), *_scalars(0x4E00, 0x9FFF)], None),
    ("fullwidth", 1, _scalars(0xFF00, 0xFFEF), None),
    ("zero-width", 2, [*range(0x200B, 0x2010), *range(0x2060, 0x2065), 0xFEFF, 0x034F, 0x180E], None),
    ("digits-other", 2, DIGITS_OTHER, None),
    ("emoji", 4, [*_scalars(0x1F300, 0x1FAFF), 0x200D, 0xFE0F, *range(0x1F3FB, 0x1F400), 0x20E3], None),
    ("astral-letters", 3, ASTRAL_LETTERS, None),
    ("astral-marks", 2, ASTRAL_MARKS, None),
    ("any-bmp", 3, None, None),
    ("any-scalar", 3, None, None),
    ("contraction", 5, None, CONTRACTIONS),
    ("added-token", 3, None, ADDED_LITERALS),
    ("added-fragment", 2, None, [t[: max(1, len(t) // 2)] for t in ADDED_LITERALS] + [t[1:] for t in ADDED_LITERALS]),
]  # fmt: skip


def random_text(rng: random.Random) -> str:
    names = list(POOLS)
    weights = [p[1] for p in POOLS]
    out = []
    for _ in range(rng.randint(1, 10)):
        _name, _w, points, strings = rng.choices(names, weights)[0]
        for _ in range(rng.randint(1, 6)):
            if strings is not None:
                out.append(rng.choice(strings))
            elif points is not None:
                out.append(chr(rng.choice(points)))
            elif _name == "any-bmp":
                c = rng.randrange(0, 0xF800)
                out.append(chr(c if c < 0xD800 else c + 0x800))
            else:
                c = rng.randrange(0, 0x10F800)
                out.append(chr(c if c < 0xD800 else c + 0x800))
    return "".join(out)


def variants(asset_json: dict) -> dict[str, dict]:
    """Option variants of the asset that reach interpreter branches the asset does not use."""
    out = {"asset": asset_json}
    trim = copy.deepcopy(asset_json)
    trim["post_processor"]["trim_offsets"] = True
    out["post-trim-offsets"] = trim
    prefix = copy.deepcopy(asset_json)
    prefix["post_processor"]["trim_offsets"] = True
    prefix["post_processor"]["add_prefix_space"] = True
    prefix["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True
    out["prefix-space-trim"] = prefix
    ignore = copy.deepcopy(asset_json)
    ignore["model"]["ignore_merges"] = True
    out["ignore-merges"] = ignore
    return out


def compact_encoding(tok: Tokenizer, text: str, pair: str | None) -> dict:
    enc = encode(tok, text, pair)
    row = {"ids": enc.ids, "o": [x for se in enc.offsets for x in se]}
    if pair is not None:
        row["seq"] = enc.sequence_ids
    row["dk"] = tok.decode(enc.ids, skip_special_tokens=False)
    return row


def differential(asset: Path, cases: int, decode_cases: int, variant_cases: int, seed: int) -> list[dict]:
    asset_json = json.loads(asset.read_text(encoding="utf-8"))
    tokenizers = {name: Tokenizer.from_str(json.dumps(doc)) for name, doc in variants(asset_json).items()}
    rng = random.Random(seed)
    texts = [random_text(rng) for _ in range(cases)]
    rows = []
    for i, text in enumerate(texts):
        pair = random_text(rng) if i % 25 == 0 else None
        for name, tok in tokenizers.items():
            if name != "asset" and i >= variant_cases:
                continue
            row = {"v": name, "t": text}
            if pair is not None:
                row["p"] = pair
            row.update(compact_encoding(tok, text, pair))
            rows.append(row)
    vocab_size = tokenizers["asset"].get_vocab_size(with_added_tokens=True)
    for _ in range(decode_cases):
        ids = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.5:
                ids.append(rng.randrange(0, 256))
            elif r < 0.85:
                ids.append(rng.randrange(0, vocab_size))
            elif r < 0.97:
                ids.append(rng.randrange(248044, 248077))
            else:
                ids.append(rng.randrange(vocab_size, vocab_size + 1000))
        tok = tokenizers["asset"]
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
# Unicode class probe


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

    contraction = Split(Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"), "isolated")
    folds = {}
    probe_scalars = [c for c in scalars if c != 0x0A]
    for template in ["'{}", "'{}e", "'r{}", "'v{}", "'{}l", "'l{}"]:
        found = []
        for start in range(0, len(probe_scalars), 50000):
            part = probe_scalars[start : start + 50000]
            instances = [template.format(chr(c)) for c in part]
            whole = {piece for piece, _ in contraction.pre_tokenize_str("\n".join(instances))}
            found.extend(c for c, inst in zip(part, instances, strict=True) if inst in whole)
        folds[template] = found
    return {
        "engine": f"tokenizers {version('tokenizers')} Split(Regex) (Oniguruma)",
        "classes": {name: ranges(p) for name, p in [("L", r"\p{L}"), ("N", r"\p{N}"), ("M", r"\p{M}"), ("s", r"\s")]},
        "contraction_folds": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--snapshot", type=Path, required=True, help="tokenizer files the profile's runtime loads")
    parser.add_argument("--asset", type=Path, required=True, help="the profile's pinned tokenizer.json")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-live", type=Path)
    parser.add_argument("--differential-out", type=Path)
    parser.add_argument("--differential-cases", type=int, default=1500)
    parser.add_argument("--decode-cases", type=int, default=500)
    parser.add_argument("--variant-cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--unicode-classes", type=Path)
    args = parser.parse_args()

    spec = PROFILES[args.profile]
    if version("transformers") != spec["transformers"] or version("tokenizers") != "0.23.2":
        print(f"environment mismatch: transformers {version('transformers')}, tokenizers {version('tokenizers')}")
        return 2

    from transformers import AutoTokenizer

    tf = AutoTokenizer.from_pretrained(str(args.snapshot))
    live = tf.backend_tokenizer.to_str(pretty=True).encode("utf-8")
    if args.write_live:
        args.write_live.parent.mkdir(parents=True, exist_ok=True)
        args.write_live.write_bytes(live)
    asset_bytes = args.asset.read_bytes()
    asset_sha = hashlib.sha256(asset_bytes).hexdigest()
    if asset_sha != spec["asset"]["sha256"]:
        print(f"{args.asset}: sha256 {asset_sha}, expected {spec['asset']['sha256']}")
        return 1
    live_record = {
        "transformers_class": type(tf).__name__,
        "expected_class": spec["class"],
        "live_pipeline_sha256": hashlib.sha256(live).hexdigest(),
        "live_pipeline_size": len(live),
        "live_equals_asset_bytes": live == asset_bytes,
        "hub_tokenizer_json": fixtures.file_record(args.snapshot / "tokenizer.json"),
    }
    if not live_record["live_equals_asset_bytes"] or type(tf).__name__ != spec["class"]:
        print(json.dumps(live_record, indent=2))
        return 1

    if args.unicode_classes:
        fixtures.write_json(args.unicode_classes, probe_unicode_classes())

    ref = Tokenizer.from_file(str(args.asset))
    if args.out:
        cases = [c for corpus in args.corpus for c in fixtures.load_corpus(corpus)]
        rows = tokenizer_rows(ref, cases)
        mismatches = []
        for case, row in zip(cases, rows, strict=True):
            if "pair" in case:
                tf_ids = tf(case["text"], case["pair"], add_special_tokens=False)["input_ids"]
            else:
                tf_ids = tf(case["text"], add_special_tokens=False)["input_ids"]
            if tf_ids != row["encoding"]["ids"]:
                mismatches.append({"id": case["id"], "transformers_ids": tf_ids})
        live_record["transformers_cross_check"] = {
            "call": "tokenizer(text[, pair], add_special_tokens=False)['input_ids']",
            "cases": len(cases),
            "mismatches": mismatches,
        }
        fixtures.write_jsonl(args.out / "tokenizer.jsonl", rows)
        fixtures.write_json(args.out / "live-pipeline.json", live_record)
        fixtures.write_json(
            args.out / "environment.json",
            {
                "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
                "runner": "reference/qwen-bpe/run_qwen_bpe_tokenizer.py",
                "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
                "profile": spec["profile"],
                "runtime_loaded": spec["load"],
                "asset": {**spec["asset"], "size": len(asset_bytes)},
                "snapshot_files": [
                    fixtures.file_record(args.snapshot / f, args.snapshot)
                    for f in SNAPSHOT_FILES
                    if (args.snapshot / f).exists()
                ],
                "reference": f"tokenizers {version('tokenizers')} Tokenizer.from_file(asset)",
                "corpora": [fixtures.file_record(c) for c in args.corpus],
                "offset_units": {
                    "source": "tokenizers Python binding, indices into the Python str (Unicode scalar values)",
                    "recorded": list(fixtures.OFFSET_UNITS),
                    "end": "exclusive",
                    "special_tokens": "(0, 0)",
                },
                "environment": fixtures.environment(PACKAGES),
            },
        )
        print(f"{spec['profile']}: {len(rows)} cases, transformers cross-check mismatches: {len(mismatches)}")

    if args.differential_out:
        rows = differential(args.asset, args.differential_cases, args.decode_cases, args.variant_cases, args.seed)
        header = {
            "v": "header",
            "profile": spec["profile"],
            "asset_sha256": asset_sha,
            "seed": args.seed,
            "cases": args.differential_cases,
            "decode_cases": args.decode_cases,
            "variant_cases": args.variant_cases,
            "generator": "reference/qwen-bpe/run_qwen_bpe_tokenizer.py random_text",
            "reference": f"tokenizers {version('tokenizers')}",
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
