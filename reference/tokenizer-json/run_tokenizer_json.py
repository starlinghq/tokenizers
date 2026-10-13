"""Conformance fixtures for any tokenizer.json profile, from huggingface/tokenizers alone.

The generic runner for a new profile (docs/adding-a-tokenizer.md). It loads the pinned
``tokenizer.json`` with ``tokenizers.Tokenizer.from_file`` (the version pinned in requirements.txt)
and records, for every corpus case, what the reference produces. No model weights, no transformers.

Outputs (``--out``, normally ``conformance/<profile>/``):

- ``tokenizer.jsonl``: per corpus case (fixture format version 1, reference/README.md): ids, tokens,
  type ids, attention and special-token masks, offsets in three units (sequence ids and per-sequence
  offsets for pairs), decode with and without special tokens; with ``--truncate N``, a ``truncated``
  record (longest_first, right) for every case longer than N tokens.
- ``environment.json``: runner, its repository commit, interpreter, packages, CPU, and the asset and
  corpus digests.

The row layout is the one ``tests/Starling.Tokenizers.Tests/ProfileConformanceTests.cs`` reads for every
profile that declares ``conformance.fixtures`` with a ``tokenizer.jsonl``.

Usage:
    python run_tokenizer_json.py --profile <id> --asset <.models/<repo>/<commit>/tokenizer.json> \\
        --corpus ../../conformance/corpus/tokenizer-corpus-v1.jsonl --out ../../conformance/<id> [--truncate 512]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reference/fixtures.py
import fixtures  # noqa: E402

PACKAGES = ["tokenizers"]


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
        # Offsets of a pair index into the sequence the token came from (sequence_ids).
        seq = enc.sequence_ids
        record["sequence_ids"] = seq
        per_unit = {unit: [] for unit in fixtures.OFFSET_UNITS}
        for (s, e), sid in zip(enc.offsets, seq, strict=True):
            source = pair if sid == 1 else text
            for unit in fixtures.OFFSET_UNITS:
                per_unit[unit].append(fixtures.convert_offsets(source, [(s, e)], unit)[0])
        record["offsets"] = per_unit
    return record


def tokenizer_rows(tok: Tokenizer, cases: list[dict], truncate: int | None) -> list[dict]:
    truncating = None
    if truncate is not None:
        truncating = copy.deepcopy(tok)
        truncating.enable_truncation(truncate)
    rows = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        enc = tok.encode(text, pair) if pair is not None else tok.encode(text)
        row = {
            "id": case["id"],
            "category": case["category"],
            "text": text,
            "encoding": encoding_record(text, enc, pair),
            "decode": {
                "skip_special_tokens": tok.decode(enc.ids, skip_special_tokens=True),
                "keep_special_tokens": tok.decode(enc.ids, skip_special_tokens=False),
            },
        }
        if pair is not None:
            row["pair"] = pair
        if truncating is not None and len(enc.ids) > truncate:
            tenc = truncating.encode(text, pair) if pair is not None else truncating.encode(text)
            row["truncated"] = {
                "max_length": truncate,
                "strategy": "longest_first",
                "direction": "right",
                "encoding": encoding_record(text, tenc, pair),
            }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--profile", required=True, help="profile id, as in profiles/<id>.yaml")
    parser.add_argument("--asset", required=True, type=Path, help="the pinned tokenizer.json")
    parser.add_argument("--corpus", required=True, type=Path, action="append", help="corpus JSON Lines (repeatable)")
    parser.add_argument("--out", required=True, type=Path, help="fixture directory, normally conformance/<profile>")
    parser.add_argument("--truncate", type=int, help="also record truncation to this many tokens for longer cases")
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(args.asset))
    if tok.truncation is not None or tok.padding is not None:
        print(
            "error: the asset enables truncation or padding; the fixture format records untruncated, "
            "unpadded encodings (use --truncate for truncation)",
            file=sys.stderr,
        )
        return 1
    cases = [c for corpus in args.corpus for c in fixtures.load_corpus(corpus)]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        print("error: duplicate case ids across the corpora", file=sys.stderr)
        return 1

    rows = tokenizer_rows(tok, cases, args.truncate)
    fixtures.write_jsonl(args.out / "tokenizer.jsonl", rows)
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "reference/tokenizer-json/run_tokenizer_json.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "profile": args.profile,
            "reference": "tokenizers.Tokenizer.from_file(asset)",
            "truncate": args.truncate,
            "asset": fixtures.file_record(args.asset),
            "corpora": [fixtures.file_record(c) for c in args.corpus],
            "cases": len(rows),
            "environment": fixtures.environment(PACKAGES),
        },
    )
    print(f"{len(rows)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
