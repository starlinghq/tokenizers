"""Conformance fixtures for any tokenizer.json profile, from huggingface/tokenizers alone.

The generic runner for a new profile (docs/adding-a-tokenizer.md). It loads the pinned
``tokenizer.json`` with ``tokenizers.Tokenizer.from_file`` (the version pinned in requirements.txt)
and records, for every corpus case, what the reference produces. No model weights, no transformers.

Outputs (``--out``, normally ``conformance/<profile>/``):

- ``tokenizer.jsonl``: per corpus case (fixture format version 1, reference/README.md): ids, tokens,
  type ids, attention and special-token masks, offsets in three units (sequence ids and per-sequence
  offsets for pairs), decode with and without special tokens; with ``--truncate N``, a ``truncated``
  record (longest_first, right) for every case longer than N tokens. With
  ``--without-special-tokens``, a ``without_special_tokens`` record per case: the encoding with
  ``add_special_tokens=False`` (and its own ``truncated`` record when ``--truncate`` applies).
- ``tokenizer-batch.json`` (only with ``--batch``): the named single-text cases encoded together with
  right padding to the longest row (``--pad-token``), each row checked against its single encode; with
  ``--pair-batch``, the named pair cases as a second padded batch under ``pairs``.
- ``environment.json``: runner, its repository commit, interpreter, packages, CPU, and the asset and
  corpus digests.

The row layout is the one ``tests/Starlings.Tokenizers.Tests/ProfileConformanceTests.cs`` reads for every
profile that declares ``conformance.fixtures`` with a ``tokenizer.jsonl``.

Usage:
    python run_tokenizer_json.py --profile <id> --asset <.models/<repo>/<commit>/tokenizer.json> \\
        --corpus ../../conformance/corpus/tokenizer-corpus-v1.jsonl --out ../../conformance/<id> [--truncate 512] \\
        [--without-special-tokens] [--batch <id,id,...> --pad-token <token> [--pair-batch <id,id,...>]]

Without the optional flags the output is exactly what earlier versions of this runner wrote.
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


def encode(tok: Tokenizer, text: str, pair: str | None, add_special_tokens: bool = True):
    if pair is not None:
        return tok.encode(text, pair, add_special_tokens=add_special_tokens)
    return tok.encode(text, add_special_tokens=add_special_tokens)


def truncated_record(
    truncating: Tokenizer, text: str, pair: str | None, truncate: int, add_special_tokens: bool
) -> dict:
    tenc = encode(truncating, text, pair, add_special_tokens)
    return {
        "max_length": truncate,
        "strategy": "longest_first",
        "direction": "right",
        "encoding": encoding_record(text, tenc, pair),
    }


def tokenizer_rows(
    tok: Tokenizer, cases: list[dict], truncate: int | None, without_special: bool = False
) -> list[dict]:
    truncating = None
    if truncate is not None:
        truncating = copy.deepcopy(tok)
        truncating.enable_truncation(truncate)
    rows = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        enc = encode(tok, text, pair)
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
            row["truncated"] = truncated_record(truncating, text, pair, truncate, add_special_tokens=True)
        if without_special:
            plain = encode(tok, text, pair, add_special_tokens=False)
            record = {"encoding": encoding_record(text, plain, pair)}
            if truncating is not None and len(plain.ids) > truncate:
                record["truncated"] = truncated_record(truncating, text, pair, truncate, add_special_tokens=False)
            row["without_special_tokens"] = record
        rows.append(row)
    return rows


def padded_batch(tok: Tokenizer, cases: list[dict], pad_id: int, pad_token: str) -> dict:
    """Encodes the cases as one batch, right-padded to the longest, and checks each row against its single encode."""
    padded = copy.deepcopy(tok)
    padded.enable_padding(direction="right", pad_id=pad_id, pad_token=pad_token)
    inputs = [(c["text"], c["pair"]) if "pair" in c else c["text"] for c in cases]
    batch = padded.encode_batch(inputs)
    rows = []
    for case, row in zip(cases, batch, strict=True):
        single = encode(tok, case["text"], case.get("pair")).ids
        n = len(single)
        matches = (
            row.ids[:n] == single
            and all(i == pad_id for i in row.ids[n:])
            and row.attention_mask == [1] * n + [0] * (len(row.ids) - n)
        )
        rows.append({"id": case["id"], "ids": row.ids, "attention_mask": row.attention_mask, "matches_single": matches})
    return {"rows": rows, "equivalent": all(r["matches_single"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--profile", required=True, help="profile id, as in profiles/<id>.yaml")
    parser.add_argument("--asset", required=True, type=Path, help="the pinned tokenizer.json")
    parser.add_argument("--corpus", required=True, type=Path, action="append", help="corpus JSON Lines (repeatable)")
    parser.add_argument("--out", required=True, type=Path, help="fixture directory, normally conformance/<profile>")
    parser.add_argument("--truncate", type=int, help="also record truncation to this many tokens for longer cases")
    parser.add_argument(
        "--without-special-tokens", action="store_true", help="also record each case with add_special_tokens=False"
    )
    parser.add_argument("--batch", help="comma-separated single-text case ids to record as one padded batch")
    parser.add_argument("--pair-batch", help="comma-separated pair case ids to record as a second padded batch")
    parser.add_argument("--pad-token", help="the padding token of --batch (its id is looked up in the vocabulary)")
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

    by_id = {c["id"]: c for c in cases}
    batch = None
    if args.batch is not None:
        pad_id = tok.token_to_id(args.pad_token) if args.pad_token is not None else None
        if pad_id is None:
            print("error: --batch needs a --pad-token that is in the vocabulary", file=sys.stderr)
            return 1
        groups = {"rows": args.batch.split(","), "pairs": args.pair_batch.split(",") if args.pair_batch else []}
        for name, group in groups.items():
            if any(i not in by_id or ("pair" in by_id[i]) != (name == "pairs") for i in group):
                flag = "--pair-batch" if name == "pairs" else "--batch"
                print(f"error: {flag} names an unknown or wrong-kind case", file=sys.stderr)
                return 1
        singles = padded_batch(tok, [by_id[i] for i in groups["rows"]], pad_id, args.pad_token)
        batch = {"padding": {"side": "right", "pad_id": pad_id, "strategy": "longest"}, **singles}
        if groups["pairs"]:
            batch["pairs"] = padded_batch(tok, [by_id[i] for i in groups["pairs"]], pad_id, args.pad_token)
    elif args.pair_batch is not None or args.pad_token is not None:
        print("error: --pair-batch and --pad-token need --batch", file=sys.stderr)
        return 1

    rows = tokenizer_rows(tok, cases, args.truncate, args.without_special_tokens)
    fixtures.write_jsonl(args.out / "tokenizer.jsonl", rows)
    if batch is not None:
        fixtures.write_json(args.out / "tokenizer-batch.json", batch)
    options = {}
    if args.without_special_tokens:
        options["without_special_tokens"] = True
    if batch is not None:
        options["batch"] = groups["rows"]
        options["pair_batch"] = groups["pairs"]
        options["pad_token"] = args.pad_token
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "reference/tokenizer-json/run_tokenizer_json.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "profile": args.profile,
            "reference": "tokenizers.Tokenizer.from_file(asset)",
            "truncate": args.truncate,
            **({"options": options} if options else {}),
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
