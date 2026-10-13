"""Compare tokenizer implementations on one corpus, against the pinned tokenizer.json (P-004).

The reference row is the `tokenizers` library reading the snapshot's `tokenizer.json` as is.
Other rows are whatever the current environment offers: transformers' default (fast) class,
transformers' slow class when it exists, and raw SentencePiece with the model's ID mapping.
Only differences from the reference are stored, so agreement is the absence of an entry.

Usage:
    python compare_tokenizers.py --model-dir <snapshot> --corpus <jsonl> --mapping xlmr|deberta-v3 --out <json>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # reference/fixtures.py
import fixtures  # noqa: E402


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sentencepiece_encoder(model_dir: Path, mapping: str):
    """Raw SentencePiece with the special-token layout transformers' slow class uses.

    No added-token matching: text such as "<s>" is encoded as characters. Differences on
    special-token look-alike cases are expected and labeled as such.
    """
    import sentencepiece as spm

    names = {"xlmr": "sentencepiece.bpe.model", "deberta-v3": "spm.model"}
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / names[mapping]))
    if mapping == "xlmr":
        # fairseq layout: <s>=0 <pad>=1 </s>=2 <unk>=3, SentencePiece ids shifted by 1, spm <unk> (0) -> 3.
        def encode(text: str) -> list[int]:
            ids = [3 if i == 0 else i + 1 for i in sp.encode(text)]
            return [0, *ids, 2]
    else:
        # DeBERTa-v3: [CLS]=1 [SEP]=2, SentencePiece ids used directly.
        def encode(text: str) -> list[int]:
            return [1, *sp.encode(text), 2]

    return encode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mapping", choices=["xlmr", "deberta-v3"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [c for c in fixtures.load_corpus(args.corpus) if "pair" not in c]
    reference = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    impls: dict[str, dict] = {}
    encoders = {}

    from transformers import AutoTokenizer

    tv = version("transformers")
    fast = AutoTokenizer.from_pretrained(str(args.model_dir))
    impls[f"transformers-{tv}-default"] = {"class": type(fast).__name__}
    encoders[f"transformers-{tv}-default"] = lambda t: fast(t)["input_ids"]
    if tv and tv.startswith("4."):
        # Build the slow class directly: AutoTokenizer(use_fast=False) returns the fast class when
        # tokenizer_config.json names it explicitly.
        slow = type(fast).slow_tokenizer_class.from_pretrained(str(args.model_dir))
        impls[f"transformers-{tv}-slow"] = {"class": type(slow).__name__}
        encoders[f"transformers-{tv}-slow"] = lambda t: slow(t)["input_ids"]
    sv = version("sentencepiece")
    impls[f"sentencepiece-{sv}-raw"] = {"class": "SentencePieceProcessor", "note": "no added-token matching"}
    encoders[f"sentencepiece-{sv}-raw"] = sentencepiece_encoder(args.model_dir, args.mapping)

    differences = {name: [] for name in encoders}
    for case in cases:
        ref_ids = reference.encode(case["text"]).ids
        for name, encode in encoders.items():
            ids = list(encode(case["text"]))
            if ids != ref_ids:
                differences[name].append(
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "reference_tokens": reference.encode(case["text"]).tokens,
                        "reference_ids": ref_ids,
                        "ids": ids,
                        "tokens": [reference.id_to_token(i) for i in ids],
                    }
                )

    fixtures.write_json(
        args.out,
        {
            "reference": {
                "implementation": f"tokenizers-{version('tokenizers')}",
                "asset": fixtures.file_record(args.model_dir / "tokenizer.json", args.model_dir),
            },
            "corpus": fixtures.file_record(args.corpus),
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "environment": fixtures.environment(["tokenizers", "transformers", "sentencepiece", "protobuf"]),
            "cases": len(cases),
            "implementations": impls,
            "summary": {name: [d["id"] for d in diffs] for name, diffs in differences.items()},
            "differences": differences,
        },
    )
    for name, diffs in differences.items():
        print(f"{name}: {len(diffs)} of {len(cases)} differ: {[d['id'] for d in diffs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
