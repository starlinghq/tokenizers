"""GLiNER2 per-word tokenization fixtures (backlog P-004, P-014). CPU, no model weights.

Reproduces the input encoding GLiNER2 builds at inference, at the pinned source revision:
`SchemaTransformer.transform_record(text, schema)` with the tokenizer the checkpoint ships,
loaded through `gliner2.models.base.load_extractor_tokenizer` (fast by default). Each word from
the regex word splitter is lower-cased and tokenized on its own; the tokenizer post-processor
is not used.

For every corpus case the fixture records the normalized text, words with offsets in three
declared units, per-word subwords, the full input ids, and routing positions. The same records
are rebuilt with the slow (SentencePiece) tokenizer and every difference is listed.

Usage:
    python run_gliner2_tokenizer.py --model-dir <snapshot> --revision <sha> --corpus <jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gliner2
from gliner2.inference.schema import Schema
from gliner2.models.base import load_extractor_tokenizer
from gliner2.processor import SchemaTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reference/fixtures.py
import fixtures  # noqa: E402

PACKAGES = ["gliner2", "transformers", "tokenizers", "sentencepiece", "torch", "protobuf"]
ASSETS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "config.json",
    "encoder_config/config.json",
]
ENTITY_LABELS = ["person", "location"]
CHAR_SPLITTER_IDS = {"chinese", "japanese", "korean", "thai", "cjk-ext-b"}


def gliner2_source_revision() -> str | None:
    """Commit the installed gliner2 came from (pip direct_url.json), if recorded."""
    import importlib.metadata
    import json

    dist = importlib.metadata.distribution("gliner2")
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    info = json.loads(text)
    return info.get("vcs_info", {}).get("commit_id")


def record(processor: SchemaTransformer, text: str, schema) -> dict:
    normalized = processor._normalize_text(text)
    rec = processor.transform_record(text, schema)
    spans = list(zip(rec.start_token_idx, rec.end_token_idx, strict=True))
    words = []
    for word, (start, end) in zip(rec.text_tokens, spans, strict=True):
        words.append(
            {
                "word": word,
                "subwords": list(processor._tokenize_cached(word)),
                "offsets": {
                    u: fixtures.convert_offsets(normalized, [(start, end)], u)[0] for u in fixtures.OFFSET_UNITS
                },
            }
        )
    return {
        "normalized_text": normalized,
        "words": words,
        "schema_tokens": rec.schema_tokens_list,
        "input_ids": list(rec.input_ids),
        "text_word_first_positions": list(rec.text_word_first_positions),
        "schema_special_positions": [list(x) for x in rec.schema_special_positions],
    }


def compare(fast: dict, slow: dict) -> list[dict]:
    diffs = []
    for i, (fw, sw) in enumerate(zip(fast["words"], slow["words"], strict=True)):
        if fw["subwords"] != sw["subwords"]:
            diffs.append({"word_index": i, "word": fw["word"], "fast": fw["subwords"], "slow": sw["subwords"]})
    if not diffs and fast["input_ids"] != slow["input_ids"]:
        diffs.append({"word_index": None, "note": "input ids differ with equal subwords (id mapping)"})
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [c for c in fixtures.load_corpus(args.corpus) if "pair" not in c]
    schema = Schema().entities(ENTITY_LABELS)
    fast_tok = load_extractor_tokenizer(str(args.model_dir))
    # AutoTokenizer(use_fast=False) still returns the fast class when tokenizer_config.json names
    # it explicitly (this checkpoint does), so build the slow class directly.
    slow_tok = type(fast_tok).slow_tokenizer_class.from_pretrained(str(args.model_dir))
    fast = SchemaTransformer(tokenizer=fast_tok, token_pooling="first")
    slow = SchemaTransformer(tokenizer=slow_tok, token_pooling="first")
    fast_char = SchemaTransformer(tokenizer=fast_tok, token_pooling="first", word_splitter="char")

    special_ids = {t: fast_tok.convert_tokens_to_ids(t) for t in SchemaTransformer.SPECIAL_TOKENS}
    slow_special_ids = {t: slow_tok.convert_tokens_to_ids(t) for t in SchemaTransformer.SPECIAL_TOKENS}

    rows, differences = [], []
    for case in cases:
        f = record(fast, case["text"], schema)
        s = record(slow, case["text"], schema)
        row = {"id": case["id"], "category": case["category"], "text": case["text"], "whitespace_splitter": f}
        if case["id"] in CHAR_SPLITTER_IDS:
            row["char_splitter"] = record(fast_char, case["text"], schema)
        rows.append(row)
        diff = compare(f, s)
        if diff:
            differences.append({"id": case["id"], "category": case["category"], "words": diff})
    fixtures.write_jsonl(args.out / "gliner2-tokenizer.jsonl", rows)

    fixtures.write_json(
        args.out / "gliner2-fast-vs-slow.json",
        {
            "fast": type(fast_tok).__name__,
            "slow": type(slow_tok).__name__,
            "special_token_ids": {
                "fast": special_ids,
                "slow": slow_special_ids,
                "equal": special_ids == slow_special_ids,
            },
            "cases": len(cases),
            "cases_with_differences": [d["id"] for d in differences],
            "differences": differences,
        },
    )
    env = fixtures.environment(PACKAGES)
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "reference/gliner2/run_gliner2_tokenizer.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": "fastino/gliner2-base-v1", "revision": args.revision},
            "gliner2_source": {
                "url": "https://github.com/fastino-ai/GLiNER2",
                "revision": gliner2_source_revision(),
                "version": gliner2.__version__,
            },
            "schema": schema.build() | {"entities": dict(schema.build()["entities"])},
            "processor": {"entry_point": "SchemaTransformer.transform_record", "token_pooling": "first"},
            "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
            "corpus": fixtures.file_record(args.corpus),
            "offset_units": {
                "source": "gliner2 word splitter, indices into the normalized Python str (Unicode scalar values)",
                "normalized_text": "text after SchemaTransformer._normalize_text "
                "(appends '.' unless it ends in . ! or ?)",
                "recorded": list(fixtures.OFFSET_UNITS),
                "end": "exclusive",
            },
            "environment": env,
        },
    )
    print(f"cases={len(cases)} fast_vs_slow_differences={len(differences)}: {[d['id'] for d in differences]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
