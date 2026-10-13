# Repository history

This library was built in the Starling planning repository (private), under `tokenizers-dotnet/`, with a planning-era package name. On 2026-10-12 it was split out into this repository as `StarlingHq.Tokenizers`. This repository starts from a single snapshot commit: the earlier commit history stays in the planning repository.

## What moved

Every path below was taken from the planning repository at its commit `c14f8a1d307d75b7af876c687208e15a7c6a0386`. Its history before the move is in that repository, under the old paths:

| Planning repository | Here |
| --- | --- |
| `tokenizers-dotnet/` | repository root |
| `tools/fetch-tokenizer-assets.sh`, `check-tokenizer-independence.sh`, `release-drill-tokenizers.sh`, `generate-tokenizer-unicode-tables.py`, `release-drill/consumer/` | `tools/fetch-assets.sh`, `check-independence.sh`, `release-drill.sh`, `generate-unicode-tables.py`, `release-drill/consumer/` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/` (tokenizer files only) | `conformance/xlmr-unigram-e5/` |
| `model-pipeline/protected/gliner2-base-v1/tokenizer/` | `conformance/gliner2-base-v1-unigram/` |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/` (tokenizer files only) | `conformance/gemma-bpe-embeddinggemma2/` |
| `model-pipeline/protected/qwen-bpe/{kev,chat}/tokenizer/` | `conformance/qwen2-bpe-kev/`, `conformance/qwen35-bpe-chat/` |
| `model-pipeline/reference/corpus/tokenizer-corpus*.jsonl` | `conformance/corpus/` |
| `model-pipeline/reference/{qwen-bpe,gemma-bpe}/`, `gliner2/run_gliner2_tokenizer.py`, `compare_tokenizers.py` | `reference/` |
| `model-pipeline/src/model_pipeline/fixtures.py` | `reference/fixtures.py` |
| `docs/benchmarks/tokenizer-*.md` | `docs/benchmarks/` |
| `docs/m0/p023-tokenizer-source-audit.md` | `docs/design/source-audit.md` |

Every fixture file is byte-identical to its original. The staged Qwen copies under `tests/.../TestData/qwen-bpe/{kev,chat}/` duplicated `conformance/` and were removed. Model-level files (embeddings, parity reports, tolerances) stayed in the planning repository.

## Renames

| Before | After |
| --- | --- |
| planning-era package and namespace | `StarlingHq.Tokenizers` |
| planning-era environment variables | `STARLING_MODELS`, `STARLING_REQUIRE_ASSETS`, `STARLING_QWEN_DIFFERENTIAL_DIR`, `STARLING_GEMMA_DIFFERENTIAL_DIR` |
| release tag `tokenizers-v<version>` | `v<version>` |
| last version `0.1.0-preview.4` (never published) | first version `0.2.0-preview.1` |

## Reading older records

The profiles, `PROVENANCE.md`, the files under `conformance/`, `docs/benchmarks/` and `docs/design/` keep their original wording. Inside them:

- `P-0xx` is a backlog item and `ADR-nn` an architecture decision of the planning repository. "Plan section N" refers to its plan document.
- `PR #nn` is a pull request in the planning repository.
- `model-pipeline/...`, `tokenizers-dotnet/...`, `docs/m0/...` and `docs/kickoff.md` are planning-repository paths. Use the table above for the ones that moved.
- `PROPOSED.md` files under `conformance/` name the runner paths and commits of the planning repository. Those commits are the provenance of the fixtures and remain valid there.
