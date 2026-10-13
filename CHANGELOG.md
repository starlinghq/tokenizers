# Changelog

All notable changes to Starling.Tokenizers. Versions follow [docs/releasing.md](docs/releasing.md): a change in token ids, offsets or decoded text is a behavior change and is always listed.

## [Unreleased]

## [0.2.0-preview.1] - 2026-10-12

First release from this repository and under this package ID. Versions up to 0.1.0-preview.4 were built under a planning-era package ID in the Starling planning repository and never published ([docs/repository-history.md](docs/repository-history.md)).

### Changed

- License: dual MIT OR Apache-2.0 (previously MIT), copyright StarlingHq. Third-party portions keep their own permissive licenses and notices (`THIRD-PARTY-NOTICES.md`).
- Package ID and namespaces renamed from the planning-era name to `Starling.Tokenizers`. Update `using` directives; the API is otherwise unchanged.
- Test and tool environment variables use the `STARLING_` prefix (`STARLING_MODELS`, `STARLING_REQUIRE_ASSETS`, `STARLING_QWEN_DIFFERENTIAL_DIR`, `STARLING_GEMMA_DIFFERENTIAL_DIR`).

### Added

- BERT-family WordPiece components, experimental: the `BertNormalizer` (defaults only) normalizer, the `BertPreTokenizer` pre-tokenizer, and the `WordPiece` model and decoder. They are covered by component reference cases and by loading `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L6-v2`. No profile yet.
- Data-driven conformance: every profile with fixture-format-1 cases is tested by `ProfileConformanceTests`, and `reference/tokenizer-json/run_tokenizer_json.py` writes those fixtures for any `tokenizer.json`. It reproduces the existing e5 and Gemma fixtures byte for byte.
- `tools/pinned-assets.txt` as the single download list, checked against the test pins and profiles.

### Behavior

- No change in output for any profile: the same conformance fixtures pass unchanged (`xlmr-unigram-e5` 49/49, `qwen2-bpe-kev` 137/137, `qwen35-bpe-chat` 137/137, `gemma-bpe-embeddinggemma2` 49/49, `gliner2-base-v1-unigram` partial).

## [0.1.0-preview.4] and earlier

Built under the planning-era package ID, not published. They added the Unigram interpreter (e5, GLiNER2), the Qwen byte-level BPE profiles, the Gemma SentencePiece-style BPE with byte fallback, and the fast load path. The records are in the profile files and `docs/benchmarks/`.
