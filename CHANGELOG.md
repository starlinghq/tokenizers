# Changelog

All notable changes to Starlings.Tokenizers (published as StarlingHq.Tokenizers for 0.2.0-preview.1). Versions follow [docs/releasing.md](docs/releasing.md): a change in token ids, offsets or decoded text is a behavior change and is always listed.

## [Unreleased]

## [0.3.2-preview.1] - 2026-10-19

### Added

- `ByteLevel` pre-tokenizer `use_regex: true`: the GPT-2 split regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`), matched over Unicode scalar values with the reference's character classes (a new `Gpt2SplitPattern`). Previously `use_regex: true` was rejected at load.
- Profile `whisper-bpe`: OpenAI Whisper's GPT-2 byte-level BPE tokenizer (vocab 50257 + 1609 added tokens, `ByteLevel(add_prefix_space false, use_regex true)` + `TemplateProcessing`).

### Behavior

- Tokenizers whose ByteLevel pre-tokenizer declares `use_regex: true` now load and split on the GPT-2 regex instead of failing with `UnsupportedTokenizerComponentException`. No change for profiles that declare `use_regex: false`.

## [0.3.1] - 2026-10-19

The code of 0.3.1-preview.1 released without a preview suffix, so that a non-prerelease ai-dotnet package can pin it (NuGet NU5104). Only the version and documentation changed.

### Behavior

- No change in output for any existing profile: none declares `lstrip` or `rstrip` (0.3.0 refused to load such tokenizers), and the same conformance fixtures pass unchanged.

## [0.3.1-preview.1] - 2026-10-18

### Added

- `lstrip` and `rstrip` added-token options: an added token can absorb the whitespace on its left/right, matching upstream `AddedVocabulary` semantics (`space_leftmost_at_end` / `space_rightmost_at_start`). `single_word` remains rejected at load time.

### Behavior

- Tokenizers whose added tokens declare `lstrip` or `rstrip` now load instead of failing with `UnsupportedTokenizerComponentException`; matching absorbs adjacent whitespace. No change for profiles that do not use these options.

## [0.3.0] - 2026-10-16

The first release without a preview suffix, so that the ai-dotnet model packages (0.2.0, the first release with Stable model profiles) can depend on a non-prerelease tokenizer. The code is the code of 0.3.0-preview.1; only the version and documentation changed.

### Changed

- Version scheme: releases are `0.MINOR.PATCH`; `-preview.N` is kept for candidates. The API can still change between minor versions (0.x). Install without `--prerelease`.

### Behavior

- No change in output for any profile: the same conformance fixtures pass unchanged.

## [0.3.0-preview.1] - 2026-10-13

### Changed

- Package ID and namespaces renamed from `StarlingHq.Tokenizers` to `Starlings.Tokenizers` (namespaces `Starlings.Tokenizers`, `Starlings.Tokenizers.Pipeline`). Replace the package reference and update `using` directives; the API and tokenizer output are otherwise unchanged. `StarlingHq.Tokenizers` 0.2.0-preview.1 is deprecated in favor of this package.

### Behavior

- No change in output for any profile: the same conformance fixtures pass unchanged.

## [0.2.0-preview.1] - 2026-10-12

First release from this repository, published as `StarlingHq.Tokenizers` (now deprecated; see 0.3.0-preview.1). Versions up to 0.1.0-preview.4 were built under a planning-era package ID in the Starling planning repository and never published ([docs/repository-history.md](docs/repository-history.md)).

### Changed

- License: dual MIT OR Apache-2.0 (previously MIT), copyright StarlingHq. Third-party portions keep their own permissive licenses and notices (`THIRD-PARTY-NOTICES.md`).
- Package ID and namespaces renamed from the planning-era name to `StarlingHq.Tokenizers` (the `Starling` ID prefix is reserved on nuget.org by another owner). Update `using` directives; the API is otherwise unchanged.
- Test and tool environment variables use the `STARLING_` prefix (`STARLING_MODELS`, `STARLING_REQUIRE_ASSETS`, `STARLING_QWEN_DIFFERENTIAL_DIR`, `STARLING_GEMMA_DIFFERENTIAL_DIR`).

### Added

- BERT-family WordPiece components, experimental: the `BertNormalizer` (defaults only) normalizer, the `BertPreTokenizer` pre-tokenizer, and the `WordPiece` model and decoder. They are covered by component reference cases and by loading `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L6-v2`. No profile yet.
- Data-driven conformance: every profile with fixture-format-1 cases is tested by `ProfileConformanceTests`, and `reference/tokenizer-json/run_tokenizer_json.py` writes those fixtures for any `tokenizer.json`. It reproduces the existing e5 and Gemma fixtures byte for byte.
- `tools/pinned-assets.txt` as the single download list, checked against the test pins and profiles.

### Behavior

- No change in output for any profile: the same conformance fixtures pass unchanged (`xlmr-unigram-e5` 49/49, `qwen2-bpe-kev` 137/137, `qwen35-bpe-chat` 137/137, `gemma-bpe-embeddinggemma2` 49/49, `gliner2-base-v1-unigram` partial).

## [0.1.0-preview.4] and earlier

Built under the planning-era package ID, not published. They added the Unigram interpreter (e5, GLiNER2), the Qwen byte-level BPE profiles, the Gemma SentencePiece-style BPE with byte fallback, and the fast load path. The records are in the profile files and `docs/benchmarks/`.
