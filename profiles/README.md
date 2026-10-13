# Tokenizer profiles

One file per supported profile: the exact asset set (with SHA-256 digests and upstream revision), the pipeline components it uses, and the conformance cases that qualify it. A profile is a tested combination, not a model name.

A new model whose `tokenizer.json` uses only supported components needs a new profile file, a pin and reference fixtures, with no library code change: see [docs/adding-a-tokenizer.md](../docs/adding-a-tokenizer.md). `ProfileConformanceTests` runs every profile whose `conformance.fixtures` directory holds a `tokenizer.jsonl`, and `PinnedAssetsTests` checks that each profile's `asset` block is pinned for the tests.

Every capability carries an explicit status: `supported` (exact on the listed conformance cases, in this repository's CI), `experimental` (implemented, with partial or unit-level evidence), `untested`, or `unsupported` (rejected at load, or not offered). A missing entry never means supported.

Fields such as `backlog`, PR numbers and `docs/m0/...` or `model-pipeline/...` paths are records from the Starling planning repository, where these profiles were first qualified ([docs/repository-history.md](../docs/repository-history.md)).

| Profile | Asset | Status |
| --- | --- | --- |
| [xlmr-unigram-e5](xlmr-unigram-e5.yaml) | `intfloat/multilingual-e5-small` `tokenizer.json` at `614241f622f53c4eeff9890bdc4f31cfecc418b3` | experimental: 49/49 fixture cases exact |
| [gliner2-base-v1-unigram](gliner2-base-v1-unigram.yaml) | `fastino/gliner2-base-v1` `tokenizer.json` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e` | experimental: per-word and 10 whole-text cases exact; offsets, pairs and decode untested |
| [qwen2-bpe-kev](qwen2-bpe-kev.yaml) | `jaredpalmer/kev-0.8b` `tokenizer.json` at `9a45d25eb2ab761841196625383fa1dff0e56c1e` (Kev's runtime pipeline under transformers 5.17.0) | experimental: 137/137 fixture cases exact; 0 mismatches on 1.15 M differential encodes; fixtures proposed, not yet accepted |
| [qwen35-bpe-chat](qwen35-bpe-chat.yaml) | the pipeline transformers 5.18.0 builds for `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (derived asset, SHA-256 `a5cd9732…`) | experimental: 137/137 fixture cases exact; 0 mismatches on 1.15 M differential encodes; fixtures proposed, not yet accepted |
| [gemma-bpe-embeddinggemma2](gemma-bpe-embeddinggemma2.yaml) | `google/embeddinggemma-2` `tokenizer.json` at `914f7f89142e33e77833254d9c9b90c3cef7303b` (SentencePiece-style BPE with byte fallback) | experimental: 49/49 accepted fixture cases exact; 0 mismatches on 1.7 M differential encodes |
