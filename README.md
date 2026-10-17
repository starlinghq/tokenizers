# Starlings.Tokenizers

Hugging Face `tokenizer.json` files in pure .NET, with the same token ids, offsets and decoded text as the Python and Rust `tokenizers` library, checked case by case.

```csharp
using Starlings.Tokenizers;

Tokenizer tokenizer = Tokenizer.FromFile("tokenizer.json");
TokenEncoding encoding = tokenizer.Encode("Hello, world!");
string text = tokenizer.Decode(encoding.Ids.AsSpan(), skipSpecialTokens: true);
```

- **Faithful.** Every supported tokenizer has a profile: a pinned `tokenizer.json` (full commit SHA and SHA-256) with reference outputs from huggingface/tokenizers for multilingual text, emoji, combining marks, control characters, pairs, truncation and padding. CI fails on any difference.
- **Explicit.** A `tokenizer.json` that uses a component this library does not implement fails at load with `UnsupportedTokenizerComponentException`, naming the JSON path. Nothing is approximated, and unknown fields are rejected rather than ignored.
- **Plain .NET.** No native code, no package dependencies, `net10.0`. Instances are immutable and thread-safe. A 17 MB vocabulary loads in about 0.27 s, about half the time the Rust library takes on the same machine.

Part of [Starling](https://github.com/starlinghq), .NET implementations of AI models with reproducible evidence of what works.

> **0.x.** The public API can still change between minor versions. A profile's status is in its file under `profiles/`; the Starling model profiles that use a profile pin the exact tokenizer version they were qualified with.

## Install

```sh
dotnet add package Starlings.Tokenizers
```

## Usage

```csharp
using Starlings.Tokenizers;

Tokenizer tokenizer = Tokenizer.FromFile("tokenizer.json")            // also FromStream, FromUtf8Json
    .WithTruncation(new TruncationOptions(MaxLength: 512))
    .WithPadding(new PaddingOptions(PadId: 1, PadToken: "<pad>"));      // no FixedLength: pad to the batch's longest

TokenEncoding one = tokenizer.Encode("query: how much protein should a female eat");
TokenEncoding pair = tokenizer.Encode("What is the capital of France?", "Paris is the capital of France.");
IReadOnlyList<TokenEncoding> batch = tokenizer.EncodeBatch(["first text", "second, longer text"]);

int[] ids = [.. one.Ids];                    // also Tokens, TypeIds, AttentionMask, SpecialTokensMask, SequenceIds
TokenOffset span = one.Offsets[1];           // UTF-16 code units: slice the C# string with these
TokenOffset scalars = one.ScalarOffsets[1];  // Unicode scalar values: the unit Python tokenizers reports

string decoded = tokenizer.Decode(one.Ids.AsSpan(), skipSpecialTokens: true);
int? eos = tokenizer.TokenToId("</s>");
```

- `WithTruncation` and `WithPadding` return new instances that share the loaded pipeline. Without them, the file's own `truncation` and `padding` sections apply.
- `Encode(text, addSpecialTokens: false)` skips the post-processor's special tokens. Special tokens written in the input text are still matched as added tokens.
- Input containing unpaired UTF-16 surrogates is rejected with `ArgumentException`.
- `tokenizer.Components` lists the pipeline stages the file uses, for example `model:BPE`.

## Supported tokenizers

A model is supported when its `tokenizer.json` uses only supported components. A profile is the evidence: one pinned file and the cases it matches.

| Profile | Model | Algorithm | Evidence |
| --- | --- | --- | --- |
| [`xlmr-unigram-e5`](https://github.com/starlinghq/tokenizers/blob/main/profiles/xlmr-unigram-e5.yaml) | `intfloat/multilingual-e5-small` | SentencePiece Unigram | 49/49 cases exact, with truncation and padded batch |
| [`gliner2-base-v1-unigram`](https://github.com/starlinghq/tokenizers/blob/main/profiles/gliner2-base-v1-unigram.yaml) | `fastino/gliner2-base-v1` | SentencePiece Unigram (DeBERTa-v3) | per-word and 10 whole-text cases; offsets, pairs and decode untested |
| [`qwen2-bpe-kev`](https://github.com/starlinghq/tokenizers/blob/main/profiles/qwen2-bpe-kev.yaml) | `jaredpalmer/kev-0.8b` (Qwen2/Qwen3 regex) | byte-level BPE | 137/137 cases exact, 0 mismatches in 1.15 M randomized encodes |
| [`qwen35-bpe-chat`](https://github.com/starlinghq/tokenizers/blob/main/profiles/qwen35-bpe-chat.yaml) | `Qwen/Qwen3.5-0.8B` | byte-level BPE | 137/137 cases exact, 0 mismatches in 1.15 M randomized encodes |
| [`gemma-bpe-embeddinggemma2`](https://github.com/starlinghq/tokenizers/blob/main/profiles/gemma-bpe-embeddinggemma2.yaml) | `google/embeddinggemma-2` | SentencePiece-style BPE, byte fallback | 49/49 cases exact, 0 mismatches in 1.7 M randomized encodes |

All five are experimental. BERT-family WordPiece files (for example `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`) load and pass component-level reference cases, but have no profile yet. Other files may load and work too. Without a profile, matching the reference is untested, not promised.

The full list of components and options, and what fails at load, is in [docs/supported-components.md](https://github.com/starlinghq/tokenizers/blob/main/docs/supported-components.md). Not offered: pre-tokenized input, word ids, overflowing windows (truncation stride), incremental (streaming) decode, and chat templates.

Need a tokenizer that is not here? [Open a tokenizer request](https://github.com/starlinghq/tokenizers/issues/new?template=tokenizer-request.yml) with the error `Tokenizer.FromFile` gives you.

## Documentation

| Page | Contents |
| --- | --- |
| [Supported components](https://github.com/starlinghq/tokenizers/blob/main/docs/supported-components.md) | Every `tokenizer.json` component and option, byte fallback and regex details |
| [Adding a tokenizer](https://github.com/starlinghq/tokenizers/blob/main/docs/adding-a-tokenizer.md) | From a new model's `tokenizer.json` to a released profile |
| [Architecture](https://github.com/starlinghq/tokenizers/blob/main/docs/architecture.md) | How the interpreter is built and where each stage lives |
| [Performance](https://github.com/starlinghq/tokenizers/blob/main/docs/performance.md) | Load and encode measurements against the Rust library |
| [Releasing](https://github.com/starlinghq/tokenizers/blob/main/docs/releasing.md) | Versions, the tag-to-NuGet workflow, one-time setup |
| [Changelog](https://github.com/starlinghq/tokenizers/blob/main/CHANGELOG.md) | What changed in each version |
| [`llms.txt`](https://github.com/starlinghq/tokenizers/blob/main/llms.txt) | A compact guide for language models and coding assistants using this library |

## Contributing

See [CONTRIBUTING.md](https://github.com/starlinghq/tokenizers/blob/main/CONTRIBUTING.md). Coding agents should read [AGENTS.md](https://github.com/starlinghq/tokenizers/blob/main/AGENTS.md) first.

## License

Copyright (c) 2026 StarlingHq. Licensed under either of

- [MIT License](https://github.com/starlinghq/tokenizers/blob/main/LICENSE-MIT), or
- [Apache License, Version 2.0](https://github.com/starlinghq/tokenizers/blob/main/LICENSE-APACHE)

at your option. Use it in open-source or commercial software, modify it, and ship it without releasing your own code. Apache-2.0 adds an explicit patent grant. Unless you state otherwise, any contribution you submit for inclusion is dual-licensed the same way, with no additional terms.

Some files are ported or adopted from huggingface/tokenizers, SentencePiece and spm_precompiled (Apache-2.0), dotnet/machinelearning (MIT), darts-clone (BSD 2-Clause), serde_json and unicode-normalization-alignments (MIT OR Apache-2.0), and Unicode data (Unicode License V3). All of these are permissive, and none of them adds conditions beyond keeping their notices. The notices are in [THIRD-PARTY-NOTICES.md](https://github.com/starlinghq/tokenizers/blob/main/THIRD-PARTY-NOTICES.md), which ships inside the package, so distributing the package as-is satisfies them. File-by-file origins are in [PROVENANCE.md](https://github.com/starlinghq/tokenizers/blob/main/PROVENANCE.md). Each model's `tokenizer.json` keeps its owner's terms and is never included in the package.
