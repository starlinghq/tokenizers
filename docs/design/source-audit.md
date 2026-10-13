# P-023: Tokenizer source audit

> **Historical record.** Written in the Starling planning repository before this repository was split out. The package then had a planning-era name (it is `Starling.Tokenizers` throughout here); paths such as `tokenizers-dotnet/`, `model-pipeline/` and `tools/*-tokenizer-*` refer to that repository's layout (see [docs/repository-history.md](../repository-history.md)).

Status: audit complete, adoption not started. Checked 2026-10-05. Backlog item P-023 (plan sections 6, 6.1, 6.3; ADR-13, ADR-14, ADR-15). Kickoff row: "Tokenizer source to adopt".

Environment limits: huggingface.co was blocked, so no `tokenizer.json` was read from the Hub. GitHub clones, raw.githubusercontent.com, PyPI and api.nuget.org worked. Claims are tagged:

- **[code]** read from source at a recorded commit or package version.
- **[run]** reproduced locally in a scratch project or venv (outside this repository).
- **[vendored]** read from a third-party GitHub copy of a Hub asset.
- **[unverified]** not confirmed.

## 1. Sources examined

| Source | URL | Revision (full SHA) | Date | License |
| --- | --- | --- | --- | --- |
| dotnet/machinelearning (Microsoft.ML.Tokenizers) | https://github.com/dotnet/machinelearning | `5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d` (HEAD of `main`) | 2026-10-01 | MIT, Copyright (c) .NET Foundation and Contributors |
| Microsoft.ML.Tokenizers NuGet `3.0.0-preview.26457.2` | https://www.nuget.org/packages/Microsoft.ML.Tokenizers | nuspec `repository commit` = `2fc13bab635827acdb4fb4ba3bef2b78673cd215`; `src/Microsoft.ML.Tokenizers` differs from HEAD above only in `PACKAGE.md` [code] | published 2026-09-09 | MIT |
| huggingface/tokenizers, HEAD | https://github.com/huggingface/tokenizers | `bbccb0513ff9afda385ca5c85c66eddb1318cfc7` | 2026-09-23 | Apache-2.0 |
| huggingface/tokenizers, latest stable `v0.23.2` | same | `88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3` | PyPI 0.23.2 uploaded 2026-09-03 | Apache-2.0 |
| huggingface/transformers, HEAD | https://github.com/huggingface/transformers | `10502571152db764f244791b7054481a7f629801` | 2026-10-05 | Apache-2.0 |
| huggingface/transformers, `v4.57.6` | same | `753d61104116eefc8ffc977327b441ee0c8d599f` | | Apache-2.0 |
| fastino-ai/GLiNER2 | https://github.com/fastino-ai/GLiNER2 | `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` | 2026-09-24 | Apache-2.0 |
| georg-jung/FastBertTokenizer | https://github.com/georg-jung/FastBertTokenizer (repo URL from NuGet registration, package 1.2.6) | `31f7cb322ff29a915fc7b4236dd4144a2b3f1ca6` | 2026-09-14 | MIT, Copyright (c) 2023 Georg Jung |
| google/sentencepiece (test models for probes only) | https://github.com/google/sentencepiece | `6ff920437d98db144ebd861a1180616fbcd5355a` | 2026-10-04 | Apache-2.0 |
| Vendored DeBERTa-v3-xsmall `tokenizer.json` (probe input only) | https://github.com/jesvijonathan/Snitch-GPT `models/debertav3xsmall/tokenizer.json` | `fe4c7f5324938114c3586825a8deb24b38035388`; file SHA-256 `05402ffae6dd382a8491b1d29bfc139bec5d332662e86a026f433ce54c25c202` (same hash as in `docs/m0/encoder-selection.md`) | | model asset, license not checked [vendored] |

Nothing was copied into this repository. All clones and probes live in the session scratchpad.

Release-lag facts from plan section 6, re-checked [code]: PR #7556 (gpt-5.1 mapping) landed on `main` as `de4eba2d` on 2025-11-19. NuGet lists no stable Microsoft.ML.Tokenizers release after 2.0.0 (2025-11-11); the two later packages are `3.0.0-preview.26160.2` (2026-03-12) and `3.0.0-preview.26457.2` (2026-09-09). Issue #6901 status was not checked (GitHub web and API blocked) [unverified].

## 2. Microsoft.ML.Tokenizers inventory

Path prefix: `src/Microsoft.ML.Tokenizers/` at `5a44dfdb`. About 20,450 lines of C#.

| Area | Classes and files | Notes |
| --- | --- | --- |
| Base API | `Tokenizer.cs`, `EncodeResults.cs`, `EncodeSettings.cs`, `EncodedToken.cs` | Token offsets are a `Range` in UTF-16 units of the normalized text when normalization is applied (`EncodeResults.NormalizedText`); no alignment map back to the original string [code]. |
| BPE | `Model/BPETokenizer.cs`, `BpeOptions.cs`, `Word.cs`, `Merge.cs`, `Pair.cs`, `Symbol.cs`, `Cache.cs`, `Utils/ByteToUnicodeEncoding.cs`, `Utils/PriorityQueue.cs`, `Utils/LruCache.cs` | vocab, merges, unk, continuing_subword_prefix, end_of_word_suffix, fuse_unk, `ByteLevel` flag (added for DeepSeek, #7425), internal dropout. No `byte_fallback`, no `ignore_merges`. Loads `vocab.json` + `merges.txt`, not `tokenizer.json`. |
| English RoBERTa | `Model/EnglishRobertaTokenizer.cs`, `PreTokenizer/RobertaPreTokenizer.cs` | GPT-2 style BPE with fairseq dictionary. |
| Tiktoken | `Model/TiktokenTokenizer.cs`, `Utils/BytePairEncoder.cs` | Model-name tables for OpenAI encodings. Data in separate packages. |
| SentencePiece BPE | `Model/SentencePieceBpeModel.cs`, `SentencePieceBaseModel.cs`, `SentencePieceTokenizer.cs`, `Model/LlamaTokenizer.cs` | Reads `.model` protobuf via `SentencepieceModel.cs` (hand-written reader; Google.Protobuf removed in #7587, 2026-03-11). |
| SentencePiece Unigram | `Model/SentencePieceUnigramModel.cs`, `Utils/DoubleArrayTrie.cs` | Viterbi over a double-array trie. Reads `.model` protobuf, an in-memory `(piece, score)` list, or `tokenizer.json` (see section 3.3). |
| WordPiece / BERT | `Model/WordPieceTokenizer.cs`, `WordPieceOptions.cs`, `BertTokenizer.cs`, `BertOptions.cs`, `Normalizer/BertNormalizer.cs` | Loads `vocab.txt`. `BertTokenizer` builds pair inputs, type ids and special-token masks in code. |
| CodeGen, Phi-2 | `Model/CodeGenTokenizer.cs`, `Model/Phi2Tokenizer.cs` | Byte-level BPE with hard-coded added tokens copied from the Hub `tokenizer.json` of those models. |
| Normalizers | `Normalizer/Normalizer.cs` (abstract, returns `string`), `LowerCaseNormalizer.cs`, `UpperCaseNormalizer.cs`, `BertNormalizer.cs`, `SentencePieceNormalizer.cs` (precompiled charsmap, dummy prefix, whitespace escaping), `SentencePieceNormalizationStep.cs` (internal: Sequence, Precompiled, Strip, Replace, Lowercase, StripAccents, NFC/NFD/NFKC/NFKD, Prepend, Nmt) | The internal step chain is only reachable from the Unigram `tokenizer.json` factory. |
| Pre-tokenizers | `PreTokenizer/PreTokenizer.cs` (factories: `CreateWordOrPunctuation` `\w+|[\p{P}]`, `CreateWordOrNonWord` `\w+|[^\w\s]+`, `CreateWhiteSpace` `\S+`), `RegexPreTokenizer.cs`, `CompositePreTokenizer.cs`, `RobertaPreTokenizer.cs` | `RegexPreTokenizer` yields regex matches only and drops gaps. |
| Decoders | No decoder classes. Decoding is a method on each model (`Decode`, `DecodeByteLevel`, SentencePiece byte pieces, WordPiece prefix) | Not composable; no `tokenizer.json` decoder is read anywhere. |
| Loaders | `SentencePieceTokenizer.CreateFromTokenizerJson` (Unigram only, #7625, 2026-07-16, shipped in `3.0.0-preview.26457.2` [code: string present in the package DLL]) | The only production `tokenizer.json` reader. A BPE `tokenizer.json` mapping exists only as test code (`test/Microsoft.ML.Tokenizers.Tests/BpeTests.cs`, `CreateBpeTokenizerFromJson`), which reads `fuse_unknown_tokens` where the format key is `fuse_unk`. |

## 3. `tokenizer.json` component coverage

### 3.1 The format

The deployed format is `"version": "1.0"`, written by tokenizers 0.x. Type tags below are the serialized names from `tokenizers/src/*/mod.rs` and the component structs at `v0.23.2` [code]. Deserialization also accepts a few aliases (`"Bert"` for `BertNormalizer`, `"Delimiter"` for `CharDelimiterSplit`) and legacy untyped models.

| Slot | Type tags | Count |
| --- | --- | --- |
| `normalizer` | `BertNormalizer`, `Strip`, `StripAccents`, `NFC`, `NFD`, `NFKC`, `NFKD`, `Sequence`, `Lowercase`, `Nmt`, `Precompiled`, `Replace`, `Prepend`, `ByteLevel` | 14 |
| `pre_tokenizer` | `BertPreTokenizer`, `ByteLevel`, `CharDelimiterSplit`, `Metaspace`, `Whitespace`, `Sequence`, `Split`, `Punctuation`, `WhitespaceSplit`, `Digits`, `UnicodeScripts`, `FixedLength` | 12 |
| `model` | `BPE`, `WordPiece`, `WordLevel`, `Unigram` | 4 |
| `post_processor` | `RobertaProcessing`, `BertProcessing`, `ByteLevel`, `TemplateProcessing`, `Sequence` | 5 |
| `decoder` | `BPEDecoder`, `ByteLevel`, `WordPiece`, `Metaspace`, `CTC`, `Sequence`, `Replace`, `Fuse`, `Strip`, `ByteFallback` | 10 |
| top level | `version`, `added_tokens` (`id`, `content`, `single_word`, `lstrip`, `rstrip`, `normalized`, `special`), `truncation` (`direction` left/right, `max_length`, `strategy` LongestFirst/OnlyFirst/OnlySecond, `stride`), `padding` (`strategy` BatchLongest/Fixed(n), `direction`, `pad_to_multiple_of`, `pad_id`, `pad_type_id`, `pad_token`) | 3 structures |

Parameterized options that matter for exactness: `Split` and `Punctuation` behaviors (`Removed`, `Isolated`, `MergedWithPrevious`, `MergedWithNext`, `Contiguous`) and `invert`; `Replace` and `Split` patterns as `String` or `Regex` (Oniguruma syntax in the reference); `Metaspace` `replacement`, `prepend_scheme` (`always`, `first`, `never`) and `split`; `ByteLevel` `add_prefix_space`, `trim_offsets`, `use_regex`; `BPE` `dropout`, `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `fuse_unk`, `byte_fallback`, `ignore_merges`; `Unigram` `unk_id`, `byte_fallback`; `TemplateProcessing` `single`, `pair`, `special_tokens` with type ids.

Format transition upstream [code]: tokenizers HEAD is a 1.0 rewrite (`1.0.0-dev.0`; PyPI `1.0.0rc1`/`rc2` uploaded 2026-09-21). Its writer emits `"version": "2.0"` (`tk-serialize/src/to_json/mod.rs`), its reader accepts only `2.0`, and `tk-convert/src/convert.rs` rewrites 1.0 files: a `Metaspace` pre-tokenizer becomes a `MetaspaceNormalizer` plus a `Split`; a `ByteLevel` pre-tokenizer becomes `"byte_level": true` on the BPE model plus an optional `Split`; `BertProcessing`/`RobertaProcessing` become `TemplateProcessing`; `merges` become pairs. The `1.0.0rc2` wheel refused to load the 1.0-format DeBERTa-style probe files (no regex backend compiled; "slim JSON reader does not support a `Metaspace` pre-tokenizer") [run]. GLiNER2 pins `transformers>=4.38,<5`, which uses tokenizers 0.x and the 1.0 format.

### 3.2 Matrix

"Impl": Microsoft.ML.Tokenizers has an equivalent algorithm. "JSON": it reads that component from `tokenizer.json` in production code. "P1": needed by the encoder/GLiNER2 Unigram profile. "P2": needed by the Qwen byte-level BPE profile. Paths are relative to `src/Microsoft.ML.Tokenizers/`.

| Slot | Type | Impl | JSON | P1 | P2 | Evidence and gaps |
| --- | --- | --- | --- | --- | --- | --- |
| normalizer | BertNormalizer | yes | no | | | `Normalizer/BertNormalizer.cs`, configured from `BertOptions`. |
| normalizer | Strip | partial | partial | yes | | Internal `StripStep` in `Normalizer/SentencePieceNormalizationStep.cs`; reached only through the Unigram factory. |
| normalizer | StripAccents | partial | partial | | | Same file, internal. |
| normalizer | NFC | partial | partial | | yes | Internal `UnicodeStep` (`string.Normalize`). No public NFC normalizer for the BPE path; a 10-line custom `Normalizer` would do it, subject to .NET's Unicode version. |
| normalizer | NFD, NFKC, NFKD | partial | partial | | | As NFC. |
| normalizer | Sequence | partial | partial | yes | | Internal `SequenceStep`; Unigram factory only. |
| normalizer | Lowercase | yes | partial | | | `Normalizer/LowerCaseNormalizer.cs` (public) and internal `LowercaseStep`. |
| normalizer | Nmt | partial | partial | | | Internal `NmtStep`. |
| normalizer | Precompiled | yes | partial | yes | | `Normalizer/SentencePieceNormalizer.cs` + `Utils/DoubleArrayTrie.cs`. Matched HF on 6/6 probe inputs with an `nmt_nfkc` charsmap and on the DeBERTa-v3 probe [run]. |
| normalizer | Replace | partial | partial | yes | | Internal `ReplaceStep`; uses .NET `Regex`, not Oniguruma. The Unigram factory recognizes `" {2,}"` and similar as whitespace collapse. |
| normalizer | Prepend | partial | partial | | | Internal `PrependStep`. |
| normalizer | ByteLevel | no | no | | | |
| pre_tokenizer | BertPreTokenizer | partial | no | | | Embedded in `BertTokenizer` basic tokenization; not a standalone stage. |
| pre_tokenizer | ByteLevel | partial | no | | yes | `BpeOptions.ByteLevel` + `Utils/ByteToUnicodeEncoding.cs`. `add_prefix_space` and `trim_offsets` not modeled for `BpeTokenizer`. Test-only JSON mapping. |
| pre_tokenizer | CharDelimiterSplit | no | no | | | |
| pre_tokenizer | Metaspace | partial | partial | yes | | Fused into the SentencePiece model (`addDummyPrefix`, `escapeWhiteSpaces`). Only U+2581. `prepend_scheme: "first"` is treated as `always`; `split` is not read. Divergence reproduced (section 3.4). |
| pre_tokenizer | Whitespace | partial | partial | | | `PreTokenizer.CreateWordOrNonWord`. The Unigram factory treats `Whitespace` as whitespace collapsing only and does not split punctuation. Divergence reproduced (section 3.4). |
| pre_tokenizer | Sequence | yes | partial | yes | yes | `PreTokenizer/CompositePreTokenizer.cs`; Unigram factory accepts only Metaspace/Whitespace/WhitespaceSplit members. |
| pre_tokenizer | Split | partial | no | | yes | `PreTokenizer/RegexPreTokenizer.cs` keeps matches and drops gaps, which equals `Split(Isolated, invert=false)` only when the pattern covers every character (true for the Qwen pattern). Other behaviors and `invert` absent. .NET `Regex` matches UTF-16 code units, so `\p{L}` and `\p{N}` do not match supplementary-plane characters. Divergence reproduced (section 3.4). |
| pre_tokenizer | Punctuation | no | no | (if `split_by_punct`) | | Unigram factory rejects it explicitly. DeBERTa-v3 default `split_by_punct=false` [code: transformers 4.57.6]. |
| pre_tokenizer | WhitespaceSplit | yes | partial | | | `PreTokenizer.CreateWhiteSpace` (`\S+`); Unigram factory maps it to `remove_extra_whitespaces`. |
| pre_tokenizer | Digits, UnicodeScripts, FixedLength | no | no | | | |
| model | BPE | partial | no | | yes | `Model/BPETokenizer.cs`. No `byte_fallback`, no `ignore_merges`. Construction from vocab/merges files or `BpeOptions`. |
| model | WordPiece | yes | no | | | `Model/WordPieceTokenizer.cs`, `vocab.txt` only. |
| model | WordLevel | no | no | | | |
| model | Unigram | yes | yes | yes | | `Model/SentencePieceUnigramModel.cs`, `Model/SentencePieceTokenizer.cs` `CreateFromTokenizerJson`. Runs of unknown characters are not fused (HF and SentencePiece emit one `unk`). Divergence reproduced (section 3.4). |
| post_processor | TemplateProcessing | partial | partial | yes | | Unigram factory: single-sequence prefix/suffix only; `pair` and type ids not applied; empty input returns no tokens where HF returns `[CLS] [SEP]`. |
| post_processor | BertProcessing | partial | partial | | | `BertTokenizer.BuildInputsWithSpecialTokens`/`CreateTokenTypeIdsFromSequences` in code; Unigram factory reads cls/sep only. |
| post_processor | RobertaProcessing | partial | partial | | | Unigram factory reads cls/sep; `trim_offsets`, `add_prefix_space`, pair `</s></s>` ignored. |
| post_processor | ByteLevel | no | no | | yes (offsets) | Unigram factory accepts and ignores it. Affects offsets only (`trim_offsets`). |
| post_processor | Sequence | partial | partial | | | Unigram factory only. |
| decoder | BPEDecoder | partial | no | | | `end_of_word_suffix` handling inside `BpeTokenizer.Decode`. |
| decoder | ByteLevel | yes | no | | yes | `BpeTokenizer.DecodeByteLevel`. |
| decoder | WordPiece | yes | no | | | Inside `WordPieceTokenizer`. |
| decoder | Metaspace | partial | no | yes | | Inside SentencePiece decode. The Unigram factory states "The `decoder` section is not read". |
| decoder | ByteFallback | partial | no | | | SentencePiece decode of `<0xNN>` pieces. |
| decoder | CTC, Sequence, Replace, Fuse, Strip | no | no | | | |
| top level | added_tokens | partial | partial | yes | yes | Special tokens as a `string -> id` map; matched by regex alternation. `lstrip`, `rstrip`, `single_word`, `normalized` not modeled. Unigram factory reads only `"special": true` entries; non-special added tokens are dropped. |
| top level | truncation | partial | no | yes | yes | `maxTokenCount` on encode (right side, no stride, no pair strategies). |
| top level | padding | no | no | yes | yes | `BertTokenizer` exposes a pad id only. |
| top level | version | | no | | | Unigram factory does not check `version`. |

Totals over 48 items (45 type tags plus `added_tokens`, `truncation`, `padding`):

| | yes | partial | no |
| --- | --- | --- | --- |
| Algorithm implemented | 9 | 25 | 14 |
| Read from `tokenizer.json` in production code | 1 (Unigram) | 21 (all via the one Unigram factory) | 26 |

Pilot slice:

| Profile | Components needed | Implemented (yes + partial) | Read from JSON |
| --- | --- | --- | --- |
| P1, DeBERTa-v3 / XLM-R Unigram (12 items: Sequence and Strip and Precompiled and Replace normalizers, Sequence and Metaspace pre-tokenizers, Unigram, TemplateProcessing, Metaspace decoder, added_tokens, truncation, padding) | 12 | 11 (padding missing) | 9 (decoder, truncation, padding missing) |
| P2, Qwen byte-level BPE (10 items: NFC, Sequence, Split, ByteLevel pre-tokenizer, BPE, ByteLevel post-processor, ByteLevel decoder, added_tokens, truncation, padding) | 10 | 8 (ByteLevel post-processor, padding missing) | 0 |

### 3.3 Architecture of the existing loader

`CreateFromTokenizerJson` is a Unigram-specific factory, not a pipeline interpreter. It reads the JSON, folds normalizer and pre-tokenizer settings into flags on one `SentencePieceUnigramModel` (`addDummyPrefix`, `escapeWhiteSpaces`, `removeExtraWhitespaces`, charsmap), and infers some behavior from shapes (a `Replace` with `" {2,}"` means "remove extra whitespace"). Approximate interpretation of `Whitespace` and `prepend_scheme: "first"` violates plan section 6.1 ("no approximate interpretation"). Type matching uses `OrdinalIgnoreCase`; the reference matches tags exactly. Its validation code is still useful as a field-by-field reference for the Unigram components.

### 3.4 Probe results

All probes ran HF `tokenizers==0.23.2` (and `transformers==4.57.6` for conversion) against `Microsoft.ML.Tokenizers 3.0.0-preview.26457.2` on .NET 10.0.112, with `SentencePieceTokenizer.CreateFromTokenizerJson` [run].

| Probe | Input | Result |
| --- | --- | --- |
| Vendored DeBERTa-v3-xsmall `tokenizer.json` plus GLiNER2's 10 added special tokens; 16 texts with and without special tokens, 16 GLiNER2-style single words | ASCII, accents, CJK, Cyrillic, Arabic, emoji with modifiers and flags, math alphanumerics, ligatures, tabs and CRLF, added tokens inline, combining mark at start, full-width | **45/48 match.** Differences: `"ꙮ unknown ꙮꙮ chars"` (HF `[...,507,3,18381,...]`, ML.NET `[...,507,3,3,18381,...]`), same without specials, and `""` (HF `[1,2]`, ML.NET `[]`). |
| `kyoto_ja_unigram_8000.model` converted with `DebertaV2Converter` | 6 texts incl. Japanese, half-width katakana, ideographic space, circled digits, ligature, emoji, supplementary CJK | 6/6 match. |
| `botchan_en_unigram_1000.model`, same converter | same 6 texts | 4/6 match; both differences are runs of unknown characters not fused. |
| Synthetic Unigram, `Sequence[Whitespace, Metaspace]` | `"hello,world"` | HF `[3,8,4]`, ML.NET `[3,7,6]`. |
| Synthetic Unigram, `Metaspace(prepend_scheme="first")` with added token `[P]` | `"[P]hello [P]world"` | HF `[21,5,9,21,6]`, ML.NET `[21,3,9,21,4]`. |
| Qwen3.5 pre-tokenizer regex, `Split(Isolated)` | `"x\U0001D400y"` | HF one piece `x𝐀y`; .NET `Regex.Matches` three pieces `x`, `𝐀`, `y`. Same divergence would apply to ML.NET's `RegexPreTokenizer` and Tiktoken path. |

Minimal reproducer for the unknown-run divergence: load the DeBERTa-v3-xsmall `tokenizer.json` (SHA-256 above) with `CreateFromTokenizerJson(stream, false, false)` and encode `"ꙮꙮ"`; HF `Tokenizer.from_file(...).encode("ꙮꙮ", add_special_tokens=False).ids` gives `[507, 3]` (`▁`, one `[UNK]`), ML.NET gives `[507, 3, 3]`. These are upstream contribution candidates under ADR-14.

## 4. Dependency and asset inventory

| Item | Finding |
| --- | --- |
| Target frameworks | `netstandard2.0;net8.0`. Helper split files `Utils/Helpers.netstandard.cs` / `Helpers.netcoreapp.cs`. A `net10.0`-only adoption drops the netstandard files. |
| NuGet dependencies | net8.0: none (nuspec group empty). netstandard2.0: `Microsoft.Bcl.HashCode` 6.0.0, `Microsoft.Bcl.Memory` 9.0.4, `System.Text.Json` 9.0.4, plus their transitive packages [code: nuspec]. On `net10.0` the adopted source needs no package references. |
| Project references | None at runtime. `src/Directory.Build.props` adds `tools-local/Microsoft.ML.InternalCodeAnalyzer` as a build-time analyzer (`ReferenceOutputAssembly=false`). Build uses the Arcade SDK. Neither is needed. |
| Unsafe code | `AllowUnsafeBlocks=true`; 31 `unsafe`/`fixed` sites. `SkipLocalsInit` module attribute in `AssemblyInfo.cs`. |
| Consumers inside ML.NET | `Microsoft.ML.TorchSharp` references the tokenizer project. Not relevant. |
| Embedded data in the core package | None. |
| Data packages | `Microsoft.ML.Tokenizers.Data.{Cl100kBase,Gpt2,O200kBase,P50kBase,R50kBase}` embed `.tiktoken` files downloaded from `openaipublic.blob.core.windows.net`, stated as MIT via openai/tiktoken. Compressed by `eng/TokenizerData.targets`. Not needed for the pilot; do not adopt. |
| Test assets | ML.NET tests use Hub `tokenizer.json` files (DeepSeek-R1, paraphrase-multilingual-MiniLM-L12-v2) and vocab files. Model-owner assets; do not copy. |
| FastBertTokenizer | `net10.0;net8.0;netstandard2.0`; `PolySharp` (build-time) and netstandard-only `System.Memory`, `System.Text.Json`. No embedded data. |

Proposed adoption set from ML.NET at `5a44dfdb` (not yet copied):

| Set | Files | Lines |
| --- | --- | --- |
| Unigram path | `Model/SentencePieceUnigramModel.cs`, `Model/SentencePieceBaseModel.cs`, `Model/SentencePieceTokenizer.cs` (loader logic as reference), `Normalizer/SentencePieceNormalizer.cs`, `Normalizer/SentencePieceNormalizationStep.cs`, `Utils/DoubleArrayTrie.cs`, `SentencepieceModel.cs` | 6,520 |
| BPE path | `Model/BPETokenizer.cs`, `Word.cs`, `Merge.cs`, `Pair.cs`, `Symbol.cs`, `Cache.cs`, `BpeOptions.cs`, `Utils/LruCache.cs`, `Utils/PriorityQueue.cs`, `Utils/ByteToUnicodeEncoding.cs`, `PreTokenizer/RegexPreTokenizer.cs`, `CompositePreTokenizer.cs`, `PreTokenizer.cs` | 3,067 |
| Shared utilities as needed | `Utils/Helpers.cs`, `ValueStringBuilder.cs`, `StringSpanOrdinalKey.cs` and similar | about 1,000 |

Do not adopt: the public `Tokenizer` base API, Tiktoken, CodeGen, Phi-2, Llama, EnglishRoberta, SentencePiece BPE, the Data packages. The owned public API comes from the interpreter design, not from ML.NET's class shapes.

## 5. Notices required

Apply to whatever is adopted, in the same change, in `tokenizers-dotnet/THIRD-PARTY-NOTICES.md` (verbatim license text) and one `PROVENANCE.md` row per file:

| Notice | Applies to | Source of text |
| --- | --- | --- |
| MIT, "Copyright (c) .NET Foundation and Contributors" | Every adopted ML.NET file. Keep the existing three-line header; it points at the ML.NET LICENSE, which must be reproduced. | `LICENSE` at `5a44dfdb` |
| BSD 2-Clause, "Copyright (c) 2008-2014, Susumu Yata" (darts-clone) | `Utils/DoubleArrayTrie.cs` and anything derived from it. Requires the notice in source and in binary distribution documentation. | ML.NET `THIRD-PARTY-NOTICES.TXT` "License notice for DoubleArrayTrie (DART)" |
| Apache-2.0, "Copyright 2016 Google Inc." (SentencePiece) | SentencePiece-derived model, normalizer, and protobuf reader files. | ML.NET `THIRD-PARTY-NOTICES.TXT` "License notice for SentencePiece" |
| Apache-2.0, HuggingFace Tokenizers | BPE model logic, and any component ported from the Rust source (Split behaviors, added-vocabulary matching, TemplateProcessing, alignment tracking). | ML.NET `THIRD-PARTY-NOTICES.TXT` and `huggingface/tokenizers` `LICENSE` |
| Apache-2.0, transformers (`tokenization_bert.py` at `8e3e145b427196e014f37aa42ba890b9bc94275e`) | Only if WordPiece/BERT files are adopted. | ML.NET `THIRD-PARTY-NOTICES.TXT` |
| MIT, FastBertTokenizer | Only if its source is adopted. | Its `LICENSE` |

Apache-2.0 section 4(b) requires modified files to carry a prominent change notice: keep the original header and add a line stating the file was modified by this project. Neither huggingface/tokenizers nor google/sentencepiece has a `NOTICE` file at the examined revisions [code], so section 4(d) adds nothing. The tiktoken and CodeGen notices are not needed unless those parts are adopted.

## 6. Other .NET seeds

| Source | Scope | Fit |
| --- | --- | --- |
| FastBertTokenizer `31f7cb32`, MIT, 2,150 lines | WordPiece with fused BertNormalizer/BertPreTokenizer, batch and parallel APIs, decode. `LoadTokenizerJson` accepts only `version` 1.0, `WordPiece`, `BertNormalizer` with `clean_text` and `handle_chinese_chars` true, `BertPreTokenizer`, post-processor with `[CLS]`/`[SEP]`; rejects `single_word` added tokens; hard-codes `[PAD]`. | Not needed for the pilot (both profiles are Unigram and BPE). Reasonable seed or comparator if a WordPiece profile is admitted later; ML.NET's WordPiece is the other candidate. Correctness was not compared. |
| SharpToken, TiktokenSharp, Tokenizers.DotNet and other NuGet tokenizers | Not examined: NuGet search (azuresearch) was blocked. Tokenizers.DotNet is reported to wrap the native Rust library, which would not meet the managed-path rule [unverified]. | |
| huggingface/tokenizers `v0.23.2` Rust, Apache-2.0 | Reference semantics for every component; port source for components ML.NET lacks. | Primary behavioral spec. |

## 7. Pilot profiles

### P1: GLiNER2 backbone and P0 encoder

GLiNER2 at `55656fbf` loads tokenizers with `AutoTokenizer.from_pretrained` (`gliner2/models/base.py`, `gliner2/processor.py`), adds 10 special tokens, and tokenizes each word separately with `tokenizer.tokenize` before `convert_tokens_to_ids`; the post-processor is not used on that path [code]. The README model table names DeBERTa-v3 (English) and mDeBERTa-v3 (multilingual) backbones, and the repository's real-encoder tests use `microsoft/deberta-v3-xsmall` [code]. The checkpoints' own `tokenizer.json` files were not read [unverified]. `docs/m0/encoder-selection.md` covers this in detail and proposes `intfloat/multilingual-e5-small` (XLM-R Unigram) as the P0 encoder.

DeBERTa-v3 `tokenizer.json` shape (vendored file, matches `DebertaV2Converter` in transformers 4.57.6 and HEAD): normalizer `Sequence[Strip, Precompiled, Replace(" {2,}" -> " ")]`, pre-tokenizer `Sequence[Metaspace(prepend_scheme="always", split=true)]`, `Unigram` (`unk_id` 3, `byte_fallback` false), `TemplateProcessing` `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B [SEP]`, `Metaspace` decoder.

### P2: Qwen byte-level BPE

Shape from transformers source, not from a Hub file [code; Hub content unverified]: `Qwen2Converter` (identical in 4.57.6 and HEAD) and `Qwen3_5Tokenizer` (`models/qwen3_5/tokenization_qwen3_5.py`, HEAD only): `BPE` (no unk, empty prefix/suffix, `fuse_unk` false, `byte_fallback` false), normalizer `NFC`, pre-tokenizer `Sequence[Split(regex, Isolated, invert=false), ByteLevel(add_prefix_space=false, use_regex=false)]`, decoder `ByteLevel`, post-processor `ByteLevel(trim_offsets=false)` (Qwen2 converter; not set in the Qwen3.5 class). Qwen3.5 changes the regex to add `\p{M}`: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. Vocabulary size, `ignore_merges`, and the added-token list (special and non-special) must be read from the pinned asset.

## 8. Recommendation

1. **Adopt selected Microsoft.ML.Tokenizers source at `5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d`** (equal in tokenizer code to the `3.0.0-preview.26457.2` package): the Unigram path and the BPE merge path listed in section 4, under the `Starling.Tokenizers` namespace, `net10.0` only, no package references. It is MIT, already managed and allocation-conscious, and on the real DeBERTa-v3 asset it matched HF on 45 of 48 probes. Porting the Unigram lattice and the precompiled charsmap from Rust instead would add an estimated 10 to 16 maintainer hours for no fidelity gain.
2. **Write the interpreter fresh**, using huggingface/tokenizers `v0.23.2` (`88a4498a`) as the behavioral spec and Apache-2.0 port source where ML.NET has nothing. Do not adopt `CreateFromTokenizerJson` as the architecture: it fuses stages into one model and approximates `Whitespace` and `prepend_scheme: "first"`. Each `type` tag gets its own stage class; any unknown tag, unknown field value, or unimplemented option fails with an explicit error naming the tag. Use the ML.NET factory only as a field-validation reference.
3. **Target format `version: "1.0"`** and reject `"2.0"` explicitly until a pinned asset needs it. Record the upstream 1.0 rewrite as a tracked risk (section 10).
4. **Fix the reproduced divergences in the owned copy** (unknown-run fusion, empty-input template, `Split` on supplementary-plane characters) and open upstream issues or PRs through the contribution track (P-026). Record each as a local change in `PROVENANCE.md`.
5. **Scope the first interpreter release to the 18 distinct component items in section 3.2's pilot slice** (12 for P1 and 10 for P2, of which the Sequence pre-tokenizer, added_tokens, truncation and padding are shared). Mark all other tags `unsupported`. Do not adopt FastBertTokenizer or any WordPiece source now.
6. Keep Microsoft.ML.Tokenizers as a comparator in test projects only, as the plan requires.

## 9. Effort estimate

Maintainer hours, meaning design, review, debugging and evidence checks; agents write most code.

| Work | Hours |
| --- | --- |
| Adoption mechanics: copy the section 4 file sets at the pinned SHA, rename namespaces, drop netstandard code, trim public surface, headers and change notices, `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md` rows, run `tools/check-tokenizer-independence.sh` | 6 to 10 |
| Interpreter core: strict 1.0 reader and component registry, explicit unsupported errors, pipeline stages, normalized-string alignment (offsets back to original text, UTF-16 with code point conversion), added-token matcher (`special`, `normalized`, `lstrip`, `rstrip`, `single_word`), TemplateProcessing single and pair with type ids, truncation and padding subset, decoder chain | 18 to 30 |
| P1 components: split Metaspace out of the SentencePiece model as a pre-tokenizer stage (`prepend_scheme`, `split`), wire Strip/Precompiled/Replace/Sequence as stages, Unigram as a model stage, unknown-run fusion, empty-input template, Metaspace decoder, `.NET Regex` versus Oniguruma check for `Replace` | 10 to 18 |
| P2 components: BPE model stage with `byte_level`, check `ignore_merges` against the asset, NFC, `Split` with a code-point-aware matcher for the Qwen pattern, ByteLevel pre-tokenizer options, ByteLevel decoder with incremental UTF-8 decode, ByteLevel post-processor offsets, non-special added tokens | 14 to 26 |
| Reference-runner fixtures and conformance wiring for both profiles (not the protected-set curation, which is P-030) | 6 to 10 |
| **Subtotal, adoption plus interpreter** | **54 to 94** |
| Standalone packaging and release-independence proof (plan 6.3), outside this audit's scope | 8 to 16 (rough) |
| **Total against the plan's 60 to 120 hour allowance** | **62 to 110** |

Assumptions:

- One owned format version (1.0) and only the 18 distinct pilot items; everything else stays `unsupported`.
- Reference behavior is HF tokenizers 0.x via transformers 4.x, matching GLiNER2's `transformers<5` pin. A move to the 2.0 format adds work not counted here.
- The Qwen pre-tokenizer regex can be matched code-point-correctly either by a hand-written matcher or a rewritten .NET pattern; if neither passes conformance, add 6 to 10 hours.
- `.NET` Unicode data (NFC, `\p{L}`) agrees with the reference for the conformance corpus; any disagreement found is handled per case and not counted.
- No mDeBERTa-v3 specific behavior beyond the XLM-R/DeBERTa shape (not inspected).
- Agents produce first drafts; hours are review and debugging time at the plan's 20 to 30 hours per week.
- Hub assets for the pinned P0 encoder, GLiNER2 checkpoint and Qwen model are available to someone with Hub access.

## 10. Open questions

1. Upstream format change: tokenizers 1.0 writes `"version": "2.0"` and its rc wheels refused 1.0-format DeBERTa-style files in this session. When will transformers default to tokenizers 1.0, and will Hub assets be rewritten? Proposed: support 1.0 only; track 1.0 final and add a 2.0 reader as a separate backlog item when a pinned asset uses it.
2. Offsets: ML.NET reports offsets in the normalized text. The plan requires a declared convention and conversion to C# string positions. Which convention does the owned API expose by default (UTF-16 into the original string, with code-point conversion as an explicit method)?
3. Qwen3.5 asset: exact model ID and revision, whether its `tokenizer.json` matches the transformers class (regex with `\p{M}`, post-processor present or absent), `ignore_merges`, and the full added-token list. Not readable here.
4. GLiNER2 checkpoint `tokenizer.json`: confirm it matches the vendored DeBERTa-v3 file plus the 10 added tokens, and record their IDs.
5. Should the owned Unigram profile reproduce HF's fused unknown output (the reference) and record ML.NET's per-character output as a comparator difference? Proposed: yes; HF is the fidelity reference.
6. Is .NET `Regex` acceptable for `Replace` patterns in pinned assets, given the reference uses Oniguruma? Proposed: allow only patterns on an explicit allow-list that the conformance suite covers; reject others.
7. Upstream reporting: whether to file the five divergences in section 3.4 (unknown-run fusion, empty-input template, `Whitespace`, `prepend_scheme: "first"`, supplementary-plane regex) against dotnet/machinelearning now (ADR-14) or after the owned fixes land.
