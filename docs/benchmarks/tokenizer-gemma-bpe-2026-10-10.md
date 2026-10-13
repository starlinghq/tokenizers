# Owned tokenizer, Gemma BPE profile: load and encode, 2026-10-10

> **Historical record.** Written in the Starling planning repository before this repository was split out. The package then had a planning-era name (it is `Starlings.Tokenizers` throughout here); paths such as `tokenizers-dotnet/`, `model-pipeline/` and `tools/*-tokenizer-*` refer to that repository's layout (see [docs/repository-history.md](../repository-history.md)).

Exploratory. `Starlings.Tokenizers` 0.1.0-preview.4 (this change) against the Rust `tokenizers` 0.23.3 (Python binding, the version the EmbeddingGemma 2 fixtures were generated with) on the `gemma-bpe-embeddinggemma2` asset. One workstation, not a controlled machine: Intel i7-5820K (6 cores), Linux x64 7.2, .NET 10.0.12 (SDK 10.0.100), Release builds, asset in the OS page cache. **Background load average about 8 during every series** (a local LLM server and other agents' test runs); absolute numbers are inflated by it. The C# and Rust series ran back to back, so the ratios are the more useful figure.

Asset: `google/embeddinggemma-2@914f7f89` `tokenizer.json`, 32,170,510 bytes, `4d777ef5…`: 262,144 vocabulary entries (256 byte tokens), 514,906 merges in pair form, 24 added tokens.

| Measure | Owned (C#) | Rust tokenizers 0.23.3 | Ratio |
| --- | ---: | ---: | ---: |
| cold load, ms (median of 15; min to max) | **542** (524–603) | 1,662 (1,642–1,705) | 3.1x |
| warm reload, ms (median of 10) | 347 (226–415) | | |
| encode, ms per round (median of 20) | **325** (310–351) | 445 (432–452) | 1.4x |

Other cold-process figures: 110 MiB allocated in the whole process, 144 MiB peak working set, 78.8 MiB managed heap after load with the tokenizer alive (the Qwen assets: 45.7 MiB; the Gemma vocabulary and merge table are 1.06 and 2.08 times larger). The load uses the same byte-scanning fast path as the Qwen profiles (`BpeLoaderTests.GemmaAssetLoadsIdenticallyOnBothPaths`); nothing in it assumes a byte-level vocabulary.

Encode is closer to the reference than for the Qwen profiles (2.5x there) because the Gemma pipeline has no effective pre-tokenization: the `Split(" ")` never fires after `Replace(" " -> "▁")`, so the 3,800-character paragraph in the round is one BPE word of 3,800 symbols, merged with a heap as upstream does (cost O(n log n) in the segment length). Upstream's word cache does not help there either (it caches words shorter than 256 bytes).

Method (as in [tokenizer-qwen-bpe-2026-10-10.md](tokenizer-qwen-bpe-2026-10-10.md)):

- Cold: `LoadBench cold <tokenizer.json> 15`, a fresh process per run, `load_ms` = `Tokenizer.FromFile` inside the process (JIT included, runtime start-up excluded). Rust: a fresh Python interpreter per run timing `Tokenizer.from_file` after the import, 15 runs.
- Encode: `LoadBench encode <tokenizer.json> 20`: 200 x 5 texts per round (short English query, English passage, German/French, CJK/Korean, a 3,800-character English paragraph), 3 warm-up rounds. Rust: the same texts and loop through `Tokenizer.encode(text).ids` in Python, so the binding's per-call overhead is in the Rust figure. Id checksums equal (169055653400 for 20 rounds in both).

```sh
B=benchmarks/Starlings.Tokenizers.LoadBench/bin/Release/net10.0/Starlings.Tokenizers.LoadBench
A="$STARLING_MODELS/google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json"
$B cold "$A" 15
$B warm "$A" 10
$B encode "$A" 20
# Rust, in the gemma-bpe reference environment (model-pipeline/reference/gemma-bpe/requirements.txt): the cold
# and encode scripts of tokenizer-qwen-bpe-2026-10-10.md with this asset.
```
