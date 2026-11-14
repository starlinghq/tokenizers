# Performance

All numbers are exploratory: measured on shared machines, not on a controlled benchmark machine. The dated records under [benchmarks/](benchmarks/) give the method, the raw output and the machine load at the time.

## Cold start against Python `tokenizers` (2026-11-14)

The current comparison, for all 16 pinned assets: a fresh .NET process running `Tokenizer.FromFile` against a fresh Python interpreter running `tokenizers` 0.23.3 `Tokenizer.from_file`. It was measured on a RunPod CPU pod (AMD EPYC 9575F, both sides pinned to one physical core), with library commit `5951cbe` and .NET 10.0.15. Medians of three interleaved rounds of 10 processes:

| Asset | Size | .NET load | Python load | .NET / Python |
| --- | ---: | ---: | ---: | ---: |
| `google/embeddinggemma-2` (`gemma-bpe-embeddinggemma2`) | 32 MB | 222 ms | 636 ms | 0.35 |
| `Qwen/Qwen3.5-0.8B` (`qwen35-bpe-chat`) | 20 MB | 125 ms | 398 ms | 0.31 |
| `intfloat/multilingual-e5-small` (`xlmr-unigram-e5`) | 17 MB | 129 ms | 379 ms | 0.34 |
| `microsoft/Phi-4-mini-instruct` (`phi4-bpe-chat`) | 16 MB | 106 ms | 276 ms | 0.39 |
| `openai/whisper-large-v3-turbo` (`whisper-bpe`) | 2.7 MB | 45 ms | 33 ms | 1.39 |
| `sentence-transformers/all-MiniLM-L6-v2` (WordPiece) | 0.5 MB | 37 ms | 9 ms | 4.13 |

Files of 9 MB and more load in 0.31 to 0.58 of Python's time, and the process exits 1.8 to 3.4 times sooner. Files under 1 MB load in about 37 ms against Python's 9 ms, because the first call compiles the .NET loader (JIT), a fixed cost the native library does not pay. The two are about even near 3 MB. All 16 rows, process wall times, memory, and why the pod had to be pinned: [docs/benchmarks/tokenizer-coldstart-2026-11-14.md](benchmarks/tokenizer-coldstart-2026-11-14.md).

## Load time on the workstation (2026-10)

The sections below were measured on the maintainer's workstation (i7-5820K, Linux x64) in October 2026, against `tokenizers` 0.23.2 and 0.23.3. Absolute times are not comparable with the pod above; the ratios agree with it.

`Tokenizer.FromFile` on the e5 `tokenizer.json` (17 MB, 250,002 Unigram pieces) takes about 0.27 s in a fresh process on the workstation above, down from 1.29 s in 0.1.0-preview.1; the Rust `tokenizers` 0.23.2 `Tokenizer.from_file` takes about 0.50 s on the same machine. Measurements and method: [docs/benchmarks/tokenizer-load-2026-10-09.md](benchmarks/tokenizer-load-2026-10-09.md).

How it is fast (since 0.1.0-preview.2): the `model.vocab` array is read straight from the UTF-8 bytes and only the rest of the file goes through `JsonDocument`; pieces stay UTF-8 (strings are made on demand); the double-array trie is built from radix-sorted keys with darts-clone's keyset builder instead of the DAWG pass; load-only hot loops are compiled fully optimized at first call (`AggressiveOptimization`) instead of starting in tier-0 code. The fast path only accepts input the document path reads to the same values; anything else (malformed entries, duplicate keys, invalid JSON) falls back to the unchanged document path, so errors and strictness are as before. `LoadFastPathTests` compares the two paths on both pinned assets and on more than 50 edge-case documents.

Gemma BPE profile, exploratory ([docs/benchmarks/tokenizer-gemma-bpe-2026-10-10.md](benchmarks/tokenizer-gemma-bpe-2026-10-10.md)): cold `Tokenizer.FromFile` of the 32 MB `gemma-bpe-embeddinggemma2` file (262,144 tokens, 514,906 merges) about 0.54 s against about 1.66 s for the Rust `tokenizers` 0.23.3 `Tokenizer.from_file` right after (background load average about 8); 78.8 MiB managed heap after load; encode 1.4x the reference's speed on the same texts (no effective pre-tokenization, so long texts are long BPE words).

Qwen BPE profiles, exploratory (same machine, 2026-10-10, under background load average 5 to 8 from other services; [docs/benchmarks/tokenizer-qwen-bpe-2026-10-10.md](benchmarks/tokenizer-qwen-bpe-2026-10-10.md)): cold `Tokenizer.FromFile` of the 20 MB Qwen `tokenizer.json` (248,044 tokens, 247,587 merges) about 0.31 s for `qwen2-bpe-kev` and 0.33 s for `qwen35-bpe-chat`, against about 0.96 s for the Rust `tokenizers` 0.23.2 `Tokenizer.from_file` measured right after; the BPE `vocab` object and `merges` array are scanned from the bytes like the Unigram vocabulary (same fast-path and fallback rules, `BpeLoaderTests`).


## Running the benchmark

`benchmarks/Starlings.Tokenizers.LoadBench` measures `Tokenizer.FromFile` in fresh processes (`cold`) and repeated in one process (`warm`), with GC allocated bytes and peak working set. CI builds it but never runs it.

```sh
tools/fetch-assets.sh
dotnet run -c Release --project benchmarks/Starlings.Tokenizers.LoadBench -- cold .models/<repo>/<commit>/tokenizer.json 15
dotnet run -c Release --project benchmarks/Starlings.Tokenizers.LoadBench -- warm .models/<repo>/<commit>/tokenizer.json 10
```

The Python reference is `benchmarks/python-cold-load.py <tokenizer.json> [runs]` (needs `pip install tokenizers`): it runs a fresh interpreter per run, and its output matches `LoadBench cold`, including the ids. Alternate the two on the same file. On a shared or rented machine, pin both to the same physical core (`taskset`), as the 2026-11-14 record explains.

Record a new measurement as a dated file under `docs/benchmarks/` with the commit, SDK and runtime versions, the machine and its load, and the reference measured right after on the same machine.
