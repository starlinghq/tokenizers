# Owned tokenizer load time, 2026-10-09

> **Historical record.** Written in the Starling planning repository before this repository was split out. The package then had a planning-era name (it is `Starling.Tokenizers` throughout here); paths such as `tokenizers-dotnet/`, `model-pipeline/` and `tools/*-tokenizer-*` refer to that repository's layout (see [docs/repository-history.md](../repository-history.md)).

`Tokenizer.FromFile` in `Starling.Tokenizers`, 0.1.0-preview.1 (main at `ff55265`) against 0.1.0-preview.2 (this change). Follow-up to decision 1 of [e5-2026-10-09.md](e5-2026-10-09.md): the owned tokenizer's load made a C# process reach its first e5 embedding 1.7x later than Python.

One workstation, not a controlled lab machine: Intel i7-5820K (6 cores, AVX2), Linux x64 7.2, .NET 10.0.12 (SDK 10.0.100), Release builds, assets in the OS page cache. Load average 0.7 to 1.8 during every run (checked before each series; the script waits while it is above 4); the user's desktop, Chrome and a local ClickHouse server were running. Exploratory evidence, reproducible with the commands below.

Assets: e5 `tokenizer.json` (`intfloat/multilingual-e5-small@614241f6`, 17,082,730 bytes, 250,002 Unigram pieces, SHA-256 `0b44a9d7…4c39`) and GLiNER2 `tokenizer.json` (`fastino/gliner2-base-v1@f9634218`, 8,650,701 bytes, 128,000 pieces, `1b7fbabf…9c16`).

## Results

`benchmarks/Starling.Tokenizers.LoadBench`. **Cold:** a fresh process per run; `load_ms` is `Tokenizer.FromFile` (file read included) timed inside the process, so JIT of the loader is included and runtime start-up is not; `wall` is spawn to exit (start-up, load, one encode, exit). **Warm:** reloads in one process after one warm-up load. Allocations are `GC.GetTotalAllocatedBytes` for the whole cold process; "heap after load" is `GC.GetTotalMemory(forceFullCollection: true)` with the tokenizer alive. Median, with min to max and the interquartile range; two interleaved rounds of 10 cold runs each for e5 (20 per version), one round of 10 for GLiNER2.

| Asset | Measure | 0.1.0-preview.1 | 0.1.0-preview.2 | Ratio |
| --- | --- | ---: | ---: | ---: |
| e5 | cold load, ms | 1,285 (1,252–1,330; IQR 1,268–1,300) | **270** (267–292; IQR 269–275) | 4.8x |
| e5 | cold process wall, ms | 1,432 (1,401–1,502) | 398 (384–457) | 3.6x |
| e5 | warm reload, ms | 523 (500–814) | 171 (157–220) | 3.1x |
| e5 | allocated, whole cold process, MiB | 128.4 | 48.7 | |
| e5 | allocated per warm load, MiB | 80.3 | 48.1 | |
| e5 | peak working set, MiB | 143 | 85 | |
| e5 | managed heap after load, MiB | 72.0 | 12.0 | |
| GLiNER2 | cold load, ms | 748 (736–814) | **174** (166–183) | 4.3x |
| GLiNER2 | warm reload, ms | 224 (217–459) | 77 (72–93) | 2.9x |
| GLiNER2 | managed heap after load, MiB | 36.5 | 6.3 | |

Rust reference on the same machine, same e5 file: Python `tokenizers` 0.23.2 (`/home/mark/Projects/AI-models/e5/venv-bench-cpu`), `Tokenizer.from_file` in a fresh interpreter, 15 runs: **501 ms** median (492–547), import 16 ms. The owned tokenizer now loads in about 54% of the Rust tokenizer's time.

Encoding is not slower: `LoadBench encode` (200 x 5 texts per round, 20 rounds after 3 warm-up rounds, two interleaved pairs): 0.1.0-preview.1 183.6 and 181.7 ms per round, 0.1.0-preview.2 175.3 and 178.8 ms; identical id checksums.

## Where the time went (e5, cold, one process, Stopwatch spans)

Temporary `Stopwatch` spans in the loader (not committed); a `dotnet-trace` CPU sample of the cold process confirmed the split.

| Phase | preview.1 ms | preview.2 ms | Change |
| --- | ---: | ---: | --- |
| File read | about 8 | 8 | |
| JSON | 166 (`JsonDocument` of the whole file) | 78 (vocab bytes scanned directly) + 3 (`JsonDocument` of the rest, 0.3 MB) | Streaming read of `model.vocab`; serde_json-compatible score parsing unchanged |
| Vocab entries to strings and doubles | 200 | (in the scan) | Pieces stay UTF-8 in one buffer; no per-piece string |
| Normalizer (charsmap base64 and trie), pre-tokenizer | 11 | 9 | |
| Piece-to-id dictionary | 33 | 0 | The trie is the map (exact match) |
| UTF-8 keys and sort | 225 (`List.Sort` with a comparison delegate) | 6 (checks) + 28 (radix sort on 8-byte prefixes, multikey quicksort for ties) + 6 (dedupe and copy) | |
| Trie build | 601 (DAWG 338, double array 263) | 111 (darts-clone keyset build, no DAWG) | Same lookups, about 1.03 M units |
| Added tokens; post-processor, decoder, truncation, padding | 6; not split | 7; 8 | |
| **Total** | **about 1,265** | **about 275** | |

JIT was a large share before: the same preview.1 code took 523 ms warm against 1,285 ms cold, because the hot loops start in tier-0 code (and on-stack replacement) during a single load. The new load-only loops are marked `MethodImplOptions.AggressiveOptimization`, so they are compiled fully optimized at first call. This is a code attribute, not a packaging option; encode paths are not marked and keep tiered PGO.

## Runtime knobs (notes, not adopted)

Environment variables on the cold runs, e5, 10 runs each (load_ms median):

| Knob | preview.1 | preview.2 |
| --- | ---: | ---: |
| none | 1,288 | 278 |
| `DOTNET_TieredPGO=0` | 1,170 | 254 |
| `DOTNET_TieredCompilation=0` | 746 | 300 |
| `DOTNET_TC_QuickJitForLoops=0` | 894 | 292 |

These are application settings, not library ones, and none helps preview.2 by more than about 10%, so the library recommends none. ReadyToRun compilation of the tokenizer package was not tried: it would make the package RID-specific or need composite images, and the algorithmic changes already meet the target. A precompiled binary cache of the vocabulary and trie (optional, versioned, keyed by the `tokenizer.json` hash) was not built either; at 0.27 s it is not needed for the target and would add a second load path to qualify.

## End to end: e5 cold start (workload harness)

`ai-dotnet/benchmarks` harness, `cold` mode as in [e5-2026-10-09.md](e5-2026-10-09.md) (spawn to first embedding of one query, CPU, 6 intra-op threads), three interleaved rounds of 5 fresh processes per language (15 each), same machine, 2026-10-09 23:39 UTC. Not the full `run-e5-matched.sh` protocol (no warm series, fewer processes); raw JSONL not committed.

| | C# median ms (95% CI) | Python median ms (95% CI) | C#/Python |
| --- | ---: | ---: | --- |
| Before (P-008 run, 30 processes each) | 2,047 (2,026–2,064) | 1,185 (1,165–1,228) | 1.73 |
| After (this run, 15 each) | **944** (933–958) | 1,181 (1,163–1,218) | **0.80** (0.78–0.82) |

C# stage medians: spawn to Main 25 ms, tokenizer load 243 ms (was 1,332), session create 544 ms, first tokenize 34 ms, first encode 19 ms. Python: tokenizer load 532 ms, session create 442 ms, imports 114 ms. The C# cold start is now about 20% shorter than Python's; ORT session creation is the largest remaining stage (100 ms slower than Python's, unchanged).

Caveat found while measuring: building `ai-dotnet/AI.slnx` builds the tokenizer project (which is not in that solution) with its default Debug configuration, and copies that Debug DLL into the ai-dotnet outputs. With a Debug DLL the JIT ignores `AggressiveOptimization`, and the harness measured 908 ms for the tokenizer load. `run-e5-matched.sh` builds the harness project directly in Release, which is what the numbers above use; packages are packed in Release. Builds of `AI.slnx` (CI, local test runs) therefore exercise a Debug tokenizer; behavior is the same, timings are not.

## Reproduce

```sh
export DOTNET_ROOT=~/.dotnet PATH=~/.dotnet:$PATH
dotnet build -c Release benchmarks/Starling.Tokenizers.LoadBench
B=benchmarks/Starling.Tokenizers.LoadBench/bin/Release/net10.0/Starling.Tokenizers.LoadBench
$B cold  "$STARLING_MODELS/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json" 15
$B warm  "$STARLING_MODELS/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json" 10
$B encode "$STARLING_MODELS/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json" 20
```

The "before" column was built from `git archive ff55265 tokenizers-dotnet` with the same benchmark program copied in. Harness: `dotnet build -c Release` of the planning repository's e5 benchmark harness, then `E5Harness cold --provider cpu --threads 6 --inter-threads 1 --impl library --processes 5 --out cold.jsonl` alternating with `bench_e5_matched.py cold` (same arguments), and `model-pipeline bench summarize cold.jsonl`.
