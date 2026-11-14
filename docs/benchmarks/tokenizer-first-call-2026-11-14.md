# Where the cold-start time goes: JIT, ReadyToRun and Native AOT, 2026-11-14

Follow-up to [tokenizer-coldstart-2026-11-14.md](tokenizer-coldstart-2026-11-14.md), which found that Python `tokenizers` loads the small (under 1 MB) WordPiece files in about 9 ms against about 37 ms for `Tokenizer.FromFile`, while .NET is about three times faster on the large ones. This record measures where the small-file time goes, with BenchmarkDotNet, and what removes it.

## Setup

| | |
| --- | --- |
| Library | `perf/benchmarkdotnet` at `153fc45` (BenchmarkDotNet runs) and `f44380e` (JIT, ReadyToRun and Native AOT runs; only LoadBench's output code differs). Both have `main` at `5951cbe` underneath, the 0.3.5 library code. |
| .NET | SDK 10.0.115, runtime 10.0.15, BenchmarkDotNet 0.15.8 |
| Reference | Python 3.12.3, `tokenizers` 0.23.3 |
| Machine | RunPod CPU pod, `cpu5c`, 4 vCPU, 8 GB, EUR-IS-1, AMD EPYC 9575F (shared host). Cpuset: two physical cores. A probe (9 cold MiniLM loads per core) gave 43 ms on cores 2,66 and 54 ms on cores 46,110, so every process ran under `taskset -c 2,66`. |
| Host load | The host's load average rose from about 50 to 77 during the last part of the run. Compare numbers within one table, not across tables. |

## BenchmarkDotNet: cold against warm

`benchmarks/Starlings.Tokenizers.Benchmarks`. **Cold** is `RunStrategy.ColdStart` with 20 launches: one call in each of 20 fresh processes, so the measured call includes compiling the code it runs. **Warm** is the steady state after 5 warm-up iterations, over 15 iterations. The file is in the page cache in both cases (read in `GlobalSetup`).

| Asset | Size | `FromFile` cold | `FromFile` warm | First-call cost | First `Encode` cold | First `Encode` warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `minilm` (all-MiniLM-L6-v2, WordPiece) | 0.5 MB | 30.1 ± 0.4 ms | **3.2 ms** | 27 ms | 24.5 ms | 8.1 µs |
| `whisper` (whisper-bpe) | 2.7 MB | 39.2 ± 0.4 ms | 8.0 ms | 31 ms | 16.2 ms | 4.0 µs |
| `qwen35` (qwen35-bpe-chat) | 20 MB | 102.9 ± 2.5 ms | 74.2 ms | 29 ms | 16.8 ms | 3.3 µs |
| `e5` (xlmr-unigram-e5) | 17 MB | 132.6 ± 8.0 ms | 80.9 ms | 52 ms | 17.5 ms | 5.8 µs |
| `gemma` (gemma-bpe-embeddinggemma2) | 32 MB | 223.3 ± 8.7 ms | 155.3 ms | 68 ms | 13.0 ms | 3.5 µs |

(± is BenchmarkDotNet's standard deviation over the 20 launches.)

- **The loader itself is fast.** Once compiled, `FromFile` takes 3.2 ms for MiniLM, a third of Python's 9 ms. The parsing is not what makes small files slow.
- **The first call costs 27 to 31 ms whatever the file** (more on e5 and Gemma, whose loaders run more distinct code: the Unigram trie, byte fallback). In one cold MiniLM load (LoadBench at `f44380e`) the JIT compiles 519 methods, 334 of them library code or generic instantiations over library types, almost all at Tier0 (`DOTNET_JitDisasmSummary=1`).
- **The first `Encode` also costs 13 to 25 ms**, against microseconds afterwards. Python's first encode is 0.2 to 0.6 ms. A real cold start pays both.

## JIT, ReadyToRun and Native AOT, fresh processes

The same LoadBench program published three ways: the default build (JIT), `dotnet publish -p:PublishReadyToRun=true` (the app and the library precompiled, JIT for what crossgen cannot precompile), and `dotnet publish -p:PublishAot=true` (Native AOT, no JIT). 20 fresh processes each, interleaved. `load` is `Tokenizer.FromFile` timed inside the process.

| Asset | JIT load | ReadyToRun load | **Native AOT load** | JIT first encode | ReadyToRun first encode | **Native AOT first encode** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `minilm` | 40.8 ms | 29.7 ms | **5.3 ms** | 26.1 ms | 12.8 ms | **2.1 ms** |
| `whisper` | 50.5 ms | 40.8 ms | **11.8 ms** | 16.2 ms | 10.8 ms | **0.33 ms** |
| `e5` | 138.6 ms | 122.5 ms | **92.5 ms** | 15.4 ms | 7.1 ms | **0.10 ms** |

Python `tokenizers` on the same pod, earlier in the run, 20 fresh interpreters, pinned the same way: MiniLM 13.1 ms load and 0.6 ms first encode. Whisper and e5 were measured under the heavier host load (69 and 691 ms) and are not comparable with this table.

- **Native AOT removes the small-file gap.** MiniLM loads in 5.3 ms, faster than Python. The library already builds and runs under Native AOT (BenchmarkDotNet's `--runtimes nativeaot10.0` run and the published LoadBench), with no trimming or AOT warnings from its code. The ids were not compared against the JIT build in this run; the conformance tests do not run under AOT. Its steady-state speed under AOT matches tiered JIT code (MiniLM 3.7 against 3.3 ms; e5 78 against 80 ms).
- **ReadyToRun removes about a third.** 281 methods are still compiled at run time. Almost all are framework generics instantiated over the library's own structs (`List<PatternMatch>`, `List<CharChange>`, `List<Alignment>`, `Array.Resize<BpeVocabulary.Entry>` and so on), which crossgen cannot precompile outside the framework's version bubble. The others are `BpeVocabulary.Seal` and `BpeModelReader.ScanVocab`: they are marked `[MethodImpl(MethodImplOptions.AggressiveOptimization)]`, which makes ReadyToRun skip them, so they are always compiled at first call.

## What this means for the library

- A caller that needs a fast cold start should publish with Native AOT. The library is compatible today, but nothing guarantees it stays so: the project does not set `IsAotCompatible`, so the trimming and AOT analyzers do not run on it.
- Without AOT, the first-call cost is spread over about 500 methods. Shrinking it would mean fewer distinct generic instantiations over library structs (replacing `List<struct>` with arrays or one library-owned growable buffer, which ReadyToRun can precompile), building the `BertUnicode` mark table (a static constructor of 11,086 IL bytes) from constant data, and keeping `AggressiveOptimization` only where ReadyToRun is not used. None of these was tried here, so the gain is not measured.

## Reproduce

```sh
tools/fetch-assets.sh
taskset -c 0,1 dotnet run -c Release --project benchmarks/Starlings.Tokenizers.Benchmarks -- --filter '*'
taskset -c 0,1 dotnet run -c Release --project benchmarks/Starlings.Tokenizers.Benchmarks -- --filter '*' --runtimes nativeaot10.0

dotnet publish benchmarks/Starlings.Tokenizers.LoadBench -c Release -r linux-x64 --self-contained false -p:PublishReadyToRun=true -o out/r2r
dotnet publish benchmarks/Starlings.Tokenizers.LoadBench -c Release -r linux-x64 -p:PublishAot=true -o out/aot
f=.models/sentence-transformers/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json
taskset -c 0,1 out/aot/Starlings.Tokenizers.LoadBench cold "$f" 20
DOTNET_JitStdOutFile=jit.txt DOTNET_JitDisasmSummary=1 out/r2r/Starlings.Tokenizers.LoadBench child "$f"
```

Native AOT needs `clang` and `zlib1g-dev`. `--runtimes nativeaot10.0` adds a steady-state job, not a cold one; the AOT cold numbers above come from the published LoadBench.
