# Native AOT: conformance and cold start on every pinned asset, 2026-11-14

The library is now marked `IsAotCompatible`. This record checks that it is correct under Native AOT (the whole test suite, run as an AOT binary) and measures cold start with Native AOT against the default JIT and Python `tokenizers` for all 16 pinned `tokenizer.json` files. It follows [tokenizer-first-call-2026-11-14.md](tokenizer-first-call-2026-11-14.md), which found that small-file cold start is almost all JIT.

## Setup

| | |
| --- | --- |
| Library | `feat/aot-compatible`: locked-mode (`CI=true`) JIT tests at `1728997` and AOT tests at `060ea3d` (which changes only the AOT test project), benchmarks at `92f8b0e` (the same library and LoadBench code) |
| .NET | SDK 10.0.115, runtime 10.0.15; xUnit.net v3 4.0.2 (`xunit.v3.aot` for the AOT run) |
| Reference | Python 3.12.3, `tokenizers` 0.23.3 |
| Machine | RunPod CPU pod, `cpu5c`, 4 vCPU, 8 GB, EUR-IS-1, AMD EPYC 9575F (shared host, load average 40 to 50). Cpuset: two physical cores. A probe gave 37.5 ms and 37.6 ms for a cold MiniLM load on the two cores; every timed process ran under `taskset -c 40,104`. |
| Cost | About $0.05 for this pod |

## Correctness under Native AOT

- **Analyzers.** With `IsAotCompatible`, the trim, AOT and single-file analyzers run on the library. It builds with no warnings (warnings are errors).
- **Package.** The setting makes the SDK add an implicit, build-time-only `Microsoft.NET.ILLink.Tasks` reference, which is now in `src/Starlings.Tokenizers/packages.lock.json`. The packed nuspec still lists no dependencies.
- **Test suite.** `tests/Starlings.Tokenizers.AotTests` compiles the `Starlings.Tokenizers.Tests` sources against xUnit.net's Native AOT packages and publishes them with Native AOT (a 19.9 MB native executable). With `STARLING_REQUIRE_ASSETS=1`, **6,004 of 6,004 tests pass**, every profile's conformance fixtures included. The JIT suite passes 6,008 of 6,008 on the same commit. The difference is the two classes that inspect metadata by reflection and are left out of the AOT build: `IndependenceTests` and `PinnedAssetsTests`.
- **Test-source change.** Two test helpers built their input `tokenizer.json` by serializing anonymous objects, which System.Text.Json cannot do under Native AOT: `BertWordPieceComponentTests._json` and `O200kSplitPatternTests.MinimalTokenizerJson`. In the first AOT run, the 14 tests that use them failed before reaching the tokenizer. Both helpers now build the same documents with `JsonObject`. A comparison program confirmed the text is byte-identical to the old output, for the BERT document and all six o200k variants. No fixture changed.
- **Locked restore.** Locked-mode restore (`CI=true`) passes for the solution and for the AOT test project. The AOT test project sets `RuntimeIdentifier` itself; `-r linux-x64` on the command line would also change the library project's restore.

## Cold start: JIT, Native AOT and Python

For each file: one untimed run of each build, then 30 rounds of a JIT child (`LoadBench child`), a Native AOT child (the same program published with `PublishAot`) and a Python child, in rotating order. **load** is `FromFile` or `from_file` timed inside the process. **first encode** is one encode of `query: how much protein should a female eat` right after the load. **wall** is spawn to exit. Medians of 30.

| Asset | Size, MB | JIT load | **AOT load** | Python load | AOT / Python | JIT first encode | AOT first encode | Python first encode | JIT wall | AOT wall | Python wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/embeddinggemma-2` | 32.2 | 202.2 | **170.6** | 525.0 | 0.32 | 12.13 | 0.06 | 0.10 | 245 | 181 | 599 |
| `Qwen/Qwen3.5-0.8B` (derived) | 20.0 | 115.6 | **82.9** | 330.5 | 0.25 | 16.36 | 0.32 | 0.18 | 160 | 92 | 386 |
| `jaredpalmer/kev-0.8b` | 20.0 | 112.2 | **79.7** | 307.9 | 0.26 | 15.69 | 0.31 | 0.18 | 155 | 88 | 358 |
| `Diffuspace/Clover-35B-A3B-Preview` | 20.0 | 114.3 | **81.7** | 318.7 | 0.26 | 16.23 | 0.32 | 0.18 | 159 | 91 | 371 |
| `LiquidAI/LFM3-1.7B` | 17.9 | 114.7 | **80.3** | 250.6 | 0.32 | 12.47 | 0.30 | 0.14 | 154 | 89 | 288 |
| `intfloat/multilingual-e5-small` | 17.1 | 127.1 | **87.0** | 325.2 | 0.27 | 14.77 | 0.07 | 0.14 | 167 | 93 | 507 |
| `microsoft/Phi-4-mini-instruct` | 15.5 | 94.9 | **65.3** | 234.7 | 0.28 | 14.17 | 0.31 | 0.16 | 136 | 73 | 278 |
| `Qwen/Qwen3-Embedding-0.6B` | 11.4 | 78.9 | **41.7** | 169.3 | 0.25 | 17.32 | 0.32 | 0.19 | 122 | 49 | 206 |
| `Qwen/Qwen3-Reranker-0.6B` | 11.4 | 76.9 | **42.6** | 169.7 | 0.25 | 16.36 | 0.31 | 0.18 | 120 | 50 | 207 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 9.1 | 124.9 | **81.5** | 324.8 | 0.25 | 15.83 | 0.09 | 0.34 | 164 | 87 | 516 |
| `fastino/gliner2-base-v1` | 8.7 | 80.3 | **37.9** | 114.9 | 0.33 | 10.25 | 0.05 | 0.12 | 114 | 42 | 193 |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 3.6 | 44.7 | **11.3** | 44.9 | 0.25 | 18.15 | 0.47 | 0.69 | 88 | 17 | 69 |
| `openai/whisper-large-v3-turbo` | 2.7 | 46.6 | **10.3** | 33.1 | 0.31 | 15.29 | 0.29 | 0.16 | 87 | 16 | 55 |
| `BAAI/bge-small-en-v1.5` | 0.7 | 37.1 | **4.7** | 9.1 | 0.52 | 21.81 | 1.57 | 0.07 | 81 | 9 | 28 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.7 | 37.4 | **4.7** | 9.2 | 0.51 | 22.03 | 1.60 | 0.07 | 82 | 9 | 28 |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.5 | 36.9 | **4.4** | 8.8 | 0.50 | 24.00 | 1.61 | 0.22 | 82 | 9 | 28 |

All times are in ms. The ids were identical across JIT, Native AOT and Python on all 16 files.

- **With Native AOT, every file loads faster than in Python**, at 0.25 to 0.52 of Python's time, and the whole process exits 3.1 to 5.9 times sooner (MiniLM: 9 ms against 28 ms).
- **With the JIT**, files of 9 MB and more load 2.1 to 2.9 times faster than in Python (GLiNER2, at 8.7 MB, 1.4 times). Files under about 3 MB are slower, because of the roughly 30 ms first-call cost.
- **The first encode** is 10 to 24 ms with the JIT and under 0.5 ms with Native AOT, except on the three WordPiece files. There it is 1.6 ms against Python's 0.07 to 0.22 ms. That is not compilation (there is none under AOT). Most likely it is the BERT normalizer building its Unicode tables (canonical decomposition, used to strip accents) on first use; this was not profiled. MiniLM's load plus first encode is still faster under AOT than in Python (6.0 against 9.0 ms).

## Reproduce

```sh
tools/fetch-assets.sh
dotnet publish tests/Starlings.Tokenizers.AotTests -c Release -o tests/Starlings.Tokenizers.AotTests/bin/aot
STARLING_REQUIRE_ASSETS=1 tests/Starlings.Tokenizers.AotTests/bin/aot/Starlings.Tokenizers.AotTests

dotnet build -c Release benchmarks/Starlings.Tokenizers.LoadBench
dotnet publish benchmarks/Starlings.Tokenizers.LoadBench -c Release -r linux-x64 -p:PublishAot=true -o out/aot
f=.models/sentence-transformers/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json
taskset -c 0,1 benchmarks/Starlings.Tokenizers.LoadBench/bin/Release/net10.0/Starlings.Tokenizers.LoadBench cold "$f" 30
taskset -c 0,1 out/aot/Starlings.Tokenizers.LoadBench cold "$f" 30
taskset -c 0,1 python benchmarks/python-cold-load.py "$f" 30
```

Native AOT needs `clang` and `zlib1g-dev`. The rotation driver used for the table runs the three `child` commands in turn, as in the commands above; the record's numbers are its medians.
