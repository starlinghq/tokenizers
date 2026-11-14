# Cold start against Python `tokenizers`, every pinned asset, 2026-11-14

`Tokenizer.FromFile` in a fresh .NET process against `Tokenizer.from_file` in a fresh Python interpreter, for all 16 pinned `tokenizer.json` assets (13 profiles and the three WordPiece files without one). The earlier records ([e5](tokenizer-load-2026-10-09.md), [Gemma](tokenizer-gemma-bpe-2026-10-10.md), [Qwen](tokenizer-qwen-bpe-2026-10-10.md)) were taken on the maintainer's workstation, under background load, against `tokenizers` 0.23.2 and 0.23.3. This run uses a rented machine and the current reference.

## Setup

| | |
| --- | --- |
| Library | `main` at `5951cbe70718d7205e26ff17d70adede56a9be59` (0.3.5 code), Release build of `benchmarks/Starlings.Tokenizers.LoadBench` |
| .NET | SDK 10.0.115, runtime 10.0.15, default runtime settings (tiered compilation and PGO on) |
| Reference | Python 3.12.3, `tokenizers` 0.23.3 from PyPI (the Rust library's Python binding), measured by [`benchmarks/python-cold-load.py`](../../benchmarks/python-cold-load.py) |
| Machine | RunPod CPU pod, flavor `cpu5c`, 4 vCPU and 8 GB, data center EUR-IS-1, image `runpod/base:1.6.0-ubuntu2404`, kernel 6.8. The host is an AMD EPYC 9575F (128 hardware threads, shared with other tenants; its load average of about 50 shows through `/proc/loadavg`). |
| CPUs | The pod's cpuset was `47,55,111,119`: two physical cores and their SMT siblings, on different CCDs. There was no CFS quota and no throttling (`cpu.max` is `max`, `nr_throttled` 0). |
| Assets | Copied from `tools/fetch-assets.sh` output and checked against `tools/pinned-assets.txt` on the pod (all 16 digests matched, plus the derived Qwen3.5 file) |
| Cost | About $0.03 (pod at $0.14/h, about 15 minutes) |

Method, for each asset: read the file once to put it in the page cache; one untimed load on each side, so the .NET binaries and the Python site-packages are cached too; then three interleaved rounds of `LoadBench cold <file> 10` and `python-cold-load.py <file> 10`. Each run is a new process. Numbers are the median of the three round medians.

- **load**: `Tokenizer.FromFile` or `Tokenizer.from_file`, timed inside the process. It includes reading the file and, for .NET, compiling the loader (JIT). It excludes runtime start-up and, for Python, `from tokenizers import Tokenizer` (5 to 6 ms on every asset).
- **process wall**: spawn to exit, timed by the parent. It includes runtime start-up, the load, one encode and process exit.

Every asset gave identical ids for the query `query: how much protein should a female eat` in .NET and Python.

## Results (both sides pinned to one physical core)

All processes were run under `taskset -c 55,119`: one physical core with its two hardware threads. .NET's background JIT and GC threads share that core with the loader, so this setting does not favor .NET. The reason for pinning is explained under [The unpinned run](#the-unpinned-run).

| Profile | Asset | Size, MB | .NET load, ms | Python load, ms | .NET / Python | .NET wall, ms | Python wall, ms | .NET peak working set, MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma-bpe-embeddinggemma2` | `google/embeddinggemma-2` | 32.2 | **222** | 636 | 0.35 | 284 | 728 | 142 |
| `clover-bpe-chat` | `Diffuspace/Clover-35B-A3B-Preview` | 20.0 | **126** | 397 | 0.32 | 193 | 469 | 100 |
| `qwen35-bpe-chat` | `Qwen/Qwen3.5-0.8B` (derived) | 20.0 | **125** | 398 | 0.31 | 189 | 473 | 100 |
| `qwen2-bpe-kev` | `jaredpalmer/kev-0.8b` | 20.0 | **119** | 371 | 0.32 | 184 | 442 | 100 |
| `lfm3-bpe-chat` | `LiquidAI/LFM3-1.7B` | 17.9 | **126** | 311 | 0.41 | 188 | 361 | 91 |
| `xlmr-unigram-e5` | `intfloat/multilingual-e5-small` | 17.1 | **129** | 379 | 0.34 | 188 | 643 | 84 |
| `phi4-bpe-chat` | `microsoft/Phi-4-mini-instruct` | 15.5 | **106** | 276 | 0.39 | 167 | 334 | 96 |
| `qwen3-bpe-embedding` | `Qwen/Qwen3-Embedding-0.6B` | 11.4 | **77** | 205 | 0.38 | 144 | 252 | 75 |
| `qwen3-bpe-chat` | `Qwen/Qwen3-Reranker-0.6B` | 11.4 | **73** | 199 | 0.37 | 138 | 246 | 74 |
| `xlmr-unigram-paraphrase-multilingual` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 9.1 | **124** | 365 | 0.34 | 182 | 621 | 80 |
| `gliner2-base-v1-unigram` | `fastino/gliner2-base-v1` | 8.7 | **80** | 136 | 0.58 | 134 | 242 | 62 |
| `modernbert-bpe-gte-reranker` | `Alibaba-NLP/gte-reranker-modernbert-base` | 3.6 | 45 | 50 | 0.90 | 109 | 79 | 52 |
| `whisper-bpe` | `openai/whisper-large-v3-turbo` | 2.7 | 45 | **33** | 1.39 | 102 | 54 | 47 |
| (no profile, WordPiece) | `BAAI/bge-small-en-v1.5` | 0.7 | 37 | **9** | 4.17 | 97 | 26 | 47 |
| (no profile, WordPiece) | `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.7 | 37 | **9** | 4.22 | 100 | 26 | 47 |
| (no profile, WordPiece) | `sentence-transformers/all-MiniLM-L6-v2` | 0.5 | 37 | **9** | 4.13 | 98 | 27 | 47 |

The three round medians agreed within 6% for every .NET row except one (a round of 90 ms against 77 ms for `qwen3-bpe-embedding`), and within 10% for every Python row.

What the numbers say:

- **Large vocabularies (9 MB and up): .NET loads in 0.31 to 0.41 of Python's time** (GLiNER2, 0.58), and the whole process exits 1.8 to 3.4 times sooner. The workstation records found the same pattern: 0.54 of the reference for e5, about 0.33 for Qwen and Gemma.
- **Small files (under 1 MB): Python wins, 9 ms against 37 ms.** The .NET load does not drop below about 37 ms however small the file. That floor is mostly a fixed cost: compiling the loader and JSON paths at first call, which the native Rust library does not pay. The .NET process wall has a floor of about 100 ms (runtime start-up), against about 27 ms for Python. In a long-running .NET process that has already started, the gap is the 28 ms load difference, once.
- **Around 3 MB the two are even** (ModernBERT 45 against 50 ms, Whisper 45 against 33 ms).
- Python's process wall is well above import plus load for the large Unigram files (e5: 643 ms against 6 + 379 ms; paraphrase-multilingual: 621 against 6 + 365). The rest is interpreter start-up and exit, which for these files appears to include freeing the loaded tokenizer. This was observed, not profiled.

## The unpinned run

The same matrix was first run without `taskset`. Files that are nearly identical gave .NET loads about 2x apart, steady within a round: `qwen3-bpe-chat` 152 ms against `qwen3-bpe-embedding` 85 ms (both 11.4 MB, same vocabulary), and `qwen2-bpe-kev` 225 ms against `qwen35-bpe-chat` 126 ms (20 MB files that differ in one regex). The per-run times were bimodal, and pinning showed where the two modes came from:

| `LoadBench cold`, 8 runs each, load ms | Qwen3-Embedding | Qwen3-Reranker |
| --- | --- | --- |
| unpinned | 78 146 80 77 147 156 76 84 | 78 142 75 74 142 76 78 72 |
| `taskset -c 47,111` | 151 159 150 156 152 157 156 148 | 154 150 154 151 151 146 147 146 |
| `taskset -c 55,119` | 79 77 81 79 85 84 84 91 | 80 82 89 85 77 74 89 83 |

Core 47 ran the same work at about half the speed of core 55, probably because another tenant was busy on its SMT sibling or its CCD. The scheduler put each child process on either core. So the unpinned medians measure where processes landed as much as the library, and they are not used above. For reference, the unpinned medians were: e5 132 against 437 ms, Gemma 504 against 745 ms, Qwen3.5 126 against 413 ms, all-MiniLM 38 against 10 ms. Anyone repeating this on a shared CPU pod should check `cpuset.cpus.effective` and pin both sides to the same core.

## Reproduce

On a Linux x64 machine with the .NET 10 SDK (`global.json`: 10.0.1xx) and Python 3.11 or later:

```sh
tools/fetch-assets.sh
python3 -m venv .venv && .venv/bin/pip install tokenizers==0.23.3
dotnet build -c Release benchmarks/Starlings.Tokenizers.LoadBench
B=benchmarks/Starlings.Tokenizers.LoadBench/bin/Release/net10.0/Starlings.Tokenizers.LoadBench
f=.models/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json
cat "$f" > /dev/null
for round in 1 2 3; do
  taskset -c 0,1 $B cold "$f" 10
  taskset -c 0,1 .venv/bin/python benchmarks/python-cold-load.py "$f" 10
done
```

Replace `0,1` with one physical core and its SMT sibling (`lscpu -e` shows the pairs).
