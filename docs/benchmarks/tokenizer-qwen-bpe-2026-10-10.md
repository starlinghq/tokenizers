# Owned tokenizer, Qwen byte-level BPE profiles: load and encode, 2026-10-10

> **Historical record.** Written in the Starling planning repository before this repository was split out. The package then had a planning-era name (it is `Starling.Tokenizers` throughout here); paths such as `tokenizers-dotnet/`, `model-pipeline/` and `tools/*-tokenizer-*` refer to that repository's layout (see [docs/repository-history.md](../repository-history.md)).

Exploratory. `Starling.Tokenizers` 0.1.0-preview.3 (this change) against the Rust `tokenizers` 0.23.2 (Python binding) on the two Qwen profile assets. One workstation, not a controlled machine: Intel i7-5820K (6 cores), Linux x64 7.2, .NET 10.0.12 (SDK 10.0.100), Release builds, assets in the OS page cache. **Background load average 5 to 8 during every series** (a local LLM server using about 2.5 cores, other sessions); absolute numbers are inflated by it (the e5 cold load measured 445 ms in the same window against 270 ms in [tokenizer-load-2026-10-09.md](tokenizer-load-2026-10-09.md)). The C# and Rust series ran back to back, so the ratios are the more useful figure.

Assets: `qwen2-bpe-kev` (`jaredpalmer/kev-0.8b@9a45d25e` `tokenizer.json`, 19,989,325 bytes, `06b95093…`) and `qwen35-bpe-chat` (derived transformers 5.18.0 pipeline, 19,989,339 bytes, `a5cd9732…`): 248,044 vocabulary entries, 247,587 merges, 33 added tokens.

| Profile | Measure | Owned (C#) | Rust tokenizers 0.23.2 | Ratio |
| --- | --- | ---: | ---: | ---: |
| qwen2-bpe-kev | cold load, ms (median of 15; min to max) | **312** (301–329) | 955 (921–1,020) | 3.1x |
| qwen2-bpe-kev | warm reload, ms (median of 10) | 216 (206–270) | | |
| qwen2-bpe-kev | encode, ms per round (median of 20) | **205** (199–219) | 529 (522–564) | 2.6x |
| qwen35-bpe-chat | cold load, ms | **330** (318–342) | 981 (965–991) | 3.0x |
| qwen35-bpe-chat | warm reload, ms | 228 (219–277) | | |
| qwen35-bpe-chat | encode, ms per round | **217** (210–229) | 532 (528–538) | 2.5x |

Other cold-process figures (both profiles): 66.9 MiB allocated in the whole process, 101 MiB peak working set, 45.7 MiB managed heap after load with the tokenizer alive. A Stopwatch split of one quieter cold load (load average 4.2; temporary instrumentation, not committed): locate `model.vocab` 0.4 ms, scan the vocab object 31 ms, hash and id table 14 ms, scan and resolve merges 92 ms (three hash lookups per merge), rest of the document, added tokens and model 17 ms; 189 ms in total.

Method:

- Cold: `LoadBench cold <tokenizer.json> 15`, a fresh process per run, `load_ms` = `Tokenizer.FromFile` inside the process (JIT included, runtime start-up excluded). Rust: a fresh Python interpreter per run timing `Tokenizer.from_file` after the import (script below), 15 runs.
- Encode: `LoadBench encode <tokenizer.json> 20`: 200 x 5 texts per round (short English query, English passage, German/French, CJK/Korean, a 3,800-character English paragraph; 160,200 tokens per round), 3 warm-up rounds. Rust: the same texts and loop through `Tokenizer.encode(text).ids` in Python, so the binding's per-call overhead is in the Rust figure. Id checksums equal (42294230800 for 20 rounds in both).
- No word cache in the owned BPE (upstream has a per-thread cache); encode is still faster on these texts.

```sh
B=benchmarks/Starling.Tokenizers.LoadBench/bin/Release/net10.0/Starling.Tokenizers.LoadBench
$B cold   "$STARLING_MODELS/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e/tokenizer.json" 15
$B warm   "$STARLING_MODELS/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e/tokenizer.json" 10
$B encode "$STARLING_MODELS/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e/tokenizer.json" 20
# Rust, in the qwen-bpe reference environment (model-pipeline/reference/qwen-bpe/requirements-chat.txt):
python - "$STARLING_MODELS/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e/tokenizer.json" <<'EOF'
import subprocess, sys, json, statistics
child = "import time,json;t=time.perf_counter;from tokenizers import Tokenizer;a=t();Tokenizer.from_file(%r);print(json.dumps((t()-a)*1000))"
runs = [json.loads(subprocess.run([sys.executable, "-c", child % sys.argv[1]], capture_output=True, text=True).stdout) for _ in range(15)]
print("load_ms median", statistics.median(runs))
EOF
```
