# Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
"""Cold-load reference for LoadBench: huggingface/tokenizers (Python binding) in a fresh interpreter per run.

  python benchmarks/python-cold-load.py <tokenizer.json> [runs]

Mirrors `LoadBench cold`: load_ms is `Tokenizer.from_file` timed inside the child (import excluded, reported
separately); process wall_ms is spawn to exit, timed by this parent. Prints a summary in LoadBench's format and the
ids of the same query LoadBench encodes, so the two outputs can be compared.
"""

import json
import subprocess
import sys
import time

CHILD = r"""
import json, sys, time
t0 = time.perf_counter()
from tokenizers import Tokenizer
t1 = time.perf_counter()
tokenizer = Tokenizer.from_file(sys.argv[1])
t2 = time.perf_counter()
ids = tokenizer.encode("query: how much protein should a female eat").ids
t3 = time.perf_counter()
print(json.dumps({"import_ms": (t1 - t0) * 1e3, "load_ms": (t2 - t1) * 1e3, "first_encode_ms": (t3 - t2) * 1e3,
                  "ids": " ".join(map(str, ids))}))
"""


def summary(name: str, values: list[float]) -> str:
    s = sorted(values)

    def q(p: float) -> float:
        return s[round(p * (len(s) - 1))]

    return (
        f"{name}: median {q(0.5):.1f}, min {s[0]:.1f}, p25 {q(0.25):.1f}, p75 {q(0.75):.1f}, max {s[-1]:.1f} "
        f"(n={len(s)})"
    )


def load_average() -> str:
    try:
        with open("/proc/loadavg") as f:
            return f.read().strip()
    except OSError:
        return "n/a"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python-cold-load.py <tokenizer.json> [runs]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    load: list[float] = []
    imports: list[float] = []
    wall: list[float] = []
    ids = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", CHILD, path], capture_output=True, text=True, check=True).stdout
        wall.append((time.perf_counter() - t0) * 1e3)
        r = json.loads(out)
        load.append(r["load_ms"])
        imports.append(r["import_ms"])
        if ids is not None and ids != r["ids"]:
            print("ids differ between runs", file=sys.stderr)
            return 1
        ids = r["ids"]
    print(f"python cold runs: {runs}; load average after: {load_average()}")
    print(summary("load_ms (Tokenizer.from_file)", load))
    print(summary("import_ms (from tokenizers import Tokenizer)", imports))
    print(summary("process wall_ms", wall))
    print(f"ids: {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
