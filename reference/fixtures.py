"""Reference fixture format helpers shared by the runners under reference/.

Pure Python so every reference runner can share it. Each runner lives in its own directory with
its own pinned requirements and imports this module from the parent directory.

Offset units are always explicit. Upstream Python tokenizers report offsets as indices into a
Python `str`, which are Unicode scalar values (code points). Fixtures record that unit and the
UTF-16 and UTF-8 conversions next to it, so a C# consumer can slice strings without guessing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

FIXTURE_FORMAT_VERSION = 1
OFFSET_UNITS = ("unicode-scalars", "utf16-code-units", "utf8-bytes")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL corpus. Each line has `id`, `category`, `text`, and optionally `pair`."""
    cases = []
    seen = set()
    # Split on LF only: str.splitlines() also splits on U+0085, U+2028 and other characters
    # that corpus text may contain literally.
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "category", "text"):
            if key not in case:
                raise ValueError(f"{path}:{number}: missing {key!r}")
        if case["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id {case['id']!r}")
        seen.add(case["id"])
        cases.append(case)
    return cases


def _prefix_lengths(text: str, width) -> list[int]:
    """Cumulative length in the target unit before each code point, plus the total."""
    out = [0]
    for ch in text:
        out.append(out[-1] + width(ch))
    return out


def _utf16_width(ch: str) -> int:
    return 2 if ord(ch) > 0xFFFF else 1


def _utf8_width(ch: str) -> int:
    return len(ch.encode("utf-8", "surrogatepass"))


def convert_offsets(text: str, offsets: Iterable[Sequence[int]], unit: str) -> list[list[int]]:
    """Convert exclusive-end code point offsets into `unit` for the same `text`."""
    if unit == "unicode-scalars":
        return [[int(s), int(e)] for s, e in offsets]
    if unit == "utf16-code-units":
        table = _prefix_lengths(text, _utf16_width)
    elif unit == "utf8-bytes":
        table = _prefix_lengths(text, _utf8_width)
    else:
        raise ValueError(f"unknown offset unit {unit!r}")
    return [[table[s], table[e]] for s, e in offsets]


def all_offset_units(text: str, offsets: Iterable[Sequence[int]]) -> dict[str, list[list[int]]]:
    scalars = [[int(s), int(e)] for s, e in offsets]
    return {unit: convert_offsets(text, scalars, unit) for unit in OFFSET_UNITS}


def float32_list(values: Iterable[Any]) -> list[float]:
    """Shortest decimal that round-trips each value as float32, as a JSON-serializable float.

    `repr(numpy.float32(x))` style digits are produced without importing numpy here: the value
    is formatted with increasing precision until it parses back to the same float32 bits.
    """
    import struct

    out = []
    for v in values:
        f = float(v)
        bits = struct.pack("<f", f)
        for digits in range(1, 18):
            text = f"{f:.{digits}g}"
            if struct.pack("<f", float(text)) == bits:
                out.append(float(text))
                break
        else:  # pragma: no cover - 17 significant digits always round-trip
            out.append(f)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    name = str(path.relative_to(relative_to)) if relative_to else path.name
    return {"path": name.replace(os.sep, "/"), "sha256": sha256_file(path), "size": path.stat().st_size}


def environment(packages: Sequence[str]) -> dict[str, Any]:
    """Interpreter, platform and package versions for a reference run."""
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "packages": versions,
    }


def git_head(path: Path | None = None) -> str | None:
    """Commit of the repository that holds the runner, so a fixture names the code that made it."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out + ("-dirty" if dirty else "")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """One compact JSON object per line, UTF-8, LF, non-ASCII kept literal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
