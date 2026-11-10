"""What every benchmark run records: the machine it ran on, summary statistics, and a results JSON.

A results file holds `machine`, `config`, `samples` and `summary`, so a reader can trace a number
to the commit, hardware, package versions and command that produced it, and recompute it.
"""

import datetime
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
PACKAGES = ("fastapi", "starlette", "httpx", "uvicorn", "uvloop", "httptools", "pydantic")


def _run(cmd: Sequence[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_bytes() -> Optional[int]:
    if sys.platform == "darwin":
        size = _run(["sysctl", "-n", "hw.memsize"])
        return int(size) if size and size.isdigit() else None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _load_average() -> Optional[List[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def describe_machine(script: str, argv: Sequence[str]) -> Dict[str, Any]:
    """Describe this run. Call it before the benchmark starts, so the load average is the one it started under."""
    status = _run(["git", "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "total_ram_bytes": _total_ram_bytes(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {name: _version(name) for name in PACKAGES},
        "load_avg_1_5_15_before": _load_average(),
        "command": shlex.join([Path(sys.executable).name, os.path.relpath(script, REPO_ROOT), *argv]),
    }


def print_machine(machine: Dict[str, Any]) -> None:
    ram = machine["total_ram_bytes"]
    packages = ", ".join(f"{k} {v}" for k, v in machine["packages"].items() if v)
    dirty = " (dirty)" if machine["git_dirty"] else ""
    print(f"run:      {machine['timestamp_utc']}  git {machine['git_sha'] or 'unknown'}{dirty}")
    print(f"command:  {machine['command']}")
    print(f"machine:  {machine['cpu_model']}, {machine['logical_cpus']} logical CPUs"
          f"{f', {ram / 2**30:.1f} GiB RAM' if ram else ''}, {machine['arch']}")
    print(f"os:       {machine['platform']}")
    print(f"python:   {machine['python_implementation']} {machine['python_version']}; {packages}")
    print(f"load avg: {machine['load_avg_1_5_15_before']} (1, 5, 15 min, before the run)")
    print()


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    """Count, mean, stdev, min, max and linearly interpolated p50/p90/p99 (statistics.quantiles, inclusive)."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    cuts = statistics.quantiles(values, n=100, method="inclusive") if n > 1 else [values[0]] * 99
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
        "min": min(values),
        "p50": cuts[49],
        "p90": cuts[89],
        "p99": cuts[98],
        "max": max(values),
    }


def stream_completed(body: bytes) -> bool:
    """True if an SSE body finished normally: a choice carried a finish_reason, no error event, and [DONE] last.

    `[DONE]` alone is not enough, because the gateway's mid-stream error frame also ends with it.
    """
    events = [e for e in body.replace(b"\r\n", b"\n").split(b"\n\n") if e.strip()]
    if not events or events[-1].strip() != b"data: [DONE]":
        return False
    finished = False
    for event in events[:-1]:
        lines = event.split(b"\n")
        if b"event: error" in lines:
            return False
        data = b"\n".join(line[5:].lstrip() for line in lines if line.startswith(b"data:"))
        try:
            chunk = json.loads(data)
        except ValueError:
            return False
        finished = finished or any(c.get("finish_reason") for c in chunk.get("choices") or [])
    return finished


def write_results(
    name: str,
    machine: Dict[str, Any],
    config: Dict[str, Any],
    samples: Any,
    summary: Dict[str, Any],
    out: Optional[str] = None,
) -> Path:
    """Write the results JSON to `out`, or to benchmarks/results/<name>-<timestamp>.json."""
    if out:
        path = Path(out)
    else:
        stamp = machine["timestamp_utc"].replace(":", "").replace("+0000", "Z")
        path = RESULTS_DIR / f"{name}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "benchmark": name,
        "machine": {**machine, "load_avg_1_5_15_after": _load_average()},
        "config": config,
        "summary": summary,
        "samples": samples,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path
