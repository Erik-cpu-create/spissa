#!/usr/bin/env python3
"""Phase 7.9E RAMA long-prompt timing benchmark.

Compares the existing full-prompt prefill path against opt-in RAMA chunked
prefill (`--rama-prefill-chunk-tokens`) with low-overhead aggregate timings from
`--rama-timing`. The harness varies actual input token count via deterministic
`--token-ids`; it does not rely on `--ctx` as a proxy for prompt length.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "target" / "release" / "rllm"
DEFAULT_ARTIFACT = ROOT / "models" / "pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm"
DEFAULT_OUT = ROOT / "target" / "phase79e-prefill-timing"
TOKEN_PATTERN = [12092, 13, 187, 309, 352, 359, 42, 253, 849, 619, 627, 368, 198, 318, 262, 257]


@dataclass
class TimingRow:
    input_tokens: int
    max_new_tokens: int
    ctx: int
    prefill_chunk_tokens: str
    exit_code: int
    elapsed_seconds: float
    seconds_per_generated_token: float | None
    generated_tokens_per_second: float | None
    max_rss_mib: float | None
    peak_memory_footprint_mib: float | None
    context_memory_mib: float | None
    peak_transient_kib: float | None
    prefill_ms: float | None
    decode_ms: float | None
    final_norm_ms: float | None
    lm_head_ms: float | None
    sampling_ms: float | None
    prefill_chunks: int | None
    decode_steps: int | None
    max_prefill_chunk_tokens: int | None
    generated_token_ids: str | None
    command: str
    error_tail: str | None


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def token_ids(length: int) -> str:
    if length <= 0:
        raise ValueError("input token length must be positive")
    repeats = (length + len(TOKEN_PATTERN) - 1) // len(TOKEN_PATTERN)
    return ",".join(str(value) for value in (TOKEN_PATTERN * repeats)[:length])


def parse_mib_line(stdout: str, label: str) -> float | None:
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)\s+MiB", stdout, re.MULTILINE)
    if match:
        return float(match.group(1))
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)\s+KiB", stdout, re.MULTILINE)
    if match:
        return float(match.group(1)) / 1024.0
    match = re.search(rf"^{re.escape(label)}:\s+([0-9]+)\s+B", stdout, re.MULTILINE)
    if match:
        return float(match.group(1)) / (1024.0 * 1024.0)
    return None


def parse_kib_line(stdout: str, label: str) -> float | None:
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)\s+KiB", stdout, re.MULTILINE)
    if match:
        return float(match.group(1))
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)\s+MiB", stdout, re.MULTILINE)
    if match:
        return float(match.group(1)) * 1024.0
    match = re.search(rf"^{re.escape(label)}:\s+([0-9]+)\s+B", stdout, re.MULTILINE)
    if match:
        return float(match.group(1)) / 1024.0
    return None


def parse_generated(stdout: str) -> str | None:
    match = re.search(r"^Generated token IDs:\s+(\[[^\n]+\])", stdout, re.MULTILINE)
    return match.group(1) if match else None


def parse_time_l(stderr: str) -> tuple[float | None, float | None]:
    rss = None
    footprint = None
    for line in stderr.splitlines():
        stripped = line.strip()
        match = re.match(r"(\d+)\s+maximum resident set size", stripped)
        if match:
            rss = int(match.group(1)) / (1024.0 * 1024.0)
        match = re.match(r"(\d+)\s+peak memory footprint", stripped)
        if match:
            footprint = int(match.group(1)) / (1024.0 * 1024.0)
    return rss, footprint


def read_timing(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("summary", {})


def run_one(args: argparse.Namespace, input_len: int, max_new_tokens: int, chunk_tokens: int | None) -> TimingRow:
    out_dir = pathlib.Path(args.out_dir)
    timing_name = f"input{input_len}_new{max_new_tokens}_chunk{chunk_tokens or 'full'}.json"
    timing_path = out_dir / timing_name
    cmd = [
        "/usr/bin/time",
        "-l",
        str(args.bin),
        "run",
        str(args.artifact),
        "--mode",
        "tile-stream",
        "--ctx",
        str(args.ctx),
        "--memory-budget",
        args.memory_budget,
        "--token-ids",
        token_ids(input_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--rama-integrity",
        args.rama_integrity,
        "--rama-timing",
        str(timing_path),
    ]
    if chunk_tokens is not None:
        cmd.extend(["--rama-prefill-chunk-tokens", str(chunk_tokens)])

    started = time.perf_counter()
    completed = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout_seconds)
    elapsed = time.perf_counter() - started
    rss, footprint = parse_time_l(completed.stderr)
    timing = read_timing(timing_path) if timing_path.exists() else {}
    sec_per_token = elapsed / max_new_tokens if max_new_tokens else None
    tok_per_sec = max_new_tokens / elapsed if elapsed > 0 else None
    return TimingRow(
        input_tokens=input_len,
        max_new_tokens=max_new_tokens,
        ctx=args.ctx,
        prefill_chunk_tokens=str(chunk_tokens) if chunk_tokens is not None else "full",
        exit_code=completed.returncode,
        elapsed_seconds=elapsed,
        seconds_per_generated_token=sec_per_token,
        generated_tokens_per_second=tok_per_sec,
        max_rss_mib=rss,
        peak_memory_footprint_mib=footprint,
        context_memory_mib=parse_mib_line(completed.stdout, "Context memory bytes"),
        peak_transient_kib=parse_kib_line(completed.stdout, "Peak transient budget"),
        prefill_ms=timing.get("prefill_ms"),
        decode_ms=timing.get("decode_ms"),
        final_norm_ms=timing.get("final_norm_ms"),
        lm_head_ms=timing.get("lm_head_ms"),
        sampling_ms=timing.get("sampling_ms"),
        prefill_chunks=timing.get("prefill_chunks"),
        decode_steps=timing.get("decode_steps"),
        max_prefill_chunk_tokens=timing.get("max_prefill_chunk_tokens"),
        generated_token_ids=parse_generated(completed.stdout),
        command=" ".join(cmd[:16] + ["--token-ids", f"<{input_len} ids>"] + cmd[18:]),
        error_tail="\n".join((completed.stderr + completed.stdout).splitlines()[-24:]) if completed.returncode else None,
    )


def write_outputs(rows: Sequence[TimingRow], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase79e_prefill_timing.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    md = [
        "# Phase 7.9E RAMA Prefill Timing Benchmark",
        "",
        "| input | new | chunk | elapsed_s | tok/s | RSS MiB | context MiB | transient KiB | prefill ms | decode ms | lm_head ms | prefill chunks |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: object, digits: int = 2) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)
        md.append(
            "| "
            + " | ".join(
                [
                    str(row.input_tokens),
                    str(row.max_new_tokens),
                    row.prefill_chunk_tokens,
                    fmt(row.elapsed_seconds),
                    fmt(row.generated_tokens_per_second, 3),
                    fmt(row.max_rss_mib),
                    fmt(row.context_memory_mib),
                    fmt(row.peak_transient_kib),
                    fmt(row.prefill_ms),
                    fmt(row.decode_ms),
                    fmt(row.lm_head_ms),
                    str(row.prefill_chunks),
                ]
            )
            + " |"
        )
    (out_dir / "phase79e_prefill_timing.md").write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=pathlib.Path, default=DEFAULT_BIN)
    parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--input-tokens", type=parse_int_list, default=[512])
    parser.add_argument("--max-new-tokens", type=parse_int_list, default=[16])
    parser.add_argument("--prefill-chunks", type=str, default="full,128")
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--memory-budget", default="100mb")
    parser.add_argument("--rama-integrity", default="verify-once")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[int | None] = []
    for item in args.prefill_chunks.split(","):
        item = item.strip().lower()
        if item in {"", "full", "none"}:
            chunks.append(None)
        else:
            chunks.append(int(item))

    rows: list[TimingRow] = []
    for input_len in args.input_tokens:
        for max_new in args.max_new_tokens:
            for chunk in chunks:
                row = run_one(args, input_len, max_new, chunk)
                rows.append(row)
                print(
                    f"input={input_len} new={max_new} chunk={chunk or 'full'} "
                    f"exit={row.exit_code} elapsed={row.elapsed_seconds:.2f}s "
                    f"tok/s={(row.generated_tokens_per_second or 0):.3f} "
                    f"rss={row.max_rss_mib}MiB prefill_ms={row.prefill_ms}"
                )
                if row.exit_code != 0:
                    raise SystemExit(row.error_tail or "benchmark row failed")
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
