#!/usr/bin/env python3
"""Phase 7.9C RAMA low-ram-fast benchmark harness.

This script builds a compute-ready raw/tile-block `.rllm` artifact and runs the
same release RSS matrix used by Phase 7.6/7.8/7.9B. The point is to measure the
RAMA low-ram-fast trade-off honestly:

- raw tile-block artifact: larger cold storage, much cheaper token-loop decode
- same streamed runtime: bounded active memory, no full PyTorch-style load
- same correctness path: optional `rllm verify` plus existing HF parity script
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

from phase76_release_rss_benchmark import (
    ROOT,
    RLLM_BIN,
    BenchResult,
    build_release,
    ensure_ready,
    parse_csv_ints,
    run_benchmark,
    run_command,
    write_csv,
    write_markdown,
)

DEFAULT_SOURCE = ROOT / "models" / "pythia-70m" / "model.safetensors"
DEFAULT_CONFIG = ROOT / "models" / "pythia-70m" / "config.json"
DEFAULT_TOKENIZER = ROOT / "models" / "pythia-70m" / "tokenizer.json"
DEFAULT_ARTIFACT = ROOT / "models" / "pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm"
DEFAULT_OUT_DIR = ROOT / "target" / "phase79c-low-ram-fast"
DEFAULT_BASELINE_CSV = ROOT / "target" / "phase79b-embedding-row-bench" / "phase76_release_rss_benchmark.csv"


def run_checked(command: list[str], *, timeout_seconds: int | None) -> None:
    completed = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def pack_artifact(
    *,
    source: Path,
    artifact: Path,
    config: Path,
    tokenizer: Path,
    tile_block_elements: int,
    codec: str,
    range_checksum_size: str | None,
    timeout_seconds: int | None,
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(RLLM_BIN),
        "pack",
        str(source),
        "--out",
        str(artifact),
        "--codec",
        codec,
        "--tile-block-elements",
        str(tile_block_elements),
        "--config",
        str(config),
        "--tokenizer",
        str(tokenizer),
    ]
    if range_checksum_size:
        command.extend(["--range-checksum-size", range_checksum_size])
    run_checked(command, timeout_seconds=timeout_seconds)


def verify_artifact(*, source: Path, artifact: Path, timeout_seconds: int | None) -> None:
    run_checked(
        [str(RLLM_BIN), "verify", str(source), str(artifact)],
        timeout_seconds=timeout_seconds,
    )


def read_csv_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def successful(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if int(row.get("exit_code", "1")) == 0]


def write_comparison_markdown(
    path: Path,
    *,
    artifact: Path,
    baseline_csv: Path | None,
    current_csv: Path,
    codec: str,
    rama_integrity: str,
    tile_block_elements: int,
) -> None:
    current_rows = successful(read_csv_results(current_csv))
    if not current_rows:
        raise SystemExit(f"no successful benchmark rows in {current_csv}")

    current_spt = [numeric(row, "seconds_per_token") for row in current_rows]
    current_rss = [numeric(row, "max_rss_mib") for row in current_rows]
    current_tokens_per_second = [1.0 / value for value in current_spt if value > 0]

    lines = [
        "# Phase 7.9C RAMA Low-RAM-Fast Benchmark Summary",
        "",
        f"- Artifact: `{artifact}`",
        f"- Codec policy: `{codec}`",
        f"- Runtime integrity: `{rama_integrity}`",
        f"- Tile-block elements: `{tile_block_elements}`",
        f"- Current CSV: `{current_csv}`",
        "",
        "## Current result",
        "",
        f"- Successful rows: `{len(current_rows)}`",
        f"- seconds/token: `{min(current_spt):.2f}`–`{max(current_spt):.2f}`; avg `{mean(current_spt):.2f}`",
        f"- tokens/second: `{min(current_tokens_per_second):.3f}`–`{max(current_tokens_per_second):.3f}`; avg `{mean(current_tokens_per_second):.3f}`",
        f"- max RSS MiB: `{min(current_rss):.2f}`–`{max(current_rss):.2f}`; avg `{mean(current_rss):.2f}`",
        "",
    ]

    if baseline_csv and baseline_csv.exists():
        baseline_rows = successful(read_csv_results(baseline_csv))
        baseline_by_key = {
            (int(row["ctx"]), int(row["max_new_tokens"])): row for row in baseline_rows
        }
        speedups: list[float] = []
        lines.extend(
            [
                "## Baseline comparison",
                "",
                f"- Baseline CSV: `{baseline_csv}`",
                "",
                "| ctx | tokens | baseline s/token | current s/token | speedup | baseline RSS | current RSS |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in current_rows:
            key = (int(row["ctx"]), int(row["max_new_tokens"]))
            baseline = baseline_by_key.get(key)
            if not baseline:
                continue
            baseline_spt = numeric(baseline, "seconds_per_token")
            current_spt_value = numeric(row, "seconds_per_token")
            speedup = baseline_spt / current_spt_value
            speedups.append(speedup)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(key[0]),
                        str(key[1]),
                        f"{baseline_spt:.2f}",
                        f"{current_spt_value:.2f}",
                        f"{speedup:.2f}×",
                        f"{numeric(baseline, 'max_rss_mib'):.2f}",
                        f"{numeric(row, 'max_rss_mib'):.2f}",
                    ]
                )
                + " |"
            )
        if speedups:
            lines.extend(
                [
                    "",
                    f"- Speedup range: `{min(speedups):.2f}×`–`{max(speedups):.2f}×`; avg `{mean(speedups):.2f}×`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Baseline comparison",
                "",
                "Baseline CSV was not found, so no paired comparison was written.",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--codec", default="raw", help="pack codec policy; default raw for low-ram-fast layout")
    parser.add_argument("--tile-block-elements", type=int, default=65536)
    parser.add_argument("--range-checksum-size", default=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", default="1,4,8,16", help="comma-separated max-new-tokens values")
    parser.add_argument("--ctx", default="128,512,1024", help="comma-separated context lengths")
    parser.add_argument("--memory-budget", default="100mb")
    parser.add_argument(
        "--rama-integrity",
        default="strict",
        help="runtime integrity mode passed to `rllm run`: strict or verify-once",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    ctx_values = parse_csv_ints(args.ctx, flag_name="--ctx")
    token_values = parse_csv_ints(args.tokens, flag_name="--tokens")
    source = args.source.resolve()
    artifact = args.artifact.resolve()
    out_dir = args.out_dir.resolve()

    build_release(skip_build=args.skip_build)

    if not args.skip_pack:
        pack_artifact(
            source=source,
            artifact=artifact,
            config=args.config.resolve(),
            tokenizer=args.tokenizer.resolve(),
            tile_block_elements=args.tile_block_elements,
            codec=args.codec,
            range_checksum_size=args.range_checksum_size,
            timeout_seconds=args.timeout_seconds,
        )

    ensure_ready(artifact)

    if not args.skip_verify:
        verify_artifact(source=source, artifact=artifact, timeout_seconds=args.timeout_seconds)

    results: list[BenchResult] = []
    extra_run_args = ["--rama-integrity", args.rama_integrity]
    for ctx in ctx_values:
        for max_new_tokens in token_values:
            result = run_benchmark(
                artifact=artifact,
                prompt=args.prompt,
                ctx=ctx,
                max_new_tokens=max_new_tokens,
                memory_budget=args.memory_budget,
                timeout_seconds=args.timeout_seconds,
                extra_run_args=extra_run_args,
            )
            results.append(result)
            write_csv(out_dir / "phase79c_low_ram_fast_benchmark.csv", results)
            write_markdown(
                out_dir / "phase79c_low_ram_fast_benchmark.md",
                results,
                artifact=artifact,
                prompt=args.prompt,
                memory_budget=args.memory_budget,
            )
            if result.exit_code != 0:
                print(
                    f"benchmark failed for ctx={ctx}, tokens={max_new_tokens}; stopping",
                    file=sys.stderr,
                )
                return result.exit_code

    comparison_path = out_dir / "phase79c_low_ram_fast_summary.md"
    write_comparison_markdown(
        comparison_path,
        artifact=artifact,
        baseline_csv=args.baseline_csv.resolve() if args.baseline_csv else None,
        current_csv=out_dir / "phase79c_low_ram_fast_benchmark.csv",
        codec=args.codec,
        rama_integrity=args.rama_integrity,
        tile_block_elements=args.tile_block_elements,
    )
    print(f"Wrote {out_dir / 'phase79c_low_ram_fast_benchmark.csv'}")
    print(f"Wrote {out_dir / 'phase79c_low_ram_fast_benchmark.md'}")
    print(f"Wrote {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
