#!/usr/bin/env python3
"""Run Phase 7.6 release RSS benchmarks for RLLM token generation.

This harness measures the compiled `target/release/rllm` binary with
macOS `/usr/bin/time -l` so RSS reflects the CLI process rather than
Cargo's wrapper process. Generated CSV/Markdown files default to
`target/phase76-bench/` so benchmark artifacts stay out of git.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "models" / "pythia-70m-phase76-16mb.rllm"
DEFAULT_OUT_DIR = ROOT / "target" / "phase76-bench"
RLLM_BIN = ROOT / "target" / "release" / "rllm"


@dataclass(frozen=True)
class BenchResult:
    ctx: int
    max_new_tokens: int
    exit_code: int
    real_seconds: float | None
    user_seconds: float | None
    sys_seconds: float | None
    max_rss_bytes: int | None
    peak_footprint_bytes: int | None
    peak_transient: str | None
    generated_token_ids: str | None
    generated_text: str | None
    full_text: str | None
    command: str

    @property
    def max_rss_mib(self) -> float | None:
        if self.max_rss_bytes is None:
            return None
        return self.max_rss_bytes / 1024 / 1024

    @property
    def peak_footprint_mib(self) -> float | None:
        if self.peak_footprint_bytes is None:
            return None
        return self.peak_footprint_bytes / 1024 / 1024

    @property
    def seconds_per_token(self) -> float | None:
        if self.real_seconds is None or self.max_new_tokens == 0:
            return None
        return self.real_seconds / self.max_new_tokens


def parse_csv_ints(raw: str, *, flag_name: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise SystemExit(f"{flag_name} must be comma-separated integers: {raw!r}") from exc
        if value <= 0:
            raise SystemExit(f"{flag_name} values must be > 0: {value}")
        values.append(value)
    if not values:
        raise SystemExit(f"{flag_name} must contain at least one value")
    return values


def parse_time_l_output(output: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    timing = re.search(
        r"^\s*([0-9]+(?:\.[0-9]+)?) real\s+([0-9]+(?:\.[0-9]+)?) user\s+([0-9]+(?:\.[0-9]+)?) sys",
        output,
        flags=re.MULTILINE,
    )
    if timing:
        metrics["real_seconds"] = float(timing.group(1))
        metrics["user_seconds"] = float(timing.group(2))
        metrics["sys_seconds"] = float(timing.group(3))

    max_rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size", output, re.MULTILINE)
    if max_rss:
        metrics["max_rss_bytes"] = int(max_rss.group(1))

    peak_footprint = re.search(r"^\s*([0-9]+)\s+peak memory footprint", output, re.MULTILINE)
    if peak_footprint:
        metrics["peak_footprint_bytes"] = int(peak_footprint.group(1))

    generated_token_ids = re.search(r"^Generated token IDs:\s*(.+)$", output, re.MULTILINE)
    if generated_token_ids:
        metrics["generated_token_ids"] = generated_token_ids.group(1).strip()

    generated_text = re.search(r"^Generated text:\s*(.*)$", output, re.MULTILINE)
    if generated_text:
        metrics["generated_text"] = generated_text.group(1)

    full_text = re.search(r"^Full text:\s*(.*)$", output, re.MULTILINE)
    if full_text:
        metrics["full_text"] = full_text.group(1)

    peak_transient = re.search(r"^Peak transient budget:\s*(.+)$", output, re.MULTILINE)
    if peak_transient:
        metrics["peak_transient"] = peak_transient.group(1).strip()

    return metrics


def format_optional_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def format_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def run_command(command: list[str], *, cwd: Path, timeout_seconds: int | None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )


def build_release(skip_build: bool) -> None:
    if skip_build:
        return
    completed = run_command(["cargo", "build", "--release"], cwd=ROOT, timeout_seconds=None)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def ensure_ready(artifact: Path) -> None:
    if not artifact.exists():
        raise SystemExit(
            f"artifact does not exist: {artifact}\n"
            "Repack first, e.g. `cargo run -- pack models/pythia-70m/model.safetensors "
            "--out models/pythia-70m-phase76-16mb.rllm --chunk-size 16mb "
            "--config models/pythia-70m/config.json --tokenizer models/pythia-70m/tokenizer.json`."
        )
    if not RLLM_BIN.exists():
        raise SystemExit(f"release binary does not exist after build: {RLLM_BIN}")


def run_benchmark(
    *,
    artifact: Path,
    prompt: str,
    ctx: int,
    max_new_tokens: int,
    memory_budget: str,
    timeout_seconds: int | None,
    extra_run_args: Sequence[str] = (),
) -> BenchResult:
    command = [
        "/usr/bin/time",
        "-l",
        str(RLLM_BIN),
        "run",
        str(artifact),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--ctx",
        str(ctx),
        "--memory-budget",
        memory_budget,
    ]
    command.extend(extra_run_args)
    completed = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    print(completed.stdout, end="")
    metrics = parse_time_l_output(completed.stdout)
    return BenchResult(
        ctx=ctx,
        max_new_tokens=max_new_tokens,
        exit_code=completed.returncode,
        real_seconds=metrics.get("real_seconds"),
        user_seconds=metrics.get("user_seconds"),
        sys_seconds=metrics.get("sys_seconds"),
        max_rss_bytes=metrics.get("max_rss_bytes"),
        peak_footprint_bytes=metrics.get("peak_footprint_bytes"),
        peak_transient=metrics.get("peak_transient"),
        generated_token_ids=metrics.get("generated_token_ids"),
        generated_text=metrics.get("generated_text"),
        full_text=metrics.get("full_text"),
        command=" ".join(shlex.quote(part) for part in command),
    )


def write_csv(path: Path, results: Iterable[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ctx",
                "max_new_tokens",
                "exit_code",
                "real_seconds",
                "seconds_per_token",
                "user_seconds",
                "sys_seconds",
                "max_rss_bytes",
                "max_rss_mib",
                "peak_footprint_bytes",
                "peak_footprint_mib",
                "peak_transient",
                "generated_token_ids",
                "generated_text",
                "full_text",
                "command",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.ctx,
                    result.max_new_tokens,
                    result.exit_code,
                    format_optional_float(result.real_seconds),
                    format_optional_float(result.seconds_per_token),
                    format_optional_float(result.user_seconds),
                    format_optional_float(result.sys_seconds),
                    format_optional_int(result.max_rss_bytes),
                    format_optional_float(result.max_rss_mib),
                    format_optional_int(result.peak_footprint_bytes),
                    format_optional_float(result.peak_footprint_mib),
                    result.peak_transient or "",
                    result.generated_token_ids or "",
                    result.generated_text or "",
                    result.full_text or "",
                    result.command,
                ]
            )


def write_markdown(path: Path, results: list[BenchResult], *, artifact: Path, prompt: str, memory_budget: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7.6 Release RSS Benchmark",
        "",
        f"- Artifact: `{artifact}`",
        f"- Prompt: `{prompt}`",
        f"- Memory budget: `{memory_budget}`",
        "- Measurement: macOS `/usr/bin/time -l target/release/rllm ...`",
        "- Note: RSS includes process/runtime overhead; `Peak transient budget` is RLLM internal memory accounting.",
        "",
        "| ctx | tokens | exit | real s | s/token | max RSS MiB | peak footprint MiB | peak transient | generated token IDs | generated text |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.ctx),
                    str(result.max_new_tokens),
                    str(result.exit_code),
                    format_optional_float(result.real_seconds),
                    format_optional_float(result.seconds_per_token),
                    format_optional_float(result.max_rss_mib),
                    format_optional_float(result.peak_footprint_mib),
                    result.peak_transient or "",
                    result.generated_token_ids or "",
                    (result.generated_text or "").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", default="1,4,8,16", help="comma-separated max-new-tokens values")
    parser.add_argument("--ctx", default="128", help="comma-separated context lengths")
    parser.add_argument("--memory-budget", default="100mb")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--run-arg",
        action="append",
        default=[],
        help="extra argument to append to each `rllm run` invocation; repeat for flags with values",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    ctx_values = parse_csv_ints(args.ctx, flag_name="--ctx")
    token_values = parse_csv_ints(args.tokens, flag_name="--tokens")
    artifact = args.artifact.resolve()
    out_dir = args.out_dir.resolve()

    build_release(skip_build=args.skip_build)
    ensure_ready(artifact)

    results: list[BenchResult] = []
    for ctx in ctx_values:
        for max_new_tokens in token_values:
            result = run_benchmark(
                artifact=artifact,
                prompt=args.prompt,
                ctx=ctx,
                max_new_tokens=max_new_tokens,
                memory_budget=args.memory_budget,
                extra_run_args=args.run_arg,
                timeout_seconds=args.timeout_seconds,
            )
            results.append(result)
            write_csv(out_dir / "phase76_release_rss_benchmark.csv", results)
            write_markdown(
                out_dir / "phase76_release_rss_benchmark.md",
                results,
                artifact=artifact,
                prompt=args.prompt,
                memory_budget=args.memory_budget,
            )
            if result.exit_code != 0:
                print(f"benchmark failed for ctx={ctx}, tokens={max_new_tokens}; stopping", file=sys.stderr)
                return result.exit_code

    print(f"Wrote {out_dir / 'phase76_release_rss_benchmark.csv'}")
    print(f"Wrote {out_dir / 'phase76_release_rss_benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
