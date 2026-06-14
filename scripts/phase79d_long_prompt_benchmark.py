#!/usr/bin/env python3
"""Phase 7.9D real long-prompt benchmark for RLLM.

Unlike the earlier ctx matrix, this harness varies the actual prompt length by
passing deterministic fixed token IDs to `rllm run --token-ids`. That separates
context capacity (`--ctx`) from real prefill/input length.

The reported speed is end-to-end generated-token throughput, so it includes
artifact open, prompt/prefill processing, decode, sampling, and process startup.
For long prompts this is intentionally conservative and reflects chat request
latency more honestly than the short-prompt matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from phase76_release_rss_benchmark import (
    ROOT,
    RLLM_BIN,
    build_release,
    ensure_ready,
    format_optional_float,
    format_optional_int,
    parse_csv_ints,
    parse_time_l_output,
    run_command,
)

DEFAULT_ARTIFACT = ROOT / "models" / "pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm"
DEFAULT_OUT_DIR = ROOT / "target" / "phase79d-long-prompt"

# Deterministic, known-valid Pythia token IDs observed in previous top-k/logit
# checks. Repeating these avoids tokenizer effects while exercising real input
# token counts. All IDs are below the Pythia-70M vocab size (50304).
BASE_TOKEN_PATTERN = [
    12092,
    13,
    187,
    309,
    352,
    359,
    42,
    253,
    849,
    619,
    627,
    368,
    198,
    318,
    262,
    257,
]


@dataclass(frozen=True)
class LongPromptResult:
    input_tokens: int
    ctx: int
    max_new_tokens: int
    memory_budget: str
    exit_code: int
    real_seconds: float | None
    user_seconds: float | None
    sys_seconds: float | None
    max_rss_bytes: int | None
    peak_footprint_bytes: int | None
    peak_transient: str | None
    generated_token_ids: str | None
    generated_text: str | None
    full_text_prefix: str | None
    rama_timing_path: str | None
    prefill_ms: float | None
    decode_ms: float | None
    final_norm_ms: float | None
    lm_head_ms: float | None
    sampling_ms: float | None
    prefill_embedding_ms: float | None
    prefill_layer_params_ms: float | None
    prefill_attention_norm_ms: float | None
    prefill_attention_ms: float | None
    prefill_attention_qkv_projection_ms: float | None
    prefill_attention_qkv_split_ms: float | None
    prefill_attention_rotary_ms: float | None
    prefill_attention_score_context_ms: float | None
    prefill_attention_output_projection_ms: float | None
    prefill_attention_kv_append_ms: float | None
    prefill_attention_residual_ms: float | None
    prefill_mlp_norm_ms: float | None
    prefill_mlp_ms: float | None
    prefill_mlp_input_projection_ms: float | None
    prefill_mlp_activation_ms: float | None
    prefill_mlp_output_projection_ms: float | None
    prefill_mlp_residual_ms: float | None
    prefill_chunks: int | None
    decode_steps: int | None
    max_prefill_chunk_tokens: int | None
    prefill_timed_blocks: int | None
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
    def seconds_per_generated_token(self) -> float | None:
        if self.real_seconds is None or self.max_new_tokens == 0:
            return None
        return self.real_seconds / self.max_new_tokens

    @property
    def generated_tokens_per_second(self) -> float | None:
        value = self.seconds_per_generated_token
        if value is None or value <= 0:
            return None
        return 1.0 / value
    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.decode_ms is None or self.decode_steps is None or self.decode_steps == 0 or self.decode_ms == 0:
            return None
        return self.decode_steps / (self.decode_ms / 1000.0)

    @property
    def prefill_tokens_per_second(self) -> float | None:
        if self.prefill_ms is None or self.input_tokens is None or self.prefill_ms == 0:
            return None
        return self.input_tokens / (self.prefill_ms / 1000.0)

def make_token_ids(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    repeated = (BASE_TOKEN_PATTERN * ((length + len(BASE_TOKEN_PATTERN) - 1) // len(BASE_TOKEN_PATTERN)))
    return repeated[:length]


def token_ids_arg(length: int) -> str:
    return ",".join(str(token_id) for token_id in make_token_ids(length))


def shortened(value: str | None, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def metric_float(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    return value if isinstance(value, float) else None


def metric_int(metrics: dict[str, object], key: str) -> int | None:
    value = metrics.get(key)
    return value if isinstance(value, int) else None


def metric_str(metrics: dict[str, object], key: str) -> str | None:
    value = metrics.get(key)
    return value if isinstance(value, str) else None


def timing_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def timing_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) else None


def read_timing_payload(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def run_long_prompt_benchmark(
    *,
    artifact: Path,
    input_tokens: int,
    ctx: int,
    max_new_tokens: int,
    memory_budget: str,
    rama_integrity: str,
    timeout_seconds: int | None,
    rama_timing_path: Path | None = None,
    extra_run_args: Sequence[str] = (),
) -> LongPromptResult:
    if input_tokens + max_new_tokens > ctx:
        raise SystemExit(
            f"input_tokens + max_new_tokens exceeds ctx: "
            f"{input_tokens}+{max_new_tokens}>{ctx}"
        )
    command = [
        "/usr/bin/time",
        "-l",
        str(RLLM_BIN),
        "run",
        str(artifact),
        "--token-ids",
        token_ids_arg(input_tokens),
        "--max-new-tokens",
        str(max_new_tokens),
        "--ctx",
        str(ctx),
        "--memory-budget",
        memory_budget,
        "--rama-integrity",
        rama_integrity,
    ]
    if rama_timing_path is not None:
        rama_timing_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--rama-timing", str(rama_timing_path)])
    command.extend(extra_run_args)
    completed = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    print(completed.stdout, end="")
    metrics = parse_time_l_output(completed.stdout)
    full_text = metric_str(metrics, "full_text")
    timing_payload = read_timing_payload(rama_timing_path)
    return LongPromptResult(
        input_tokens=input_tokens,
        ctx=ctx,
        max_new_tokens=max_new_tokens,
        memory_budget=memory_budget,
        exit_code=completed.returncode,
        real_seconds=metric_float(metrics, "real_seconds"),
        user_seconds=metric_float(metrics, "user_seconds"),
        sys_seconds=metric_float(metrics, "sys_seconds"),
        max_rss_bytes=metric_int(metrics, "max_rss_bytes"),
        peak_footprint_bytes=metric_int(metrics, "peak_footprint_bytes"),
        peak_transient=metric_str(metrics, "peak_transient"),
        generated_token_ids=metric_str(metrics, "generated_token_ids"),
        generated_text=metric_str(metrics, "generated_text"),
        full_text_prefix=shortened(full_text),
        rama_timing_path=str(rama_timing_path) if rama_timing_path is not None else None,
        prefill_ms=timing_float(timing_payload, "prefill_ms"),
        decode_ms=timing_float(timing_payload, "decode_ms"),
        final_norm_ms=timing_float(timing_payload, "final_norm_ms"),
        lm_head_ms=timing_float(timing_payload, "lm_head_ms"),
        sampling_ms=timing_float(timing_payload, "sampling_ms"),
        prefill_embedding_ms=timing_float(timing_payload, "prefill_embedding_ms"),
        prefill_layer_params_ms=timing_float(timing_payload, "prefill_layer_params_ms"),
        prefill_attention_norm_ms=timing_float(timing_payload, "prefill_attention_norm_ms"),
        prefill_attention_ms=timing_float(timing_payload, "prefill_attention_ms"),
        prefill_attention_qkv_projection_ms=timing_float(
            timing_payload, "prefill_attention_qkv_projection_ms"
        ),
        prefill_attention_qkv_split_ms=timing_float(
            timing_payload, "prefill_attention_qkv_split_ms"
        ),
        prefill_attention_rotary_ms=timing_float(timing_payload, "prefill_attention_rotary_ms"),
        prefill_attention_score_context_ms=timing_float(
            timing_payload, "prefill_attention_score_context_ms"
        ),
        prefill_attention_output_projection_ms=timing_float(
            timing_payload, "prefill_attention_output_projection_ms"
        ),
        prefill_attention_kv_append_ms=timing_float(
            timing_payload, "prefill_attention_kv_append_ms"
        ),
        prefill_attention_residual_ms=timing_float(timing_payload, "prefill_attention_residual_ms"),
        prefill_mlp_norm_ms=timing_float(timing_payload, "prefill_mlp_norm_ms"),
        prefill_mlp_ms=timing_float(timing_payload, "prefill_mlp_ms"),
        prefill_mlp_input_projection_ms=timing_float(
            timing_payload, "prefill_mlp_input_projection_ms"
        ),
        prefill_mlp_activation_ms=timing_float(timing_payload, "prefill_mlp_activation_ms"),
        prefill_mlp_output_projection_ms=timing_float(
            timing_payload, "prefill_mlp_output_projection_ms"
        ),
        prefill_mlp_residual_ms=timing_float(timing_payload, "prefill_mlp_residual_ms"),
        prefill_chunks=timing_int(timing_payload, "prefill_chunks"),
        decode_steps=timing_int(timing_payload, "decode_steps"),
        max_prefill_chunk_tokens=timing_int(timing_payload, "max_prefill_chunk_tokens"),
        prefill_timed_blocks=timing_int(timing_payload, "prefill_timed_blocks"),
        command=" ".join(shlex.quote(part) for part in command),
    )


def write_csv(path: Path, results: Iterable[LongPromptResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input_tokens",
                "ctx",
                "max_new_tokens",
                "memory_budget",
                "exit_code",
                "real_seconds",
                "seconds_per_generated_token",
                "generated_tokens_per_second",
                "decode_tokens_per_second",
                "prefill_tokens_per_second",
                "user_seconds",
                "sys_seconds",
                "max_rss_bytes",
                "max_rss_mib",
                "peak_footprint_bytes",
                "peak_footprint_mib",
                "peak_transient",
                "rama_timing_path",
                "prefill_ms",
                "decode_ms",
                "final_norm_ms",
                "lm_head_ms",
                "sampling_ms",
                "prefill_embedding_ms",
                "prefill_layer_params_ms",
                "prefill_attention_norm_ms",
                "prefill_attention_ms",
                "prefill_attention_qkv_projection_ms",
                "prefill_attention_qkv_split_ms",
                "prefill_attention_rotary_ms",
                "prefill_attention_score_context_ms",
                "prefill_attention_output_projection_ms",
                "prefill_attention_kv_append_ms",
                "prefill_attention_residual_ms",
                "prefill_mlp_norm_ms",
                "prefill_mlp_ms",
                "prefill_mlp_input_projection_ms",
                "prefill_mlp_activation_ms",
                "prefill_mlp_output_projection_ms",
                "prefill_mlp_residual_ms",
                "prefill_chunks",
                "decode_steps",
                "max_prefill_chunk_tokens",
                "prefill_timed_blocks",
                "generated_token_ids",
                "generated_text",
                "full_text_prefix",
                "command",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    str(result.input_tokens),
                    str(result.ctx),
                    str(result.max_new_tokens),
                    result.memory_budget,
                    str(result.exit_code),
                    format_optional_float(result.real_seconds),
                    format_optional_float(result.seconds_per_generated_token),
                    format_optional_float(result.generated_tokens_per_second, precision=4),
                    format_optional_float(result.decode_tokens_per_second, precision=4),
                    format_optional_float(result.prefill_tokens_per_second, precision=4),
                    format_optional_float(result.user_seconds),
                    format_optional_float(result.sys_seconds),
                    format_optional_int(result.max_rss_bytes),
                    format_optional_float(result.max_rss_mib),
                    format_optional_int(result.peak_footprint_bytes),
                    format_optional_float(result.peak_footprint_mib),
                    result.peak_transient or "",
                    result.rama_timing_path or "",
                    format_optional_float(result.prefill_ms),
                    format_optional_float(result.decode_ms),
                    format_optional_float(result.final_norm_ms),
                    format_optional_float(result.lm_head_ms),
                    format_optional_float(result.sampling_ms),
                    format_optional_float(result.prefill_embedding_ms),
                    format_optional_float(result.prefill_layer_params_ms),
                    format_optional_float(result.prefill_attention_norm_ms),
                    format_optional_float(result.prefill_attention_ms),
                    format_optional_float(result.prefill_attention_qkv_projection_ms),
                    format_optional_float(result.prefill_attention_qkv_split_ms),
                    format_optional_float(result.prefill_attention_rotary_ms),
                    format_optional_float(result.prefill_attention_score_context_ms),
                    format_optional_float(result.prefill_attention_output_projection_ms),
                    format_optional_float(result.prefill_attention_kv_append_ms),
                    format_optional_float(result.prefill_attention_residual_ms),
                    format_optional_float(result.prefill_mlp_norm_ms),
                    format_optional_float(result.prefill_mlp_ms),
                    format_optional_float(result.prefill_mlp_input_projection_ms),
                    format_optional_float(result.prefill_mlp_activation_ms),
                    format_optional_float(result.prefill_mlp_output_projection_ms),
                    format_optional_float(result.prefill_mlp_residual_ms),
                    format_optional_int(result.prefill_chunks),
                    format_optional_int(result.decode_steps),
                    format_optional_int(result.max_prefill_chunk_tokens),
                    format_optional_int(result.prefill_timed_blocks),
                    result.generated_token_ids or "",
                    result.generated_text or "",
                    result.full_text_prefix or "",
                    result.command,
                ]
            )


def successful(results: Iterable[LongPromptResult]) -> list[LongPromptResult]:
    return [result for result in results if result.exit_code == 0]


def write_markdown(
    path: Path,
    results: list[LongPromptResult],
    *,
    artifact: Path,
    memory_budget: str,
    rama_integrity: str,
    ctx: int,
) -> None:
    ok = successful(results)
    lines = [
        "# Phase 7.9D Real Long-Prompt Benchmark",
        "",
        f"- Artifact: `{artifact}`",
        f"- Runtime integrity: `{rama_integrity}`",
        f"- Memory budget: `{memory_budget}`",
        f"- Context capacity: `{ctx}`",
        f"- Input token pattern: deterministic fixed token IDs, not tokenizer text",
        "",
        "## Method",
        "",
        "This benchmark varies the *actual* `--token-ids` prompt length. The throughput column is end-to-end generated-token throughput and includes process startup, artifact open, prefill/input processing, decode, sampling, and output printing.",
        "",
    ]
    if ok:
        spt = [r.seconds_per_generated_token for r in ok if r.seconds_per_generated_token is not None]
        tps = [r.generated_tokens_per_second for r in ok if r.generated_tokens_per_second is not None]
        rss = [r.max_rss_mib for r in ok if r.max_rss_mib is not None]
        lines.extend(
            [
                "## Summary",
                "",
                f"- Successful rows: `{len(ok)}/{len(results)}`",
                f"- seconds/generated-token: `{min(spt):.2f}`–`{max(spt):.2f}`; avg `{mean(spt):.2f}`",
                f"- generated tokens/sec: `{min(tps):.3f}`–`{max(tps):.3f}`; avg `{mean(tps):.3f}`",
                f"- max RSS MiB: `{min(rss):.2f}`–`{max(rss):.2f}`; avg `{mean(rss):.2f}`",
                "",
            ]
        )
    else:
        lines.extend(["## Summary", "", "No successful rows.", ""])

    lines.extend(
        [
            "## Rows",
            "",
            "| input tokens | new tokens | budget | real sec | sec/gen token | end-to-end tok/s | decode tok/s | prefill tok/s | OS max RSS MiB | peak transient | prefill ms | embed ms | layer-param ms | attn ms | qkv ms | score/context ms | attn out ms | rotary ms | kv append ms | mlp ms | mlp in ms | gelu ms | mlp out ms | decode ms | lm_head ms | timed blocks | exit |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.input_tokens),
                    str(r.max_new_tokens),
                    r.memory_budget,
                    format_optional_float(r.real_seconds),
                    format_optional_float(r.seconds_per_generated_token),
                    format_optional_float(r.generated_tokens_per_second, precision=3),
                    format_optional_float(r.decode_tokens_per_second, precision=3),
                    format_optional_float(r.prefill_tokens_per_second, precision=3),
                    format_optional_float(r.max_rss_mib),
                    r.peak_transient or "",
                    format_optional_float(r.prefill_ms),
                    format_optional_float(r.prefill_embedding_ms),
                    format_optional_float(r.prefill_layer_params_ms),
                    format_optional_float(r.prefill_attention_ms),
                    format_optional_float(r.prefill_attention_qkv_projection_ms),
                    format_optional_float(r.prefill_attention_score_context_ms),
                    format_optional_float(r.prefill_attention_output_projection_ms),
                    format_optional_float(r.prefill_attention_rotary_ms),
                    format_optional_float(r.prefill_attention_kv_append_ms),
                    format_optional_float(r.prefill_mlp_ms),
                    format_optional_float(r.prefill_mlp_input_projection_ms),
                    format_optional_float(r.prefill_mlp_activation_ms),
                    format_optional_float(r.prefill_mlp_output_projection_ms),
                    format_optional_float(r.decode_ms),
                    format_optional_float(r.lm_head_ms),
                    format_optional_int(r.prefill_timed_blocks),
                    str(r.exit_code),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is not a pure decode-loop microbenchmark; it is an end-to-end CLI request benchmark.",
            "- Use `--rama-timing-dir <dir>` to collect aggregate prefill/decode/lm_head/sampling timing JSON per row.",
            "- macOS RSS does not include all OS page-cache effects; cold-cache/warm-cache measurement remains a separate follow-up.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-tokens", default="1,128,512,1024", help="comma-separated actual prompt token counts")
    parser.add_argument("--max-new-tokens", default="1,4,16", help="comma-separated generation lengths")
    parser.add_argument("--ctx", type=int, default=2048, help="context capacity; must exceed input+generated tokens")
    parser.add_argument("--memory-budget", default="100mb")
    parser.add_argument("--rama-integrity", default="verify-once")
    parser.add_argument("--rama-prefill-chunk-tokens", type=int)
    parser.add_argument("--rama-timing-dir", type=Path, help="optional directory for per-row --rama-timing JSON")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    out_dir = args.out_dir.resolve()
    input_lengths = parse_csv_ints(args.input_tokens, flag_name="--input-tokens")
    generation_lengths = parse_csv_ints(args.max_new_tokens, flag_name="--max-new-tokens")

    budgets = [b.strip() for b in args.memory_budget.split(",") if b.strip()]

    build_release(skip_build=args.skip_build)
    ensure_ready(artifact)

    results: list[LongPromptResult] = []
    csv_path = out_dir / "phase79d_long_prompt_benchmark.csv"
    md_path = out_dir / "phase79d_long_prompt_benchmark.md"
    for input_tokens in input_lengths:
        for max_new_tokens in generation_lengths:
            for memory_budget in budgets:
                extra_run_args: list[str] = []
                if args.rama_prefill_chunk_tokens is not None:
                    extra_run_args.extend(["--rama-prefill-chunk-tokens", str(args.rama_prefill_chunk_tokens)])
                rama_timing_path = None
                if args.rama_timing_dir is not None:
                    timing_dir = args.rama_timing_dir
                    if not timing_dir.is_absolute():
                        timing_dir = out_dir / timing_dir
                    rama_timing_path = (
                        timing_dir / f"rama_timing_input{input_tokens}_new{max_new_tokens}_budget{memory_budget}.json"
                    )
                result = run_long_prompt_benchmark(
                    artifact=artifact,
                    input_tokens=input_tokens,
                    ctx=args.ctx,
                    max_new_tokens=max_new_tokens,
                    memory_budget=memory_budget,
                    rama_integrity=args.rama_integrity,
                    timeout_seconds=args.timeout_seconds,
                    rama_timing_path=rama_timing_path,
                    extra_run_args=extra_run_args,
                )
                results.append(result)
                write_csv(csv_path, results)
                write_markdown(
                    md_path,
                    results,
                    artifact=artifact,
                    memory_budget=args.memory_budget,
                    rama_integrity=args.rama_integrity,
                    ctx=args.ctx,
                )
                if result.exit_code != 0:
                    print(
                        f"benchmark failed for input_tokens={input_tokens}, max_new_tokens={max_new_tokens}, budget={memory_budget}; stopping",
                        file=sys.stderr,
                    )
                    return result.exit_code

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
