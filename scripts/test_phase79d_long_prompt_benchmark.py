#!/usr/bin/env python3
"""Focused checks for the Phase 7.9D benchmark report writer."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from phase79d_long_prompt_benchmark import LongPromptResult, write_csv


def make_result() -> LongPromptResult:
    return LongPromptResult(
        input_tokens=128,
        ctx=2048,
        max_new_tokens=4,
        memory_budget="100mb",
        exit_code=0,
        real_seconds=8.0,
        user_seconds=6.0,
        sys_seconds=1.0,
        max_rss_bytes=104_857_600,
        peak_footprint_bytes=52_428_800,
        peak_transient="32.00 MiB",
        generated_token_ids="[1, 2, 3, 4]",
        generated_text="abcd",
        full_text_prefix="promptabcd",
        rama_timing_path="timing.json",
        prefill_ms=1000.0,
        decode_ms=2000.0,
        final_norm_ms=30.0,
        lm_head_ms=40.0,
        sampling_ms=5.0,
        prefill_embedding_ms=10.0,
        prefill_layer_params_ms=20.0,
        prefill_attention_norm_ms=30.0,
        prefill_attention_ms=40.0,
        prefill_attention_qkv_projection_ms=50.0,
        prefill_attention_qkv_split_ms=60.0,
        prefill_attention_rotary_ms=70.0,
        prefill_attention_score_context_ms=80.0,
        prefill_attention_output_projection_ms=90.0,
        prefill_attention_kv_append_ms=100.0,
        prefill_attention_residual_ms=110.0,
        prefill_mlp_norm_ms=120.0,
        prefill_mlp_ms=130.0,
        prefill_mlp_input_projection_ms=140.0,
        prefill_mlp_activation_ms=150.0,
        prefill_mlp_output_projection_ms=160.0,
        prefill_mlp_residual_ms=170.0,
        prefill_chunks=2,
        decode_steps=4,
        max_prefill_chunk_tokens=64,
        prefill_timed_blocks=12,
        command="rllm run example.rllm",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.csv"
        write_csv(path, [make_result()])
        with path.open(newline="") as f:
            row = next(csv.DictReader(f))

    assert row["real_seconds"] == "8.00"
    assert row["user_seconds"] == "6.00"
    assert row["sys_seconds"] == "1.00"
    assert row["seconds_per_generated_token"] == "2.00"
    assert row["generated_tokens_per_second"] == "0.5000"
    assert row["decode_tokens_per_second"] == "2.0000"
    assert row["prefill_tokens_per_second"] == "128.0000"
    assert row["max_rss_bytes"] == "104857600"
    assert row["peak_footprint_bytes"] == "52428800"
    assert row["rama_timing_path"] == "timing.json"
    assert row["command"] == "rllm run example.rllm"


if __name__ == "__main__":
    main()
