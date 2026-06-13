#!/usr/bin/env python3
"""Compare RLLM fixed-token logits against a local HuggingFace/PyTorch reference.

This Phase 7.7 harness intentionally uses fixed token IDs instead of prompt text
so tokenizer fidelity is not part of the comparison. It calls the release RLLM
binary to dump first-step logits, loads the local HuggingFace model directory,
and writes JSON/Markdown comparison reports under target/phase77 by default.

Run with dependencies via uv, for example:

    uv run --with torch --with transformers --with safetensors \
      scripts/phase77_compare_logits.py --token-ids 12092
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "pythia-70m"
DEFAULT_RLLM_ARTIFACT = ROOT / "models" / "pythia-70m-phase76-16mb.rllm"
DEFAULT_RLLM_BIN = ROOT / "target" / "release" / "rllm"
DEFAULT_OUT_DIR = ROOT / "target" / "phase77"


@dataclass(frozen=True)
class Comparison:
    prompt_token_ids: list[int]
    rllm_generated_token_ids: list[int]
    vocab_size: int
    max_abs_diff: float
    mean_abs_diff: float
    rms_abs_diff: float
    rllm_top1_id: int
    rllm_top1_logit: float
    hf_top1_id: int
    hf_top1_logit: float
    top1_match: bool
    top5_overlap: int
    top10_overlap: int
    rllm_top10: list[tuple[int, float]]
    hf_top10: list[tuple[int, float]]
    command: str


def parse_token_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise SystemExit(f"invalid token id in --token-ids: {part!r}") from exc
        if value < 0:
            raise SystemExit(f"token ids must be non-negative: {value}")
        ids.append(value)
    if not ids:
        raise SystemExit("--token-ids must contain at least one token id")
    return ids


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")


def run_rllm_logits(
    *,
    rllm_bin: Path,
    artifact: Path,
    token_ids: list[int],
    ctx: int,
    memory_budget: str,
    logits_path: Path,
    timeout_seconds: int,
    rama_integrity: str,
    rama_prefill_chunk_tokens: int | None,
) -> tuple[dict[str, Any], str]:
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(rllm_bin),
        "run",
        str(artifact),
        "--token-ids",
        ",".join(str(token_id) for token_id in token_ids),
        "--max-new-tokens",
        "1",
        "--ctx",
        str(ctx),
        "--memory-budget",
        memory_budget,
        "--logits-out",
        str(logits_path),
        "--rama-integrity",
        rama_integrity,
    ]
    if rama_prefill_chunk_tokens is not None:
        command.extend(["--rama-prefill-chunk-tokens", str(rama_prefill_chunk_tokens)])
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    with logits_path.open() as f:
        return json.load(f), " ".join(shlex.quote(part) for part in command)


def topk(logits: Any, k: int) -> list[tuple[int, float]]:
    values, indices = logits.topk(k)
    return [(int(idx), float(value)) for idx, value in zip(indices.tolist(), values.tolist())]


def compare_logits(
    *,
    model_dir: Path,
    rllm_payload: dict[str, Any],
    command: str,
) -> Comparison:
    try:
        import torch
        from transformers import GPTNeoXForCausalLM
    except Exception as exc:  # pragma: no cover - exercised in dependency-missing environments
        raise SystemExit(
            "Missing HF comparison dependencies. Run with: "
            "uv run --with torch --with transformers --with safetensors "
            "scripts/phase77_compare_logits.py --token-ids 12092"
        ) from exc

    prompt_token_ids = [int(token_id) for token_id in rllm_payload["prompt_token_ids"]]
    rllm_generated_token_ids = [int(token_id) for token_id in rllm_payload["generated_token_ids"]]
    rllm_logits = torch.tensor(rllm_payload["logits"], dtype=torch.float32)

    torch.set_grad_enabled(False)
    model = GPTNeoXForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    with torch.no_grad():
        hf_logits = model(input_ids, use_cache=False).logits[0, -1].detach().cpu().float()

    if hf_logits.numel() != rllm_logits.numel():
        raise SystemExit(
            f"vocab/logits length mismatch: hf={hf_logits.numel()} rllm={rllm_logits.numel()}"
        )

    diff = (hf_logits - rllm_logits).abs()
    rllm_top10 = topk(rllm_logits, 10)
    hf_top10 = topk(hf_logits, 10)
    rllm_top5_ids = {idx for idx, _ in rllm_top10[:5]}
    hf_top5_ids = {idx for idx, _ in hf_top10[:5]}
    rllm_top10_ids = {idx for idx, _ in rllm_top10}
    hf_top10_ids = {idx for idx, _ in hf_top10}

    return Comparison(
        prompt_token_ids=prompt_token_ids,
        rllm_generated_token_ids=rllm_generated_token_ids,
        vocab_size=int(hf_logits.numel()),
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        rms_abs_diff=float(torch.sqrt((diff * diff).mean()).item()),
        rllm_top1_id=rllm_top10[0][0],
        rllm_top1_logit=rllm_top10[0][1],
        hf_top1_id=hf_top10[0][0],
        hf_top1_logit=hf_top10[0][1],
        top1_match=rllm_top10[0][0] == hf_top10[0][0],
        top5_overlap=len(rllm_top5_ids & hf_top5_ids),
        top10_overlap=len(rllm_top10_ids & hf_top10_ids),
        rllm_top10=rllm_top10,
        hf_top10=hf_top10,
        command=command,
    )


def comparison_to_json(comparison: Comparison) -> dict[str, Any]:
    return {
        "prompt_token_ids": comparison.prompt_token_ids,
        "rllm_generated_token_ids": comparison.rllm_generated_token_ids,
        "vocab_size": comparison.vocab_size,
        "max_abs_diff": comparison.max_abs_diff,
        "mean_abs_diff": comparison.mean_abs_diff,
        "rms_abs_diff": comparison.rms_abs_diff,
        "rllm_top1_id": comparison.rllm_top1_id,
        "rllm_top1_logit": comparison.rllm_top1_logit,
        "hf_top1_id": comparison.hf_top1_id,
        "hf_top1_logit": comparison.hf_top1_logit,
        "top1_match": comparison.top1_match,
        "top5_overlap": comparison.top5_overlap,
        "top10_overlap": comparison.top10_overlap,
        "rllm_top10": comparison.rllm_top10,
        "hf_top10": comparison.hf_top10,
        "rllm_command": comparison.command,
    }


def write_markdown(path: Path, comparison: Comparison, *, model_dir: Path, artifact: Path) -> None:
    lines = [
        "# Phase 7.7 HF/PyTorch Logits Comparison",
        "",
        "Fixed token IDs are used so tokenizer behavior is not part of this comparison.",
        "",
        f"- HF model dir: `{model_dir}`",
        f"- RLLM artifact: `{artifact}`",
        f"- Prompt token IDs: `{comparison.prompt_token_ids}`",
        f"- Vocab/logits length: `{comparison.vocab_size}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| max abs diff | {comparison.max_abs_diff:.8f} |",
        f"| mean abs diff | {comparison.mean_abs_diff:.8f} |",
        f"| RMS abs diff | {comparison.rms_abs_diff:.8f} |",
        f"| RLLM top-1 id | {comparison.rllm_top1_id} |",
        f"| HF top-1 id | {comparison.hf_top1_id} |",
        f"| top-1 match | {str(comparison.top1_match).lower()} |",
        f"| top-5 overlap | {comparison.top5_overlap}/5 |",
        f"| top-10 overlap | {comparison.top10_overlap}/10 |",
        "",
        "## Top-10 logits",
        "",
        "| rank | RLLM token/logit | HF token/logit |",
        "|---:|---|---|",
    ]
    for rank, (rllm, hf) in enumerate(zip(comparison.rllm_top10, comparison.hf_top10), start=1):
        lines.append(
            f"| {rank} | {rllm[0]} / {rllm[1]:.6f} | {hf[0]} / {hf[1]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passing scientific comparison should have top-1 match and small absolute-difference metrics. Large differences mean the RLLM runtime is internally self-consistent but not yet HF/PyTorch numerically faithful.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--rllm-artifact", type=Path, default=DEFAULT_RLLM_ARTIFACT)
    parser.add_argument("--rllm-bin", type=Path, default=DEFAULT_RLLM_BIN)
    parser.add_argument("--token-ids", default="12092")
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--memory-budget", default="100mb")
    parser.add_argument("--rama-integrity", default="strict")
    parser.add_argument("--rama-prefill-chunk-tokens", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    artifact = args.rllm_artifact.resolve()
    rllm_bin = args.rllm_bin.resolve()
    out_dir = args.out_dir.resolve()
    token_ids = parse_token_ids(args.token_ids)

    require_path(model_dir, "HF model dir")
    require_path(artifact, "RLLM artifact")
    require_path(rllm_bin, "RLLM binary")

    logits_path = out_dir / "rllm_logits.json"
    comparison_json_path = out_dir / "phase77_logits_comparison.json"
    comparison_md_path = out_dir / "phase77_logits_comparison.md"

    rllm_payload, command = run_rllm_logits(
        rllm_bin=rllm_bin,
        artifact=artifact,
        token_ids=token_ids,
        ctx=args.ctx,
        memory_budget=args.memory_budget,
        logits_path=logits_path,
        timeout_seconds=args.timeout_seconds,
        rama_integrity=args.rama_integrity,
        rama_prefill_chunk_tokens=args.rama_prefill_chunk_tokens,
    )
    comparison = compare_logits(model_dir=model_dir, rllm_payload=rllm_payload, command=command)

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_json_path.write_text(json.dumps(comparison_to_json(comparison), indent=2) + "\n")
    write_markdown(comparison_md_path, comparison, model_dir=model_dir, artifact=artifact)

    print(f"Wrote {comparison_json_path}")
    print(f"Wrote {comparison_md_path}")
    print(
        "summary: "
        f"top1_match={comparison.top1_match} "
        f"max_abs_diff={comparison.max_abs_diff:.8f} "
        f"mean_abs_diff={comparison.mean_abs_diff:.8f} "
        f"rllm_top1={comparison.rllm_top1_id} "
        f"hf_top1={comparison.hf_top1_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
