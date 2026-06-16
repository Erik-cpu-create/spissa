# R97 Post-R96 Prefill Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-profile Llama 3.2 1B Q8 prefill after R96 so the next optimization targets the real remaining bottleneck.

**Architecture:** R97 makes no runtime code changes. It runs the existing `llama-test` phase profiler and R93 Q8 branch profiler on the R96 runtime, records normal and profiled runs, and writes a diagnostic benchmark report. The report must identify the next target with measured evidence and preserve the CPU-only, single-thread benchmark discipline.

**Tech Stack:** Rust release `llama-test`, `RLLM_Q8_KERNEL_PROFILE`, `--profile-phases`, benchmark trial docs.

---

## Why This Stage Exists

R96 succeeded:

- pre-control prefill: `13.85s`
- best R96 prefill: `9.03s`
- output: `No`
- internal peak transient: `1,050,673,152 bytes`
- `batch_gt1_scaled` profiled elapsed after R96: `6129.00ms`

The goal is still prefill speed. R97 must determine whether the next stage should keep targeting `batch_gt1_scaled`, move to another Q8 branch, target attention projection, target LM head, or target overhead outside the profiled kernels.

## Scope

Allowed:

- run release `llama-test` with R96 code
- run normal single-thread controls
- run `RLLM_Q8_KERNEL_PROFILE=1` profiled trials
- compare phase timing and Q8 branch timing against R96 report
- write an active/success diagnostic report and update the benchmark index

Not allowed:

- changing runtime code
- changing model artifact
- changing prompt, chat template, sampling, or RAM budget settings
- claiming a speedup from R97
- starting R98 kernel work before report conclusion

## Success Gate

R97 is accepted if:

- release `llama-test` builds
- all benchmark runs answer `No`
- internal peak transient stays `1,050,673,152 bytes`
- profiled runs print `Q8KernelProfile`
- report identifies the next bottleneck using measured phase and branch data

R97 is rejected if:

- profiler output is missing
- output changes
- measurements are too incomplete to decide the next stage

## Files

- Create: `docs/benchmarks/trials/success/2026-06-16-r97-post-r96-prefill-profile.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r97-post-r96-prefill-profile.md`

## Task 1: Build Benchmark Binary

- [x] **Step 1: Build release `llama-test`**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: build passes.

## Task 2: Run Normal Controls

- [x] **Step 1: Run three normal single-thread controls**

Run:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r97-control${i}.txt" 2> "target/r97-control${i}.time"
done
```

Expected for each run:

- output contains `> No`
- metrics line contains no `Q8KernelProfile`
- peak transient is `1,050,673,152 bytes`

## Task 3: Run Profiled Trials

- [x] **Step 1: Run two Q8 profiled trials**

Run:

```bash
for i in 1 2; do
  RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r97-profile${i}.txt" 2> "target/r97-profile${i}.time"
done
```

Expected for each run:

- output contains `> No`
- metrics line contains `Q8KernelProfile`
- top profile row is captured

## Task 4: Extract Evidence

- [x] **Step 1: Extract summary rows**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in [*sorted(Path("target").glob("r97-control*.txt")), *sorted(Path("target").glob("r97-profile*.txt"))]:
    text = path.read_text(errors="replace")
    metric = next((line for line in text.splitlines() if "TTFT/Prefill:" in line), "")
    print(f"--- {path.name} ---")
    print("answer No:", "> No" in text)
    print(metric)
for path in [*sorted(Path("target").glob("r97-control*.time")), *sorted(Path("target").glob("r97-profile*.time"))]:
    text = path.read_text(errors="replace")
    print(f"--- {path.name} ---")
    print("\\n".join(line.strip() for line in text.splitlines() if "real" in line or "maximum resident set size" in line))
PY
```

Expected: enough data to populate the report table.

## Task 5: Report and Commit

- [x] **Step 1: Write R97 report**

Report must include:

- normal control table
- profiled trial table
- Q8 branch top rows
- comparison to R96 accepted result
- conclusion for R98 target

- [x] **Step 2: Update benchmark index**

Add one row for `2026-06-16-r97-post-r96-prefill-profile.md`.

- [x] **Step 3: Final verification**

Run:

```bash
cargo fmt --check
git diff --check
```

Expected: both pass.

- [x] **Step 4: Commit**

Run:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r97-post-r96-prefill-profile.md docs/superpowers/plans/2026-06-16-r97-post-r96-prefill-profile.md
git commit -m "bench(runtime): profile post-r96 q8 prefill bottleneck"
```
