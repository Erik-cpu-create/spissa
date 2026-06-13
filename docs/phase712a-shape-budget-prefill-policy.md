# Phase 7.12A Generic Shape/Budget-Aware RAMA Prefill Policy

Phase 7.12A turns the previous measured per-model prefill-window recommendations into a generic runtime policy. The goal is to keep the RAMA CLI default family-aware without hardcoding model names such as Pythia-70M or Pythia-160M.

## Scope

Implemented:

```text
rllm run --rama-prefill-policy low-ram|speed
```

Default behavior:

```text
--rama-prefill-policy low-ram
```

Existing manual controls remain:

```text
--rama-prefill-chunk-tokens <n>   # fixed explicit override
--no-rama-prefill-chunking        # full-prompt prefill / reproduction mode
```

## Policy shape

The policy uses the prepared GPT-NeoX/Pythia runtime config rather than the model name.

Inputs:

```text
num_layers
hidden_size = num_heads * head_dim
intermediate_size
prompt token count
explicit transient memory budget, when provided
```

Low-RAM policy:

```text
base window = 32 tokens
hidden baseline = 512
layer baseline = 6
candidate = round_up_power_of_two(max(32, 32 * hidden/512, 32 * layers/6))
candidate clamp = 32..128
candidate = min(candidate, prompt_len)
```

Speed policy:

```text
candidate = min(low_ram_candidate * 2, 256)
candidate = min(candidate, prompt_len)
```

Budget downshift:

```text
estimated transient bytes ~= chunk_tokens * (intermediate_size + 7 * hidden_size) * sizeof(f32)
while estimate > explicit_memory_budget:
    chunk_tokens /= 2
```

This keeps the previous measured behavior generic:

| shape | low-ram policy | speed policy |
|---|---:|---:|
| Pythia-70M-like: 6 layers, hidden 512, intermediate 2048 | 32 | 64 |
| Pythia-160M-like: 12 layers, hidden 768, intermediate 3072 | 64 | 128 |

## Why this phase exists

Phase 7.10B measured 32 real input tokens as the best default for local Pythia-70M after row-span optimization. Phase 7.11B then showed Pythia-160M benefits from larger windows: 64 for low-RAM-safe runs and 128 for speed-biased runs. Phase 7.12A generalizes that result by tying the default to model shape and explicit transient budget instead of adding any Pythia-160M special case.

## Runtime behavior

The CLI now prints the selected window before generation:

```text
RAMA prefill window: 64 token(s) (auto low-ram policy)
RAMA prefill window: 128 token(s) (auto speed policy)
RAMA prefill window: 64 token(s) (fixed override)
RAMA prefill window: disabled; full prompt prefill
```

The actual generation path remains unchanged after selection: the selected value is still passed into `GptNeoxRamaGenerationOptions::prefill_chunk_tokens`, so chunked prefill semantics remain the same as Phase 7.9E/7.10B.

## Verification

Targeted tests added:

```text
gpt_neox::tests::recommended_rama_prefill_policy_selects_shape_aware_windows
gpt_neox::tests::recommended_rama_prefill_policy_respects_prompt_len_and_budget
commands::run::tests::parse_rama_prefill_policy_accepts_low_ram_and_speed
commands::run::tests::parse_rama_prefill_policy_rejects_unknown_values
commands::run::tests::effective_rama_prefill_chunk_tokens_uses_auto_policy_by_shape
commands::run::tests::effective_rama_prefill_chunk_tokens_honors_fixed_and_disabled_modes
```

Commands run during implementation:

```bash
cargo test -p rllm-cli run::tests:: -- --nocapture
cargo test -p rllm-runtime recommended_rama_prefill_policy -- --nocapture
cargo fmt
```

Result:

```text
rllm-cli run tests: 8 passed
rllm-runtime policy tests: 2 passed
```

## Remaining work

Phase 7.12A chooses smarter defaults; it does not make the MLP/QKV kernels faster. The next performance slice should optimize the remaining measured Pythia-160M MLP/QKV projection bottlenecks, still using timing evidence rather than model-specific hacks.
