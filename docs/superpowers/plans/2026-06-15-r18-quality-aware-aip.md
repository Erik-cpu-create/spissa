# R18 Quality-Aware AIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RLLM AIP policy routing, quality-aware sparse MLP selection, and repetition telemetry while keeping exact-lowram inference unchanged.

**Architecture:** Keep the R17 sparse kernels as the execution primitive, but move the mode decision into an RLLM-native AIP policy in `speed.rs`. LLaMA session code passes layer context into each transformer block, generation code asks the policy before using AIP, and session/CLI metrics report both AIP activity and generated-token repetition. Benchmark documentation records R18 as approximate experimental-speed research with explicit related-work boundaries.

**Tech Stack:** Rust workspace, `rllm-runtime`, `rllm-cli`, existing LLaMA streaming path, existing raw BF16/FP16 sparse kernels, `cargo fmt`, `cargo clippy`, `cargo test`, local benchmark reports under `docs/benchmarks/trials`.

---

## File Structure

- Modify `crates/rllm-runtime/src/speed.rs`
  - Owns AIP env parsing, policy enum, projection decision logic, compatibility parsing for `RLLM_TURBO_TOPK`, sparse projection telemetry, and top-k selection.
- Modify `crates/rllm-runtime/src/lib.rs`
  - Re-exports new AIP policy/decision/repetition types used by runtime tests and CLI formatting tests.
- Modify `crates/rllm-runtime/src/models/llama/generate.rs`
  - Adds layer index/total layer count to `LlamaStreamingBlockConfig` and gates `mlp_gate_up` / `mlp_down` sparse calls through the AIP policy.
- Modify `crates/rllm-runtime/src/models/llama/session.rs`
  - Passes layer context into the block config and updates test-only experimental config constructors.
- Modify `crates/rllm-runtime/src/models/llama/api.rs`
  - Supplies layer context for the non-session LLaMA generation path with experimental speed disabled.
- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Updates sparse kernel tests to use the new `aip_topk` config field.
- Modify `crates/rllm-runtime/src/session.rs`
  - Adds `RamaRepetitionStats`, computes it for every generated turn, and exposes it in `RamaSessionTurnMetrics`.
- Modify `crates/rllm-cli/src/bin/llama-test.rs`
  - Renames the printed experimental-speed suffix to `AIP` and prints repetition telemetry.
- Modify `crates/rllm-cli/src/commands/chat_session_token.rs`
  - Renames report note fields to AIP names and adds repetition telemetry to the benchmark notes.
- Create `docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md`
  - Records planned R18 commands and explicit `not-measured` result rows before benchmark execution.
- Modify `docs/benchmarks/trials/index.md`
  - Adds an active R18 row after implementation, then updates folder/status after measured evidence.

## Task 1: AIP Config, Policy, And Sparse Telemetry

**Files:**
- Modify: `crates/rllm-runtime/src/speed.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Add failing tests for AIP parsing and policy decisions**

In `crates/rllm-runtime/src/speed.rs`, extend the existing `#[cfg(test)] mod tests` with:

```rust
#[test]
fn parse_aip_policy_accepts_quality_and_speed() {
    assert_eq!(parse_aip_policy(Some("quality")), Some(RamaAipPolicyKind::Quality));
    assert_eq!(parse_aip_policy(Some("speed")), Some(RamaAipPolicyKind::Speed));
    assert_eq!(parse_aip_policy(Some(" QUALITY ")), Some(RamaAipPolicyKind::Quality));
    assert_eq!(parse_aip_policy(Some("bad")), None);
    assert_eq!(parse_aip_policy(None), None);
}

#[test]
fn parse_aip_topk_keeps_only_positive_values() {
    assert_eq!(parse_aip_topk(Some("128")), Some(128));
    assert_eq!(parse_aip_topk(Some("1")), Some(1));
    assert_eq!(parse_aip_topk(Some("0")), None);
    assert_eq!(parse_aip_topk(Some("-2")), None);
    assert_eq!(parse_aip_topk(Some("bad")), None);
    assert_eq!(parse_aip_topk(None), None);
}

#[test]
fn quality_policy_uses_only_middle_layer_gate_up() {
    let config = RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: RamaAipPolicyKind::Quality,
        aip_topk: Some(96),
    };

    assert_eq!(
        config.aip_decision_for_projection(0, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::exact()
    );
    assert_eq!(
        config.aip_decision_for_projection(7, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::exact()
    );
    assert_eq!(
        config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::aip(96)
    );
    assert_eq!(
        config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpDown, 8192, 512),
        RamaAipProjectionDecision::exact()
    );
}

#[test]
fn quality_policy_stays_exact_for_tiny_layer_counts() {
    let config = RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: RamaAipPolicyKind::Quality,
        aip_topk: Some(64),
    };

    assert_eq!(
        config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::exact()
    );
    assert_eq!(
        config.aip_decision_for_projection(1, 3, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::exact()
    );
}

#[test]
fn speed_policy_uses_aip_for_gate_up_and_down() {
    let config = RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: RamaAipPolicyKind::Speed,
        aip_topk: Some(128),
    };

    assert_eq!(
        config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpGateUp, 2048, 256),
        RamaAipProjectionDecision::aip(128)
    );
    assert_eq!(
        config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpDown, 8192, 512),
        RamaAipProjectionDecision::aip(128)
    );
}

#[test]
fn disabled_config_always_selects_exact() {
    let config = RamaExperimentalSpeedConfig::disabled();

    assert_eq!(
        config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
        RamaAipProjectionDecision::exact()
    );
}

#[test]
fn stats_record_policy_without_losing_sparse_counts() {
    let mut stats = RamaExperimentalSpeedStats::default();
    stats.record_aip_policy(RamaAipPolicyKind::Quality);
    stats.record_sparse_projection(4, 16, 3, 2);

    let mut other = RamaExperimentalSpeedStats::default();
    other.record_aip_policy(RamaAipPolicyKind::Speed);
    other.record_exact_fallback();
    stats.add_assign(other);

    assert_eq!(stats.aip_policy, Some(RamaAipPolicyKind::Quality));
    assert_eq!(stats.sparse_projection_calls, 1);
    assert_eq!(stats.exact_fallbacks, 1);
    assert!(!stats.is_empty());
}
```

Also update existing tests in the same file so every `RamaExperimentalSpeedConfig` literal uses:

```rust
RamaExperimentalSpeedConfig {
    enabled: true,
    aip_policy: RamaAipPolicyKind::Speed,
    aip_topk: Some(512),
}
```

and:

```rust
RamaExperimentalSpeedConfig {
    enabled: true,
    aip_policy: RamaAipPolicyKind::Speed,
    aip_topk: None,
}
```

- [ ] **Step 2: Run targeted tests and verify they fail**

Run:

```bash
cargo test -p rllm-runtime speed -- --nocapture
```

Expected: compile failure naming missing `RamaAipPolicyKind`, `RamaAipProjectionKind`, `RamaAipProjectionDecision`, `parse_aip_policy`, `parse_aip_topk`, `aip_decision_for_projection`, or `record_aip_policy`.

- [ ] **Step 3: Implement AIP policy types and env parsing**

In `crates/rllm-runtime/src/speed.rs`, replace the config section above `RamaExperimentalSpeedStats` with:

```rust
pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_AIP_POLICY_ENV: &str = "RLLM_AIP_POLICY";
pub const RLLM_AIP_TOPK_ENV: &str = "RLLM_AIP_TOPK";
pub const RLLM_TURBO_TOPK_ENV: &str = "RLLM_TURBO_TOPK";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RamaAipPolicyKind {
    #[default]
    Quality,
    Speed,
}

impl RamaAipPolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Speed => "speed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamaAipProjectionKind {
    MlpGateUp,
    MlpDown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaAipProjectionDecision {
    pub enabled: bool,
    pub topk: usize,
}

impl RamaAipProjectionDecision {
    pub fn exact() -> Self {
        Self {
            enabled: false,
            topk: 0,
        }
    }

    pub fn aip(topk: usize) -> Self {
        Self {
            enabled: true,
            topk,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedConfig {
    pub enabled: bool,
    pub aip_policy: RamaAipPolicyKind,
    pub aip_topk: Option<usize>,
}

impl RamaExperimentalSpeedConfig {
    pub fn from_env() -> Self {
        let aip_topk = parse_aip_topk(std::env::var(RLLM_AIP_TOPK_ENV).ok().as_deref())
            .or_else(|| parse_turbo_topk(std::env::var(RLLM_TURBO_TOPK_ENV).ok().as_deref()));

        Self {
            enabled: parse_experimental_speed_enabled(
                std::env::var(RLLM_EXPERIMENTAL_SPEED_ENV).ok().as_deref(),
            ),
            aip_policy: parse_aip_policy(std::env::var(RLLM_AIP_POLICY_ENV).ok().as_deref())
                .unwrap_or_default(),
            aip_topk,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: None,
        }
    }

    pub fn topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        self.aip_topk
            .unwrap_or(default_topk.max(1))
            .min(input_len)
            .max(1)
    }

    pub fn aip_decision_for_projection(
        self,
        layer_index: usize,
        total_layers: usize,
        projection: RamaAipProjectionKind,
        input_len: usize,
        default_topk: usize,
    ) -> RamaAipProjectionDecision {
        if !self.enabled || input_len == 0 || layer_index >= total_layers {
            return RamaAipProjectionDecision::exact();
        }

        match self.aip_policy {
            RamaAipPolicyKind::Speed => RamaAipProjectionDecision::aip(
                self.topk_for_input(input_len, default_topk),
            ),
            RamaAipPolicyKind::Quality => {
                if projection != RamaAipProjectionKind::MlpGateUp
                    || !quality_policy_allows_layer(layer_index, total_layers)
                {
                    return RamaAipProjectionDecision::exact();
                }
                RamaAipProjectionDecision::aip(self.topk_for_input(input_len, default_topk))
            }
        }
    }
}

fn quality_policy_allows_layer(layer_index: usize, total_layers: usize) -> bool {
    if total_layers < 4 || layer_index >= total_layers {
        return false;
    }
    let exact_edge_layers = total_layers / 4;
    layer_index >= exact_edge_layers
        && layer_index < total_layers.saturating_sub(exact_edge_layers)
}
```

Add the parsers near `parse_turbo_topk`:

```rust
pub fn parse_aip_policy(value: Option<&str>) -> Option<RamaAipPolicyKind> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("quality") => Some(RamaAipPolicyKind::Quality),
        Some("speed") => Some(RamaAipPolicyKind::Speed),
        _ => None,
    }
}

pub fn parse_aip_topk(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}
```

Keep `parse_turbo_topk` as a compatibility wrapper:

```rust
pub fn parse_turbo_topk(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}
```

- [ ] **Step 4: Add policy name to sparse telemetry**

In `RamaExperimentalSpeedStats`, add the field:

```rust
pub aip_policy: Option<RamaAipPolicyKind>,
```

Update `add_assign` so the first observed policy is preserved:

```rust
if self.aip_policy.is_none() {
    self.aip_policy = other.aip_policy;
}
```

Add this method:

```rust
pub fn record_aip_policy(&mut self, policy: RamaAipPolicyKind) {
    if self.aip_policy.is_none() {
        self.aip_policy = Some(policy);
    }
}
```

Update `is_empty` to include:

```rust
&& self.aip_policy.is_none()
```

Update every `RamaExperimentalSpeedStats { ... }` literal in runtime and CLI tests to include:

```rust
aip_policy: Some(rllm_runtime::RamaAipPolicyKind::Speed),
```

or use:

```rust
..Default::default()
```

when the policy value is irrelevant.

- [ ] **Step 5: Export the new AIP types**

In `crates/rllm-runtime/src/lib.rs`, update the `pub use speed::{ ... }` block to include:

```rust
parse_aip_policy, parse_aip_topk, RamaAipPolicyKind, RamaAipProjectionDecision,
RamaAipProjectionKind,
```

Keep `parse_turbo_topk` exported for compatibility.

- [ ] **Step 6: Run targeted tests and verify pass**

Run:

```bash
cargo test -p rllm-runtime speed -- --nocapture
```

Expected: all `speed` tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add crates/rllm-runtime/src/speed.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): add quality-aware aip policy"
```

Expected: commit succeeds with only `speed.rs` and `lib.rs` staged.

## Task 2: Route LLaMA MLP Through AIP Policy

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/generate.rs`
- Modify: `crates/rllm-runtime/src/models/llama/session.rs`
- Modify: `crates/rllm-runtime/src/models/llama/api.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] **Step 1: Update tests to express quality-vs-speed routing**

In `crates/rllm-runtime/src/models/llama/session.rs`, add this helper near `write_bf16_mlp_speed_model`:

```rust
fn write_bf16_mlp_speed_model_with_layers(
    path: &std::path::Path,
    vocab_size: usize,
    layer_count: usize,
) {
    let mut writer = RllmWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
    let mut tensor_id = 0u64;
    add_f32_tensor(
        &mut writer,
        tensor_id,
        "model.embed_tokens.weight",
        vec![vocab_size as u64, HIDDEN_SIZE as u64],
        &vec![0.0; vocab_size * HIDDEN_SIZE],
    );
    tensor_id += 1;
    add_bf16_tensor(
        &mut writer,
        tensor_id,
        "lm_head.weight",
        vec![vocab_size as u64, HIDDEN_SIZE as u64],
        &vec![0x0000; vocab_size * HIDDEN_SIZE],
    );
    tensor_id += 1;
    for layer_idx in 0..layer_count {
        add_layer_norms(&mut writer, &mut tensor_id, layer_idx);
        let prefix = format!("model.layers.{layer_idx}");
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            &[0x0000; HIDDEN_SIZE * INTERMEDIATE_SIZE],
        );
        tensor_id += 1;
    }
    writer.finalize().unwrap();
}
```

Then add this test near `llama_session_reports_experimental_speed_stats_when_enabled_for_test`:

```rust
#[test]
fn llama_session_quality_policy_uses_fewer_aip_calls_than_speed_policy() {
    let path = temp_path("aip-quality-vs-speed");
    write_bf16_mlp_speed_model_with_layers(&path, 8, 4);

    let mut quality_model = LazyRllmModel::open(&path).unwrap();
    let quality_prepared = prepared_with_layers(4);
    let mut quality_budget = MemoryBudget::unbounded();
    let mut quality_adapter =
        LlamaRamaSessionAdapter::new(&mut quality_model, &quality_prepared, &mut quality_budget)
            .unwrap();
    quality_adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: crate::RamaAipPolicyKind::Quality,
        aip_topk: Some(1),
    });
    quality_adapter
        .append_tokens(&[0], &mut quality_budget, true)
        .unwrap();
    let quality_stats = quality_adapter.take_last_experimental_speed_stats().unwrap();

    let mut speed_model = LazyRllmModel::open(&path).unwrap();
    let speed_prepared = prepared_with_layers(4);
    let mut speed_budget = MemoryBudget::unbounded();
    let mut speed_adapter =
        LlamaRamaSessionAdapter::new(&mut speed_model, &speed_prepared, &mut speed_budget).unwrap();
    speed_adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: crate::RamaAipPolicyKind::Speed,
        aip_topk: Some(1),
    });
    speed_adapter
        .append_tokens(&[0], &mut speed_budget, true)
        .unwrap();
    let speed_stats = speed_adapter.take_last_experimental_speed_stats().unwrap();

    assert_eq!(quality_stats.aip_policy, Some(crate::RamaAipPolicyKind::Quality));
    assert_eq!(speed_stats.aip_policy, Some(crate::RamaAipPolicyKind::Speed));
    assert!(quality_stats.sparse_projection_calls > 0);
    assert!(quality_stats.sparse_projection_calls < speed_stats.sparse_projection_calls);
    assert_eq!(quality_stats.max_selected_topk, 1);
    assert_eq!(speed_stats.max_selected_topk, 1);

    std::fs::remove_file(path).ok();
}
```

Update the existing `llama_session_reports_experimental_speed_stats_when_enabled_for_test` config literal to:

```rust
crate::RamaExperimentalSpeedConfig {
    enabled: true,
    aip_policy: crate::RamaAipPolicyKind::Speed,
    aip_topk: Some(1),
}
```

In `crates/rllm-runtime/src/streaming/tests.rs`, update sparse-kernel config literals to use `aip_policy` and `aip_topk`:

```rust
crate::RamaExperimentalSpeedConfig {
    enabled: true,
    aip_policy: crate::RamaAipPolicyKind::Speed,
    aip_topk: Some(2),
}
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run:

```bash
cargo test -p rllm-runtime llama_session_quality_policy_uses_fewer_aip_calls_than_speed_policy -- --nocapture
```

Expected: compile failure because `LlamaStreamingBlockConfig` does not yet carry layer context and generation code still calls sparse kernels directly from `config.experimental_speed.enabled`.

- [ ] **Step 3: Add layer context to the LLaMA block config**

In `crates/rllm-runtime/src/models/llama/generate.rs`, add fields to `LlamaStreamingBlockConfig`:

```rust
pub layer_index: usize,
pub total_layers: usize,
```

In `crates/rllm-runtime/src/models/llama/session.rs`, when constructing `LlamaStreamingBlockConfig`, add:

```rust
layer_index: i,
total_layers: self.prepared.layers.len(),
```

In `crates/rllm-runtime/src/models/llama/api.rs`, when constructing `LlamaStreamingBlockConfig`, add:

```rust
layer_index: i,
total_layers: prepared.layers.len(),
```

- [ ] **Step 4: Replace direct sparse gating with AIP decisions**

In `crates/rllm-runtime/src/models/llama/generate.rs`, add `RamaAipProjectionKind` to the imports from `crate`:

```rust
RamaAipProjectionKind,
```

Immediately before the `sparse_gate_up` block, add:

```rust
let gate_up_aip_decision = config.experimental_speed.aip_decision_for_projection(
    config.layer_index,
    config.total_layers,
    RamaAipProjectionKind::MlpGateUp,
    config.hidden_size,
    128,
);
```

Replace the `sparse_gate_up` condition with:

```rust
let sparse_gate_up = if gate_up_aip_decision.enabled {
    if let Some(stats) = &mut experimental_speed_stats {
        stats.record_aip_policy(config.experimental_speed.aip_policy);
        let sparse_config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: config.experimental_speed.aip_policy,
            aip_topk: Some(gate_up_aip_decision.topk),
        };
        streaming_sparse_silu_gate_up_from_model(
            model,
            &names.gate_weight,
            &names.up_weight,
            &mlp_input,
            mlp_config,
            sparse_config,
            stats,
            budget,
        )?
    } else {
        None
    }
} else {
    None
};
```

Immediately before the `sparse_down` block, add:

```rust
let down_aip_decision = config.experimental_speed.aip_decision_for_projection(
    config.layer_index,
    config.total_layers,
    RamaAipProjectionKind::MlpDown,
    config.intermediate_size,
    512,
);
```

Replace the `sparse_down` condition with:

```rust
let sparse_down = if down_aip_decision.enabled {
    if let Some(stats) = &mut experimental_speed_stats {
        stats.record_aip_policy(config.experimental_speed.aip_policy);
        let sparse_config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: config.experimental_speed.aip_policy,
            aip_topk: Some(down_aip_decision.topk),
        };
        streaming_sparse_tile_linear_from_model(
            model,
            &names.down_weight,
            &gate,
            None,
            down_config,
            sparse_config,
            stats,
            budget,
        )?
    } else {
        None
    }
} else {
    None
};
```

This keeps policy-exact projections out of `exact_fallbacks`; fallbacks remain reserved for unsupported AIP attempts.

- [ ] **Step 5: Run targeted runtime tests**

Run:

```bash
cargo test -p rllm-runtime llama_session_reports_experimental_speed_stats_when_enabled_for_test -- --nocapture
cargo test -p rllm-runtime llama_session_quality_policy_uses_fewer_aip_calls_than_speed_policy -- --nocapture
cargo test -p rllm-runtime sparse_ -- --nocapture
```

Expected: all targeted tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add crates/rllm-runtime/src/models/llama/generate.rs crates/rllm-runtime/src/models/llama/session.rs crates/rllm-runtime/src/models/llama/api.rs crates/rllm-runtime/src/streaming/tests.rs
git commit -m "feat(runtime): route llama mlp through aip policy"
```

Expected: commit succeeds with only LLaMA routing and sparse test updates staged.

## Task 3: Generated-Token Repetition Telemetry

**Files:**
- Modify: `crates/rllm-runtime/src/session.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Add failing repetition metric tests**

In `crates/rllm-runtime/src/session.rs`, add these tests inside the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn repetition_stats_from_tokens_detects_runs_and_unique_count() {
    let stats = RamaRepetitionStats::from_tokens(&[7, 7, 7, 8, 9, 9]);

    assert_eq!(stats.generated_tokens, 6);
    assert_eq!(stats.unique_generated_tokens, 3);
    assert_eq!(stats.max_repeated_token_run, 3);
    assert!((stats.repeated_token_ratio - 0.6).abs() < f64::EPSILON);
}

#[test]
fn repetition_stats_from_empty_tokens_are_zero() {
    let stats = RamaRepetitionStats::from_tokens(&[]);

    assert_eq!(stats.generated_tokens, 0);
    assert_eq!(stats.unique_generated_tokens, 0);
    assert_eq!(stats.max_repeated_token_run, 0);
    assert_eq!(stats.repeated_token_ratio, 0.0);
}

#[test]
fn turn_metrics_include_repetition_stats() {
    let mut adapter = RecordingAdapter::new(16);
    adapter.steps = vec![
        Ok(Some(RamaSessionStep {
            token_id: 7,
            logits: None,
            cached_context_len_after: 1,
        })),
        Ok(Some(RamaSessionStep {
            token_id: 7,
            logits: None,
            cached_context_len_after: 2,
        })),
        Ok(Some(RamaSessionStep {
            token_id: 8,
            logits: None,
            cached_context_len_after: 3,
        })),
        Ok(Some(RamaSessionStep {
            token_id: 8,
            logits: None,
            cached_context_len_after: 4,
        })),
    ];
    let mut session = RamaChatSession::new(adapter);
    let mut budget = MemoryBudget::unbounded();

    let result = session
        .generate_turn(&[1], 4, &mut budget, |_| true)
        .unwrap();

    assert_eq!(result.generated_token_ids, [7, 7, 8, 8]);
    assert_eq!(result.metrics.repetition_stats.generated_tokens, 4);
    assert_eq!(result.metrics.repetition_stats.unique_generated_tokens, 2);
    assert_eq!(result.metrics.repetition_stats.max_repeated_token_run, 2);
    assert!(
        (result.metrics.repetition_stats.repeated_token_ratio - (2.0 / 3.0)).abs()
            < f64::EPSILON
    );
}
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run:

```bash
cargo test -p rllm-runtime repetition_stats -- --nocapture
```

Expected: compile failure because `RamaRepetitionStats` and `metrics.repetition_stats` do not exist.

- [ ] **Step 3: Add `RamaRepetitionStats`**

In `crates/rllm-runtime/src/session.rs`, add this struct before `RamaSessionTurnMetrics`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RamaRepetitionStats {
    pub generated_tokens: usize,
    pub unique_generated_tokens: usize,
    pub max_repeated_token_run: usize,
    pub repeated_token_ratio: f64,
}

impl RamaRepetitionStats {
    pub fn from_tokens(tokens: &[usize]) -> Self {
        if tokens.is_empty() {
            return Self::default();
        }

        let mut unique = std::collections::HashSet::new();
        let mut max_run = 1usize;
        let mut current_run = 1usize;
        let mut adjacent_repeats = 0usize;
        unique.insert(tokens[0]);

        for window in tokens.windows(2) {
            unique.insert(window[1]);
            if window[0] == window[1] {
                adjacent_repeats = adjacent_repeats.saturating_add(1);
                current_run = current_run.saturating_add(1);
                max_run = max_run.max(current_run);
            } else {
                current_run = 1;
            }
        }

        let denominator = tokens.len().saturating_sub(1);
        let repeated_token_ratio = if denominator == 0 {
            0.0
        } else {
            adjacent_repeats as f64 / denominator as f64
        };

        Self {
            generated_tokens: tokens.len(),
            unique_generated_tokens: unique.len(),
            max_repeated_token_run: max_run,
            repeated_token_ratio,
        }
    }
}
```

Add this field to `RamaSessionTurnMetrics`:

```rust
pub repetition_stats: RamaRepetitionStats,
```

- [ ] **Step 4: Populate repetition stats in all turn result paths**

In the early stop path where only the first generated token is returned, add before `return Ok(self.turn_result(...))`:

```rust
let repetition_stats = RamaRepetitionStats::from_tokens(&generated_token_ids);
```

and set the metrics field:

```rust
repetition_stats,
```

In the normal path before `Ok(self.turn_result(...))`, add:

```rust
let repetition_stats = RamaRepetitionStats::from_tokens(&generated_token_ids);
```

and set the metrics field:

```rust
repetition_stats,
```

Update any test literal of `RamaSessionTurnMetrics` outside these constructors by adding:

```rust
repetition_stats: RamaRepetitionStats::default(),
```

when the test does not inspect repetition.

- [ ] **Step 5: Export repetition stats**

In `crates/rllm-runtime/src/lib.rs`, add `RamaRepetitionStats` to the `pub use chat_session::{ ... }` block.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
cargo test -p rllm-runtime repetition_stats -- --nocapture
cargo test -p rllm-runtime turn_metrics_include_repetition_stats -- --nocapture
```

Expected: all targeted repetition tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add crates/rllm-runtime/src/session.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): report generated token repetition"
```

Expected: commit succeeds with only session metrics and export changes staged.

## Task 4: CLI And Report Telemetry Labels

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`

- [ ] **Step 1: Update `llama-test` formatting tests first**

In `crates/rllm-cli/src/bin/llama-test.rs`, rename `format_experimental_speed_suffix` to `format_aip_suffix` in tests and production. Replace the nonzero suffix test with:

```rust
#[test]
fn aip_suffix_reports_nonzero_activity() {
    let suffix = format_aip_suffix(rllm_runtime::RamaExperimentalSpeedStats {
        aip_policy: Some(rllm_runtime::RamaAipPolicyKind::Quality),
        sparse_projection_calls: 4,
        exact_fallbacks: 1,
        selected_topk_sum: 256,
        max_selected_topk: 128,
        estimated_skipped_madds: 2048,
        peak_scratch_bytes: 512,
    });

    assert!(suffix.contains("AIP: policy=quality"));
    assert!(suffix.contains("calls=4"));
    assert!(suffix.contains("fallbacks=1"));
    assert!(suffix.contains("max_topk=128"));
    assert!(suffix.contains("skipped_madds=2048"));
}

#[test]
fn repetition_suffix_reports_nonzero_activity() {
    let suffix = format_repetition_suffix(rllm_runtime::RamaRepetitionStats {
        generated_tokens: 6,
        unique_generated_tokens: 3,
        max_repeated_token_run: 3,
        repeated_token_ratio: 0.6,
    });

    assert!(suffix.contains("Repetition: unique=3/6"));
    assert!(suffix.contains("max_run=3"));
    assert!(suffix.contains("adjacent_ratio=0.60"));
}
```

Keep the empty suffix test and rename it to:

```rust
#[test]
fn aip_suffix_is_empty_without_activity() {
    assert_eq!(
        format_aip_suffix(rllm_runtime::RamaExperimentalSpeedStats::default()),
        ""
    );
}
```

- [ ] **Step 2: Run `llama-test` formatting tests and verify failure**

Run:

```bash
cargo test -p rllm-cli --bin llama-test aip_suffix -- --nocapture
```

Expected: compile failure because `format_aip_suffix` and `format_repetition_suffix` are not implemented yet.

- [ ] **Step 3: Implement `llama-test` AIP and repetition formatting**

In `crates/rllm-cli/src/bin/llama-test.rs`, replace the old formatter with:

```rust
fn format_aip_suffix(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        let policy = stats
            .aip_policy
            .map(|policy| policy.as_str())
            .unwrap_or("unknown");
        format!(
            " | AIP: policy={} calls={} fallbacks={} max_topk={} skipped_madds={} scratch={} bytes",
            policy,
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        )
    }
}

fn format_repetition_suffix(stats: rllm_runtime::RamaRepetitionStats) -> String {
    if stats.generated_tokens == 0 {
        String::new()
    } else {
        format!(
            " | Repetition: unique={}/{} max_run={} adjacent_ratio={:.2}",
            stats.unique_generated_tokens,
            stats.generated_tokens,
            stats.max_repeated_token_run,
            stats.repeated_token_ratio
        )
    }
}
```

Replace the metrics suffix construction with:

```rust
let aip_suffix = format_aip_suffix(result.metrics.experimental_speed_stats);
let repetition_suffix = format_repetition_suffix(result.metrics.repetition_stats);
```

and include both in the metrics line:

```rust
rolling_suffix,
aip_suffix,
repetition_suffix
```

- [ ] **Step 4: Update `chat_session_token` report formatting tests**

In `crates/rllm-cli/src/commands/chat_session_token.rs`, rename `format_experimental_speed_note` to `format_aip_note` and add:

```rust
fn format_repetition_note(stats: rllm_runtime::RamaRepetitionStats) -> String {
    if stats.generated_tokens == 0 {
        String::new()
    } else {
        format!(
            " repetition_unique={}/{} repetition_max_run={} repetition_adjacent_ratio={:.2}",
            stats.unique_generated_tokens,
            stats.generated_tokens,
            stats.max_repeated_token_run,
            stats.repeated_token_ratio
        )
    }
}
```

Replace the nonzero AIP test with:

```rust
#[test]
fn format_aip_note_reports_nonzero_stats() {
    let note = format_aip_note(rllm_runtime::RamaExperimentalSpeedStats {
        aip_policy: Some(rllm_runtime::RamaAipPolicyKind::Speed),
        sparse_projection_calls: 4,
        exact_fallbacks: 1,
        selected_topk_sum: 256,
        max_selected_topk: 128,
        estimated_skipped_madds: 2048,
        peak_scratch_bytes: 512,
    });

    assert!(note.contains("aip_policy=speed"));
    assert!(note.contains("aip_calls=4"));
    assert!(note.contains("aip_fallbacks=1"));
    assert!(note.contains("aip_max_topk=128"));
}

#[test]
fn format_repetition_note_reports_nonzero_stats() {
    let note = format_repetition_note(rllm_runtime::RamaRepetitionStats {
        generated_tokens: 6,
        unique_generated_tokens: 3,
        max_repeated_token_run: 3,
        repeated_token_ratio: 0.6,
    });

    assert!(note.contains("repetition_unique=3/6"));
    assert!(note.contains("repetition_max_run=3"));
    assert!(note.contains("repetition_adjacent_ratio=0.60"));
}
```

- [ ] **Step 5: Implement `chat_session_token` AIP note and report wiring**

Replace the old note formatter with:

```rust
fn format_aip_note(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        let policy = stats
            .aip_policy
            .map(|policy| policy.as_str())
            .unwrap_or("unknown");
        format!(
            " aip_policy={} aip_calls={} aip_fallbacks={} aip_max_topk={} aip_skipped_madds={} aip_scratch_bytes={}",
            policy,
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        )
    }
}
```

In the `phase_note` construction inside `write_report`, append:

```rust
+ &format_aip_note(row.session_result.metrics.experimental_speed_stats)
+ &format_repetition_note(row.session_result.metrics.repetition_stats);
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
cargo test -p rllm-cli --bin llama-test aip_suffix -- --nocapture
cargo test -p rllm-cli --bin llama-test repetition_suffix -- --nocapture
cargo test -p rllm-cli format_aip_note -- --nocapture
cargo test -p rllm-cli format_repetition_note -- --nocapture
```

Expected: all targeted CLI formatting tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add crates/rllm-cli/src/bin/llama-test.rs crates/rllm-cli/src/commands/chat_session_token.rs
git commit -m "feat(cli): report aip and repetition telemetry"
```

Expected: commit succeeds with only CLI formatting/report changes staged.

## Task 5: Benchmark Trial Documentation

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Create the active R18 trial report**

Create `docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md` with:

```markdown
# Trial: R18 Quality-Aware AIP

Date: 2026-06-15
Owner: RLLM
Status: running
Folder: active

## Hypothesis

RLLM AIP quality policy can keep a useful part of the R17 sparse MLP speed gain
while reducing generated-token repetition by keeping early layers, final
layers, `mlp_down`, attention, and LM-head exact.

## Scope

- Mode: experimental-speed
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: MLP projection arithmetic and memory access
- Bottleneck tag: CPU arithmetic
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- AIP policy sweep: `RLLM_AIP_POLICY=quality`, `RLLM_AIP_POLICY=speed`
- Top-k: `RLLM_AIP_TOPK=128`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Runtime context:

- build profile: release
- CPU: record from benchmark machine
- RAM: record from benchmark machine
- OS: record from benchmark machine
- relevant env/config: `RLLM_EXPERIMENTAL_SPEED`, `RLLM_AIP_POLICY`, `RLLM_AIP_TOPK`

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | AIP calls | fallbacks | max top-k | repeated ratio | max run | unique tokens | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | exact baseline | not-measured | not-measured | not-measured | not-measured | 0 | 0 | 0 | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured |
| Llama-3.2-1B-Instruct | AIP quality top-k 128 | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured |
| Llama-3.2-1B-Instruct | AIP speed top-k 128 | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured |
| SmolLM2-135M | AIP quality top-k 128 | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured | not-measured |

## Analysis

The report has commands and classification criteria. Measurements have not run
yet.

## Decision

needs follow-up

Reason: implementation and benchmark execution have not completed yet.

Paper value:

- use as limitation until measured evidence is added

## Next Experiment

Run the listed exact, AIP quality, and AIP speed commands, then classify the
trial using the success and failure criteria from the R18 spec.
```

- [ ] **Step 2: Add R18 to benchmark index as active**

In `docs/benchmarks/trials/index.md`, add this row after R17:

```markdown
| 2026-06-15 | 2026-06-15-r18-quality-aware-aip.md | active | SmolLM2-135M-raw.rllm, Llama-3.2-1B-Instruct-raw.rllm | experimental-speed | CPU arithmetic | R17 Llama 1B top-k 128 1.61 tok/s with repetition | not measured yet; quality-vs-speed AIP sweep queued | running | evidence row reserved for R18 |
```

- [ ] **Step 3: Verify docs diff**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit Task 5**

Run:

```bash
git add docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md docs/benchmarks/trials/index.md
git commit -m "docs(benchmarks): add r18 quality-aware aip trial"
```

Expected: commit succeeds with only benchmark docs staged.

## Task 6: Workspace Verification

**Files:**
- Verify all files modified by Tasks 1-5.

- [ ] **Step 1: Run formatter check**

Run:

```bash
cargo fmt --check
```

Expected: exit code 0. If formatting fails, run `cargo fmt`, inspect the diff, then run `cargo fmt --check` again.

- [ ] **Step 2: Run clippy**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: exit code 0 with no warnings.

- [ ] **Step 3: Run full tests**

Run:

```bash
cargo test --workspace
```

Expected: exit code 0 with all workspace tests passing.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short --branch
git log --oneline --decorate -6
```

Expected: branch is `codex/r18-quality-aware-aip`; working tree is clean except for changes intentionally left uncommitted by a failed verification fix. Latest commits should include the Task 1-5 commits.

## Task 7: R18 Benchmark Execution And Classification

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md`
- Modify: `docs/benchmarks/trials/index.md`
- Move after evidence: `docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md` to `docs/benchmarks/trials/success/`, `failed/`, or `inconclusive/`

- [ ] **Step 1: Build benchmark binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: release binary exists at `target/release/llama-test`.

- [ ] **Step 2: Run Llama exact baseline**

Run:

```bash
printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: command completes and prints decode speed, repetition stats, peak transient bytes, and `/usr/bin/time -l` memory stats.

- [ ] **Step 3: Run Llama AIP quality**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: command completes and prints `AIP: policy=quality`.

- [ ] **Step 4: Run Llama AIP speed**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: command completes and prints `AIP: policy=speed`.

- [ ] **Step 5: Run SmolLM2 AIP quality control**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: command completes and prints `AIP: policy=quality` if quality policy selects AIP for that model's layer count; if it prints no AIP suffix because the model has fewer than four layers, record that exact reason in the report.

- [ ] **Step 6: Fill benchmark report and classify**

Update the R18 report table with measured values. Use these decision rules:

- Move to `success` if Llama quality mode is at least 2x exact baseline, RLLM tracked peak transient memory does not materially rise, and repetition telemetry improves versus R17 top-k 128.
- Move to `failed` if quality mode falls below 2x exact baseline or memory rises enough to weaken the low-RAM claim.
- Move to `inconclusive` if measurements are noisy, the prompt output cannot be compared, or AIP routing does not activate for the intended model.

Update the `Folder`, `Status`, `Decision`, `Analysis`, and `Next Experiment` sections to match the chosen folder.

- [ ] **Step 7: Update benchmark index row**

In `docs/benchmarks/trials/index.md`, update the R18 row folder, result, decision, and paper value fields using the measured report values.

- [ ] **Step 8: Commit benchmark evidence**

Run one of these command sets depending on classification:

```bash
git mv docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md docs/benchmarks/trials/success/2026-06-15-r18-quality-aware-aip.md
git add docs/benchmarks/trials/index.md
git commit -m "docs(benchmarks): record r18 quality-aware aip trial"
```

```bash
git mv docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md docs/benchmarks/trials/failed/2026-06-15-r18-quality-aware-aip.md
git add docs/benchmarks/trials/index.md
git commit -m "docs(benchmarks): record r18 quality-aware aip trial"
```

```bash
git mv docs/benchmarks/trials/active/2026-06-15-r18-quality-aware-aip.md docs/benchmarks/trials/inconclusive/2026-06-15-r18-quality-aware-aip.md
git add docs/benchmarks/trials/index.md
git commit -m "docs(benchmarks): record r18 quality-aware aip trial"
```

Expected: benchmark report is moved to exactly one status folder and index points at the same folder.

## Final Verification

- [ ] **Step 1: Run full verification after benchmark docs are committed**

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
git status --short --branch
```

Expected: formatter, clippy, and tests all pass; git status shows a clean `codex/r18-quality-aware-aip` branch.

- [ ] **Step 2: Summarize outcome**

Report:

- final Llama exact baseline decode tok/s
- final Llama AIP quality decode tok/s
- final Llama AIP speed decode tok/s
- repetition ratio and max run for each Llama variant
- RLLM peak transient memory for each Llama variant
- benchmark folder classification
- residual risk, especially whether output quality improved enough for a paper claim
