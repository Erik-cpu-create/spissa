---
name: modular-code-guard
description: Keep rllm modular and prevent spaghetti code. Use when adding or refactoring Rust/Python code, when a file or function is growing large, when deciding where new code should live, or when reviewing a diff for structural smells (god modules, long functions, tangled imports). Pairs with an automatic edit/commit hook that warns on oversized files/functions and circular imports.
---

# Modular Code Guard

Guardrails that keep `rllm` modular as it grows. The goal is simple: **new code should be easy to find, easy to delete, and easy to test in isolation.** This skill is the human-readable policy; an automatic hook (`.claude/hooks/modular_code_guard.py`) enforces the cheap, mechanical parts on every edit and before every commit.

## When to apply

- Before writing new code: decide *where* it belongs before *how* it works (see "Placement" below).
- When the hook warns that a file/function is oversized — treat the warning as a prompt to split, not a number to suppress.
- When reviewing a diff: run the checklist.
- When something feels hard to change: that friction is the spaghetti signal.

## Thresholds (enforced by the hook)

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Source file length | **> 600 lines** | A file you can't hold in your head invites tangled edits. Test files (`tests.rs`, `tests_*.rs`, `*/tests/*`) are exempt — fixtures are allowed to be long. |
| Function length | **> 100 lines** | A function past ~100 lines is usually doing several jobs; extract helpers. |
| Circular imports | any cycle | See "Imports & cycles" — Python cycles are flagged; Rust intra-crate cycles are legal and intentionally *not* flagged. |

Thresholds live at the top of `.claude/hooks/modular_code_guard.py` (`MAX_FILE_LINES`, `MAX_FN_LINES`). Tune them there, not by sprinkling allow-comments.

These are **warnings, never blocks** — they nudge, they don't stop you committing. A deliberate exception is fine; an accidental slide into a 1500-line module is what we're catching.

## Placement — decide before you write

`rllm` is a Cargo workspace. New code belongs in the smallest scope that owns the concern:

- A new **kernel / hot-loop** variant → its own module under `crates/rllm-runtime/src/...`, behind the existing gating pattern (the `ree*` microbench convention), not appended to an existing kernel file.
- A new **model** → its own file under `crates/rllm-runtime/src/models/`.
- **CLI surface** → `crates/rllm-cli/src/commands/`, one command per file.
- Throwaway analysis / one-off repro → a `*.py` script at repo root (already the convention), never inside a crate.

If a change spans more than one of these, that's a sign it should be split into more than one change.

## Modular discipline (the checklist)

1. **Single responsibility per module.** If you can't name a file in one noun phrase, it's doing too much.
2. **Extract, don't append.** When a function crosses ~100 lines, pull cohesive blocks into named helpers. The name is documentation.
3. **Narrow public surface.** Keep items `pub(crate)` / private by default; export only what callers outside the module truly need. A module that's all `pub` is a module with no boundary.
4. **Dependencies point one way.** Lower layers (kernels, math) must not reach back up into orchestration (session, CLI). If you need an upward call, pass a value or a closure instead.
5. **Test in isolation.** If a unit needs half the engine spun up to test, its boundaries are wrong.

## Imports & cycles

- **Rust:** the compiler *allows* module cycles within a crate, so the hook does **not** treat them as errors — flagging them would be noise. What the guard cares about instead is *layering*: kernels/math at the bottom, `session`/orchestration above, `cli` on top. Cross-*crate* cycles are impossible (Cargo forbids them) — if you reach for one, you've put code in the wrong crate.
- **Python:** the helper scripts can genuinely deadlock on circular imports, so the hook builds an import graph across project `.py` files at commit time and warns on any cycle. Break a cycle by moving the shared symbol into a third module both can import.

## How the hook fits in

- **On every `Write`/`Edit`** of a `.rs`/`.py` file: checks that one file's length and longest function, warns inline.
- **Before `git commit`**: re-scans changed `.rs`/`.py` files and runs the Python circular-import scan.

The hook is advisory context — when it fires, read the warning and decide: split now, or note why the exception is justified. Don't reflexively bump the threshold.
