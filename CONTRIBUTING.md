# Contributing to Spissa — MANDATORY Process / Proses WAJIB

> This document is **mandatory** (`dokumen wajib`). Every change to this
> repository **must** follow the rules below. CI enforces them; pull requests
> that violate them will not be merged.

Spissa is proprietary software (see [`LICENSE`](LICENSE)). These rules exist so
that every change is **versioned automatically** and recorded in a
**changelog**, and so the runtime keeps building (including the Android
cross-compile artifacts).

---

## 1. Versioning scheme — `A.B.C.D`

Spissa uses a **four-part** version: `MAJOR.MINOR.PATCH.REVISION`.

| Part | Name | Bumped when… |
|------|------|--------------|
| `A` | **MAJOR** | a breaking change (format/API/runtime incompatibility) |
| `B` | **MINOR** | a new backward-compatible feature |
| `C` | **PATCH** | a bug fix or any normal change |
| `D` | **REVISION** | docs/CI/chore-only changes (no code behaviour change) |

When a higher part is bumped, every lower part resets to `0`
(e.g. a feature bump takes `0.1.4.2` → `0.2.0.0`).

**A normal change advances the patch part**, exactly as requested:

```
0.0.1.0  →  (new change)  →  0.0.2.0
```

The single source of truth is the [`VERSION`](VERSION) file. The first three
parts (`A.B.C`) are mirrored into `Cargo.toml` (`[workspace.package].version`)
because Cargo only accepts three-part SemVer; the fourth part lives only in
`VERSION`, `CHANGELOG.md`, and the git tag `vA.B.C.D`.

**You do not edit `VERSION`, `CHANGELOG.md`, or the Cargo version by hand.** The
release workflow does it for you on every push to `main` (see §4).

---

## 2. Commit messages — Conventional Commits (REQUIRED)

The automatic version bump and the changelog are generated **from your commit
messages**, so they must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <summary>
```

| `type`   | Effect on version | Changelog section |
|----------|-------------------|-------------------|
| `feat`   | MINOR (`B`)       | Features |
| `fix`    | PATCH (`C`)       | Fixes |
| `perf`   | PATCH (`C`)       | Performance |
| `refactor` | PATCH (`C`)     | Refactor |
| `docs`   | REVISION (`D`)*   | Documentation |
| `build` / `ci` | REVISION (`D`)* | Build & CI |
| `chore` / `style` / `test` | REVISION (`D`)* | (not listed) |
| any `type!:` or body `BREAKING CHANGE:` | MAJOR (`A`) | Features (marked breaking) |

\* A push is a REVISION bump **only if every** commit since the last release is
docs/CI/chore-class. If any commit is `feat`/`fix`/`perf`/`refactor`, that wins.

Examples:

```
feat(codec): add rtc-delta-v1 base-exponent coder
fix(tokenizer): group digits in runs of <=3
perf(llama): stream tied-embedding rows from cache
docs: reconcile RLLM->Spissa naming
feat(format)!: bump .spsa container to v2   # MAJOR
```

---

## 3. Before you push (local gates)

```bash
cargo fmt --all          # format
cargo build --workspace  # must compile
cargo test --workspace   # must pass
git diff --check         # no whitespace errors
```

CI (`.github/workflows/ci.yml`) re-runs build + test on every push and PR.

---

## 4. What happens automatically (release pipeline)

On every push to **`main`** (`.github/workflows/release.yml`):

1. **Version** — `scripts/bump-version.sh` reads the commits since the last
   `v*` tag, decides the bump level (§1–§2), and rewrites `VERSION`,
   `Cargo.toml`, `Cargo.lock`, and prepends a new `CHANGELOG.md` section.
2. The bot commits this as `chore(release): vA.B.C.D [skip ci]` and pushes a
   matching annotated git tag `vA.B.C.D`.
3. **Android build** — the runtime is cross-compiled for Android
   (`arm64-v8a`, `armeabi-v7a`, `x86_64`) and the `spissa` binary is collected.
4. **Release** — a GitHub Release is created for the tag, with the generated
   changelog as its notes and the Android binaries attached as assets.

Because the release commit carries `[skip ci]`, it does not trigger another
release run (no infinite loop).

> Releases are produced from `main`. Feature branches and PRs only run CI and a
> build-only Android job (`.github/workflows/android.yml`); they do not bump the
> version or publish a release.

---

## 5. Android cross-compile (local reproduction)

The CI uses [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk) + the Android NDK:

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
cargo install cargo-ndk --locked
export ANDROID_NDK_HOME=/path/to/android-ndk
cargo ndk -t arm64-v8a -p 24 build --release -p spissa-cli --bin spissa
# output: target/aarch64-linux-android/release/spissa
```

`spissa` links no system OpenSSL (TLS is rustls), so no extra native libraries
are needed beyond the NDK toolchain.

---

## 6. Project invariants (do not break)

These mirror the design principles in the README; CI/review will reject changes
that violate them:

1. **Lossless by default** — decoded weights are bit-identical to the originals.
2. **Honest metrics** — never overclaim compression/speed.
3. **From scratch** — no wrapping Ollama/llama.cpp.
4. **Custom RTC codecs** — no generic compression libraries by default.
5. **Every codec round-trips** — `decode(encode(x)) == x`, with tests.
