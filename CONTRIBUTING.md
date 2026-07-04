# Contributing to Spissa — MANDATORY Process / Proses WAJIB

> This document describes the **mandatory** process for changes to this
> repository. Every change **must** follow the rules below. CI enforces them;
> pull requests that violate them will not be merged.

Spissa is open source under the [MIT License](LICENSE). Contributions are very
welcome. These rules exist so that every change is **versioned automatically**
and recorded in a **changelog**, and so the runtime keeps building (including the
Android cross-compile artifacts).

---

## 0. Licensing & Developer Certificate of Origin (DCO)

By contributing to Spissa you agree that your contributions are licensed under
the project's [MIT License](LICENSE) (inbound = outbound).

We use the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) instead of a CLA. It is a lightweight, one-line attestation that you wrote
the patch or otherwise have the right to submit it under the project license.

**Every commit must be signed off.** Add the `-s` flag when you commit:

```bash
git commit -s -m "fix(tokenizer): group digits in runs of <=3"
```

This appends a line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name/email must be real and match your Git identity. Forgot to sign off?
Amend the last commit with `git commit --amend -s`, or for a whole branch:
`git rebase --signoff main`.

---

## 1. Versioning scheme — `A.B.C.D`

Spissa uses a **four-part** version: `MAJOR.MINOR.PATCH.REVISION`.

| Part | Name | When it advances |
|------|------|------------------|
| `A` | **MAJOR** | breaking change — only with an explicit `[major]` marker (or `BREAKING CHANGE`) |
| `B` | **MINOR** | new feature — only with an explicit `[minor]` marker |
| `C` | **PATCH** | **the default — every push to `main` advances this** |
| `D` | **REVISION** | docs/CI-only touch-up — only with an explicit `[revision]` marker |

When a higher part advances, every lower part resets to `0`
(e.g. a `[minor]` bump takes `0.1.4.2` → `0.2.0.0`).

**Every push to `main` advances the patch part by default**, exactly as required:

```
0.0.1.0  →  (any push)  →  0.0.2.0
```

Need a different bump? Put a marker anywhere in the latest commit message:
`[major]`, `[minor]`, or `[revision]`.

The single source of truth is the [`VERSION`](VERSION) file. The first three
parts (`A.B.C`) are mirrored into `Cargo.toml` (`[workspace.package].version`)
because Cargo only accepts three-part SemVer; the fourth part lives only in
`VERSION`, `CHANGELOG.md`, and the git tag `vA.B.C.D`.

**You do not edit `VERSION`, `CHANGELOG.md`, or the Cargo version by hand.** The
release workflow does it for you on every push to `main` (see §4).

---

## 2. Commit messages — Conventional Commits (recommended)

The **changelog** is generated from your commit messages, so following
[Conventional Commits](https://www.conventionalcommits.org/) keeps it tidy. The
commit type does **not** decide the version — the push policy in §1 does. The
type only decides which changelog section the commit lands in:

```
<type>(<optional scope>): <summary>
```

| `type` | Changelog section |
|--------|-------------------|
| `feat` | Features |
| `fix` | Fixes |
| `perf` | Performance |
| `refactor` | Refactor |
| `docs` | Documentation |
| `build` / `ci` | Build & CI |
| anything else | Other |

To override the default patch bump, add `[minor]`, `[major]`, or `[revision]`
anywhere in the commit message (see §1).

Examples:

```
fix(tokenizer): group digits in runs of <=3        # patch (default)
perf(llama): stream tied-embedding rows from cache  # patch (default)
feat(codec): add rtc-delta-v1 coder [minor]         # minor bump
feat(format): bump .spsa container to v2 [major]    # major bump
docs: fix a typo [revision]                         # revision bump
```

---

## 3. Before you push (local gates)

```bash
cargo fmt --all                              # format (CI enforces --check)
cargo clippy --workspace --all-targets -- -D warnings  # lint (enforced in CI)
cargo build --workspace                      # must compile
cargo test --workspace                       # must pass
git diff --check                             # no whitespace errors
```

CI (`.github/workflows/ci.yml`) re-runs fmt (blocking), clippy (blocking, `-D
warnings`), build, and test on every push and PR, and checks that every commit
in a PR is **signed off** (DCO, see §0).

---

## 4. What happens automatically (release pipeline)

On every push to **`main`** (`.github/workflows/release.yml`):

1. **Version** — `scripts/bump-version.sh` advances the patch part by default
   (or the `[minor]`/`[major]`/`[revision]` marker level, §1), then rewrites
   `VERSION`, `Cargo.toml`, `Cargo.lock`, and prepends a new `CHANGELOG.md`
   section built from the commits since the last `v*` tag.
2. The bot commits this as `chore(release): vA.B.C.D [skip ci]` and pushes a
   matching annotated git tag `vA.B.C.D`.
3. **Binaries** — the `spissa` CLI is cross-compiled for Android
   (`arm64-v8a`, `x86_64`) and built natively for Linux (`x86_64`).
4. **Release** — a GitHub Release is created for the tag, with the generated
   changelog as its notes and the Android + Linux binaries (each with a
   `.sha256`) attached as assets.

Because the release commit carries `[skip ci]`, it does not trigger another
release run (no infinite loop).

> Releases are produced from `main`. Feature branches and PRs only run CI and
> build-only Android/Linux jobs (`.github/workflows/android.yml`, `linux.yml`);
> they do not bump the version or publish a release.

---

## 5. Android cross-compile (local reproduction)

The CI uses [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk) + the Android NDK:

```bash
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk --locked
export ANDROID_NDK_HOME=/path/to/android-ndk
cargo ndk -t arm64-v8a -t x86_64 -p 24 build --release -p spissa-cli --bin spissa
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
