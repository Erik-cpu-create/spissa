# Running RLLM on Android (Termux)

First Android bring-up. The compute kernels are ARM64 (NEON / `sdot` / `smmla`), the same
ISA family as Android phones, so RLLM builds and runs natively in **Termux** — no
cross-compilation needed. This is the easiest path: build on the phone, run on the phone.

> Status: the code is ARM-portable. Big.LITTLE performance-core detection helpers
> (macOS `sysctl` / Linux-Android sysfs) exist and are `cfg`-gated, but they are
> **not currently wired into the worker-thread count** — decode parallelizes across
> all cores the OS reports (`available_parallelism()`), capped by `RLLM_THREADS`.
> The aarch64-android cross-build is **verified to compile** (R174, ~3.9 MB
> stripped ELF, Bionic libc); it has NOT yet been run on physical hardware — report what breaks.

## Two ways to run

**Option A — pre-built binary (recommended; no on-device build).** Cross-compile once on a
dev machine, push the binary + a model to the phone. The phone needs **no Rust/cargo** — it
just runs the binary.

One-time setup on the dev machine:
```sh
brew install --cask android-ndk          # or the NDK from Android Studio
rustup target add aarch64-linux-android
```

Build a **generic-aarch64** binary (runs on ALL ARM64 phones — RLLM runtime-detects
NEON/dotprod/i8mm, so no per-SoC rebuild):
```sh
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
TC=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$(ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt)/bin
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$TC/aarch64-linux-android24-clang
export CC_aarch64_linux_android=$TC/aarch64-linux-android24-clang
export AR_aarch64_linux_android=$TC/llvm-ar
cargo build --release --target aarch64-linux-android -p rllm-cli --bin rllm
$TC/llvm-strip target/aarch64-linux-android/release/rllm     # 5.3 -> 3.9 MB (optional)
```
> Note: we set the linker via env vars instead of `cargo-ndk` — cargo-ndk 4.1.2 panics on
> NDK r29. The `.cargo/config.toml` keeps `target-cpu=native` scoped to the host target so it
> does NOT leak into this cross-build (which must stay generic aarch64).

Push + run (via `adb shell`, or copy into Termux and run there):
```sh
adb push target/aarch64-linux-android/release/rllm /data/local/tmp/
adb push models/gemma-3-1b-it-q8-raw.spsa /data/local/tmp/
adb shell
cd /data/local/tmp && chmod +x rllm
RLLM_INTEGRITY=unchecked ./rllm chat gemma-3-1b-it-q8-raw.spsa --fast
```

**Option B — build on the phone (Termux).** No dev machine needed, but heavier (Rust
toolchain + a multi-minute on-device build). Steps 1–6 below.

## 1. Install Termux

Install **Termux from F-Droid** (the Google Play build is outdated and will fail):
https://f-droid.org/packages/com.termux/

## 2. Install the toolchain

```sh
pkg update && pkg upgrade
pkg install rust git clang binutils
```

## 3. Get the source

Either clone (if the repo is reachable) or copy the project folder onto the phone
(e.g. `adb push` to `/sdcard/`, then `cp -r` into Termux's home so the build can write):

```sh
cd ~
git clone <your-repo-url> rllm    # or: cp -r /sdcard/rllm ~/rllm
cd rllm
```

## 4. Build

```sh
cargo build --release -p rllm-cli --bin rllm
```

Notes:
- `.cargo/config.toml` sets `target-cpu=native` — on a Termux **native** build that
  targets the phone's own CPU, which is correct (do NOT change it for Termux).
- A full release build is RAM- and time-heavy. On a low-RAM phone, build one crate at a
  time or close other apps. If the linker OOMs, try `cargo build --release` without
  `-p` after a clean, or reduce codegen units.

## 5. Copy a model

Copy ONE `.spsa` file to the phone — the file format is platform-independent. Use a small
model (a phone gives a process only a few GB of usable RAM):

| model | size | notes |
|-------|------|-------|
| `gemma-3-1b-it-q8-raw.spsa` | 1.38 GB | fastest (q8, lossy); use `--fast` |
| `gemma-3-1b-it-rans.spsa`   | 1.36 GB | lossless; slower (~3 tok/s) |

Do **not** use the 4B (≈4.79 GB) — it won't fit most phones' usable RAM.

```sh
# from your computer, before running in Termux:
adb push gemma-3-1b-it-q8-raw.spsa /sdcard/
# then in Termux:
mkdir -p ~/rllm/models && cp /sdcard/gemma-3-1b-it-q8-raw.spsa ~/rllm/models/
```

## 6. Run

```sh
cd ~/rllm
# q8 (fast):
RLLM_INTEGRITY=unchecked ./target/release/rllm chat models/gemma-3-1b-it-q8-raw.spsa --fast
# rANS (lossless):
RLLM_INTEGRITY=unchecked ./target/release/rllm chat models/gemma-3-1b-it-rans.spsa
```

Type a message at `you> `, Enter to send. `/exit` to quit, `/reset` for a new conversation.

## Android performance notes

- **Thread count:** the decode path spreads work across all cores the OS reports.
  Performance-core detection helpers exist (sysfs `cpuinfo_max_freq` on Android) but
  are not yet wired into the thread-count decision, so on big.LITTLE SoCs the
  efficiency cores are also used. Tune with `RLLM_THREADS=<n>` — setting it to the
  number of performance cores is the manual way to get the P-core-only behavior.
- **mlock** (`--fast` enables it) may fail in Termux without privileges — that is
  non-fatal (the run continues, just without pinning). Drop `--fast` to skip it.
- **RAM:** a 1B model needs ~1.7–2.3 GB resident. If the phone is tight, the OS will
  thrash — close other apps first.
- `RLLM_INTEGRITY=unchecked` skips the load-time SHA pass (faster start). Remove it for a
  full integrity check.
