# Running RLLM on Android (Termux)

First Android bring-up. The compute kernels are ARM64 (NEON / `sdot` / `smmla`), the same
ISA family as Android phones, so RLLM builds and runs natively in **Termux** — no
cross-compilation needed. This is the easiest path: build on the phone, run on the phone.

> Status: the code is ARM-portable and the macOS-specific bits (P-core detection via
> `sysctl`) are `cfg`-gated with Android/Linux fallbacks (R172 adds big.LITTLE detection
> from sysfs). It has NOT yet been tested on physical Android hardware — treat this as a
> bring-up guide and report what breaks.

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

Copy ONE `.rllm` file to the phone — the file format is platform-independent. Use a small
model (a phone gives a process only a few GB of usable RAM):

| model | size | notes |
|-------|------|-------|
| `gemma-3-1b-it-q8-raw.rllm` | 1.38 GB | fastest (q8, lossy); use `--fast` |
| `gemma-3-1b-it-rans.rllm`   | 1.36 GB | lossless; slower (~3 tok/s) |

Do **not** use the 4B (≈4.79 GB) — it won't fit most phones' usable RAM.

```sh
# from your computer, before running in Termux:
adb push gemma-3-1b-it-q8-raw.rllm /sdcard/
# then in Termux:
mkdir -p ~/rllm/models && cp /sdcard/gemma-3-1b-it-q8-raw.rllm ~/rllm/models/
```

## 6. Run

```sh
cd ~/rllm
# q8 (fast):
RLLM_INTEGRITY=unchecked ./target/release/rllm chat models/gemma-3-1b-it-q8-raw.rllm --fast
# rANS (lossless):
RLLM_INTEGRITY=unchecked ./target/release/rllm chat models/gemma-3-1b-it-rans.rllm
```

Type a message at `you> `, Enter to send. `/exit` to quit, `/reset` for a new conversation.

## Android performance notes

- **big.LITTLE is handled (R172):** the q8 decode path defaults to the phone's
  performance cores (read from `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`,
  excluding the slow efficiency tier) — same 2.2× win measured on Apple Silicon. Override
  with `RLLM_THREADS=<n>` if the auto-detection is wrong on your SoC.
- **mlock** (`--fast` enables it) may fail in Termux without privileges — that is
  non-fatal (the run continues, just without pinning). Drop `--fast` to skip it.
- **RAM:** a 1B model needs ~1.7–2.3 GB resident. If the phone is tight, the OS will
  thrash — close other apps first.
- `RLLM_INTEGRITY=unchecked` skips the load-time SHA pass (faster start). Remove it for a
  full integrity check.
