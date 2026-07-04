# Changelog

All notable changes to Spissa are documented in this file.

This file is **auto-generated** by the release workflow from Conventional
Commit messages — do not edit the entries by hand. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions use Spissa's
four-part `A.B.C.D` scheme (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

<!-- BUMP:INSERT -->

## [0.1.7.0] - 2026-07-04

### Documentation
- remove remaining 'no wrapping / from scratch' claims
- drop the 'no wrapping Ollama/llama.cpp' design principle


## [0.1.6.0] - 2026-07-04

### Documentation
- use consistent contact email across SECURITY and CoC
- translate AI agent spec to English
- remove internal superpowers research plans and specs

### Build & CI
- fix clippy -D warnings on non-aarch64 targets


## [0.1.5.0] - 2026-06-27

### Fixes
- gate unix-only mmap advise/lock behind cfg(unix)


## [0.1.4.0] - 2026-06-26

### Features
- wire REEBORN-FOR edge codec into CLI + runtime registries


## [0.1.3.0] - 2026-06-25

### Features
- REEBORN edge codec — coderless FOR exponent + 8-lane rANS probe

### Other
- research(reeborn): REEBORN lossless-codec investigation E0-E8 + prior-art survey


## [0.1.2.0] - 2026-06-24

### Features
- VibeThinker-3B / Qwen2 support via Llama adapter + QKV bias

### Fixes
- correct f32_to_fp16 subnormal shift overflow that emitted NaN q8 scales


## [0.1.1.0] - 2026-06-24

### Build & CI
- simplify bump to patch-per-push, trim Android ABIs, add Linux x86_64 release


## [0.1.0.0] - 2026-06-24

### Build & CI
- Add automated four-part (`A.B.C.D`) versioning + changelog pipeline.
- Add CI (format/build/test) plus Android cross-compile (`arm64-v8a`, `x86_64`)
  and native Linux (`x86_64`) build workflows producing `spissa` binaries and a
  GitHub Release on every push to `main`.

### Documentation
- Add mandatory `CONTRIBUTING.md` describing the versioning, changelog, and
  release process.
