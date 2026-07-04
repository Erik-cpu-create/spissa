# Security Policy

## Supported versions

Spissa is pre-1.0 and evolves quickly. Security fixes are applied to the latest
release on `main`. Older tagged releases are not maintained.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately via one of:

- GitHub's [private vulnerability reporting](https://github.com/Erik-cpu-create/spissa/security/advisories/new)
  (preferred — **Security → Report a vulnerability**), or
- Email the maintainer at **ramaerikesprada.ganteng@gmail.com** with the subject
  `[SECURITY] Spissa`.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal `.spsa` file or command line if relevant),
- affected version / commit,
- any suggested mitigation.

## What to expect

- **Acknowledgement** within 5 business days.
- An assessment and, if confirmed, a fix timeline. Simple issues are typically
  patched within 30 days; complex ones may take longer.
- Credit in the release notes if you'd like it (let us know your preference).

## Scope

In scope: memory-safety bugs, crashes or panics on untrusted `.spsa` input,
integer overflow in the codec/container parsers, and path-traversal or
resource-exhaustion issues when reading model files.

Out of scope: quality/accuracy of model outputs, performance regressions, and
issues that require a malicious build environment.
