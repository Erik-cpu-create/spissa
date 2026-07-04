<!-- Thanks for contributing to Spissa! Please fill this out. -->

## Summary

<!-- What does this PR do and why? -->

## Related issues

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance
- [ ] Refactor
- [ ] Docs / CI only

## Checklist

- [ ] Commits are **signed off** (`git commit -s`) per the [DCO](../CONTRIBUTING.md#0-licensing--developer-certificate-of-origin-dco)
- [ ] `cargo fmt --all` — formatted
- [ ] `cargo clippy --workspace --all-targets` — no new warnings
- [ ] `cargo build --workspace` — compiles
- [ ] `cargo test --workspace` — passes
- [ ] Conventional Commit message; used a `[minor]`/`[major]`/`[revision]` marker if needed
- [ ] I did not break the [project invariants](../CONTRIBUTING.md#6-project-invariants-do-not-break)
      (lossless by default, honest metrics, custom RTC codecs, codecs round-trip)

## Notes for reviewers

<!-- Benchmarks, REE kernel name, trade-offs, anything worth flagging. -->
