# Superseded: Spissa ECHO Architecture and EMBER Runtime

Status: superseded by [`docs/spissa-rama-architecture.md`](spissa-rama-architecture.md)

This document name is retained as a compatibility pointer because earlier Phase 5D.5/5E work used the ECHO/EMBER naming.

The accepted official naming is now:

```text
Product/system: Spissa = Runtime-compressed Local LLM
Codec layer:    RTC  = Rama Tensor Codec
Architecture:   RAMA = Rama Active Memory Architecture
Future kernel:  ERIK = Episodic Recall Inference Kernel
```

Use [`docs/spissa-rama-architecture.md`](spissa-rama-architecture.md) as the source of truth for the brain-inspired, memory-first runtime architecture.

Implementation note: some runtime code and tests may still use legacy `echo` / `ContextEcho` names until a dedicated migration slice renames them safely with tests.
