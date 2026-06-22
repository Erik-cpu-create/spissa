# Trial R70: Hybrid Exact Configuration (Prefix N=2 + Global MLP Exact)

**Date**: 2026-06-16
**Goal**: Test whether combining fully exact Prefix Layers (N=2) with globally exact MLPs can stabilize hidden-state numerical drift and prevent hallucination, while keeping decode speed high.
**Model**: Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa

## Configuration

- `RLLM_AIP_EXACT_PREFIX_LAYERS=2` (Layer 0 and 1 are 100% exact)
- `RLLM_AIP_EXACT_LAYER_PROJECTION=mlp` (MLP Gate/Up/Down are exact across all layers)
- Top-K: 4
- Tokens: 2

## Results

```text
> Howellas

[TTFT/Prefill: 21.87s | Decode: 0.12 tok/s | E2E: 0.07 tok/s | Total: 2 tokens | Context: 55 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=57 fallbacks=0 max_topk=4 skipped_madds=408669184 scratch=32 bytes input_tile_reads=228 input_tile_bytes=1599488 phrase_novelty=0/1 max_ngram=0 layer_drift_probe=1 layers=16 mismatch_layers=12 first_mismatch_layer=5 pre_mismatch_max_l2_milli=2784 pre_mismatch_max_cosine_gap_milli=424 max_l2_milli=21037 max_cosine_gap_milli=950 max_exact_margin_milli=5268 layer_attribution_probe=1 attribution_layer=5 attention_l2_milli=1646 attention_cosine_gap_milli=1050 gate_up_l2_milli=4480 gate_up_cosine_gap_milli=809 down_l2_milli=3541 down_cosine_gap_milli=901 | Repetition: ratio=0.00 max_run=1 unique=2/2]
```

## Analysis

- **Output Quality**: Failed. `Howellas` instead of `How can`.
- **First Mismatch Layer**: 5
- **Mismatch Layers Count**: 12
- **Attribution Probe (Layer 5)**:
  - Attention L2 Milli: 1646
  - Gate/Up L2 Milli: 4480
  - Down L2 Milli: 3541
- **Interpretation**: Even though MLP was exact globally, the sparse attention in layers 2, 3, and 4 accumulated enough error (pre_mismatch_max_l2_milli=2784) that by Layer 5, the model drifted and produced a mismatch. The input to the MLP at Layer 5 is already corrupted because the attention in Layers 2-5 was sparse. The exact MLP is amplifying the error present in the hidden state, not fixing it.

## Conclusion

Combining Prefix N=2 and Exact MLP delays the drift from Layer 2 to Layer 5, but it is **not sufficient** to prevent hallucination. The attention layers are contributing too much error once the prefix exactness ends.
