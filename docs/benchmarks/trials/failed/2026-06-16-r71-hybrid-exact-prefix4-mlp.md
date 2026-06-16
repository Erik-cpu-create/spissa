# Trial R71: Hybrid Exact Configuration (Prefix N=4 + Global MLP Exact)

**Date**: 2026-06-16
**Goal**: Push the hybrid exactness strategy further. Since Trial R70 (Prefix N=2 + Exact MLP) failed at Layer 5 due to sparse attention drift in Layers 2-4, we extended the fully exact prefix to **N=4 layers**.
**Model**: Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm

## Configuration

- `RLLM_AIP_EXACT_PREFIX_LAYERS=4` (Layers 0, 1, 2, and 3 are 100% exact)
- `RLLM_AIP_EXACT_LAYER_PROJECTION=mlp` (MLP Gate/Up/Down are exact across all layers)
- Top-K: 4
- Tokens: 2

## Results

```text
> How space

[TTFT/Prefill: 19.40s | Decode: 0.11 tok/s | E2E: 0.07 tok/s | Total: 2 tokens | Context: 55 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=49 fallbacks=0 max_topk=4 skipped_madds=387738624 scratch=32 bytes input_tile_reads=196 input_tile_bytes=1517568 phrase_novelty=0/1 max_ngram=0 layer_drift_probe=1 layers=16 mismatch_layers=9 first_mismatch_layer=7 pre_mismatch_max_l2_milli=3087 pre_mismatch_max_cosine_gap_milli=354 max_l2_milli=22114 max_cosine_gap_milli=912 max_exact_margin_milli=5268 layer_attribution_probe=1 attribution_layer=7 attention_l2_milli=1872 attention_cosine_gap_milli=979 gate_up_l2_milli=8629 gate_up_cosine_gap_milli=923 down_l2_milli=5547 down_cosine_gap_milli=884 | Repetition: ratio=0.00 max_run=1 unique=2/2]
```

## Analysis

- **Output Quality**: Failed. `How space` instead of `How can`.
- **First Mismatch Layer**: 7
- **Mismatch Layers Count**: 9
- **Attribution Probe (Layer 7)**:
  - Attention L2 Milli: 1872
  - Gate/Up L2 Milli: 8629
  - Down L2 Milli: 5547
- **Interpretation**: By extending the exact prefix to 4 layers, the first top-1 hidden-state mismatch was delayed from Layer 5 to Layer 7. However, the sparse attention in Layers 4, 5, and 6 still accumulated enough error (pre_mismatch_max_l2_milli=3087) to corrupt the exact MLP in Layer 7, yielding a hallucination. The error entering the MLP at Layer 7 is highly amplified, as seen by the Gate/Up L2 mismatch of 8629.

## Conclusion

Prefix N=4 + Exact MLP is **insufficient** to prevent hallucination. We have strong evidence that extending the exact prefix simply shifts the drift further down the network. As long as Sparse Attention operates uncorrected, it will accumulate numerical drift until the hidden state is corrupted. The next step must focus on increasing the quality of the Sparse Attention projections.
