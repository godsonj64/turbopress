# Certificate format

Every compression run writes `quantization_config.json` into the artifact. It is
the reproducibility record for the quantized model — the settings that produced
it and the fidelity measured against the full-precision model.

```json
{
  "meta": {
    "format_version": 1,
    "pipeline": "rotate -> quarter-power equilibration -> TCQ (analytic codebook)",
    "model_id": "Qwen/Qwen3-4B",
    "bits": 3,
    "n_states": 64,
    "generators": [91, 121],
    "bits_per_weight_total": 3.02,
    "seed": 0
  },
  "config":  { "...": "the exact run configuration" },
  "metrics": {
    "mean_kl": 0.289,
    "top1_agreement": 0.710,
    "ppl_fp": 29.11,
    "ppl_q": 38.6,
    "n_tokens": 4096
  },
  "quantized_layers": ["model.layers.0.self_attn.q_proj.weight", "..."]
}
```

| field | meaning |
|-------|---------|
| `meta.pipeline` | the transform chain, in order |
| `meta.bits_per_weight_total` | storage-weighted average including all overheads |
| `meta.generators` | trellis code generator polynomials (octal → int) |
| `meta.seed` | rotation-sign seed; signs are stored, so runs are reproducible |
| `metrics.mean_kl` | token-level KL(fp ‖ quant) in nats — the "visible loss" metric |
| `metrics.top1_agreement` | fraction of tokens where argmax matches the fp model |
| `metrics.ppl_fp` / `ppl_q` | perplexity of the fp and quantized models on the eval set |

!!! note "Signed manifests"
    A signed certificate (model hash, eval-set hash, and an Ed25519 signature
    over the manifest) is planned. Until then, treat `quantization_config.json`
    as an **unsigned** record: reproduce the numbers yourself with
    `turbopress validate`.
