# CLI reference

```
turbopress --version
turbopress {compress,validate} ...
```

## `turbopress compress`

Quantize every decoder linear of a Hugging Face causal LM and write a certified,
self-contained artifact.

```bash
turbopress compress <model> [options]
```

| flag | default | description |
|------|---------|-------------|
| `model` | — | Hugging Face model id or local path |
| `--bits` | `3` | bits/weight, 2–6 |
| `--n-states` | `64` | trellis states: `4`, `8`, `16`, `64` (16 = faster, 64 = best) |
| `--out` | `turbopress_out` | output directory |
| `--seqlen` | `512` | tokens per eval/calibration sequence |
| `--eval-seqs` | `8` | eval set = `eval-seqs × seqlen` tokens |
| `--calib-seqs` | `8` | disjoint calibration slice |
| `--batch` | `2` | sequences per forward |
| `--seed` | `0` | RNG seed (rotation signs are stored, so runs are reproducible) |
| `--device` | auto | `cuda` or `cpu` |
| `--no-self-test` | off | skip reloading the artifact to verify it round-trips |

## `turbopress validate`

Measure KL(reference ‖ candidate), top-1 next-token agreement, and perplexity
of both models on held-out text. The two models must share a tokenizer/vocab.

```bash
turbopress validate <reference> <candidate> [options]
```

| flag | default | description |
|------|---------|-------------|
| `reference` | — | reference model id/path (e.g. the fp16 model) |
| `candidate` | — | candidate model id/path (e.g. the quantized copy) |
| `--seqs` | `16` | eval sequences |
| `--seqlen` | `256` | tokens per sequence |
| `--batch` | `4` | sequences per forward |
| `--device` | auto | `cuda` or `cpu` |
| `--dtype` | `float16` | `float16`, `bfloat16`, or `float32` |
| `--data-dir` | `data` | cache dir for eval text |
| `--out` | — | write the full JSON report to this file |
