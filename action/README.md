# TurboPress compress-action

A GitHub Action that compresses a model with TurboPress in CI and **fails the
build when fidelity gates are not met** — so a regression in a model you ship
can't merge silently.

## Usage

```yaml
- uses: godsonj64/turbopress/action@v1   # or a dedicated turbopress/compress-action repo
  with:
    api-key: ${{ secrets.TURBOPRESS_API_KEY }}
    model: your-org/your-model
    targets: "4bit,3bit"
    gates: "mean_kl_max=0.10,top1_agreement_min=0.70"
    api-url: https://api.turbopress.ai
```

## Inputs

| input | default | description |
|-------|---------|-------------|
| `api-key` | — | TurboPress API key (store as a repo secret) |
| `model` | — | Hugging Face model id or path |
| `targets` | `4bit` | comma-separated bit-width targets |
| `gates` | `""` | comma-separated `metric_max`/`metric_min` thresholds |
| `private` | `true` | treat model as private (billed) |
| `api-url` | `https://api.turbopress.ai` | control-plane base URL |
| `fail-on-gate` | `true` | fail the job when any gate is unmet |
| `timeout-seconds` | `3600` | max wait for the job |

## Outputs

- `job-id` — the created job id
- `gates-passed` — `true`/`false`

Gate keys are `<metric>_max` / `<metric>_min` where `<metric>` is any key in the
certificate metrics (`mean_kl`, `top1_agreement`, `ppl_q`, …). `mean_kl_max=0.10`
requires `mean_kl <= 0.10`; `top1_agreement_min=0.70` requires `>= 0.70`.
