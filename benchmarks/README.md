# Quantization benchmark: TurboPress vs real SOTA

Compresses a **Qwen ≤2B** model (default `Qwen/Qwen3-1.7B`) with each method's
**real library** — bitsandbytes, HQQ, GPTQ (`gptqmodel`), AWQ (`autoawq`), and
TurboPress — then scores every result on **one shared eval** against the fp16
model: mean KL, top-1 agreement, and perplexity on held-out WikiText-2.

## Why separate scripts

These backends pin **conflicting `transformers` / CUDA-kernel versions**. Trying
to install them together usually breaks at least one. So each method is its own
script, meant to run in its **own fresh environment** (venv, container, or a
Colab runtime restart between methods). They all import `common_eval.py`, so the
eval slice, fp16 reference, and metrics are identical no matter where each ran —
the results/*.json files drop into one table via `compare.py`.

## Aggressiveness

| method | script | bit-widths | notes |
|---|---|---|---|
| TurboPress (TCQ) | `run_turbopress.py` | 2, 3, 4 | rotation + quarter-power equil + trellis |
| HQQ | `run_hqq.py` | 2, 3, 4 | calibration-free, strong at low bits |
| GPTQ | `run_gptq.py` | 2, 3, 4 | real error-feedback PTQ, group-128 |
| bitsandbytes | `run_bnb.py` | 4 (NF4) | its weight-only floor |
| AWQ | `run_awq.py` | 4 | native regime; no 2/3-bit |

## Run it

Each method in a clean environment. Example with venvs on a Linux GPU box:

```bash
MODEL=Qwen/Qwen3-1.7B

python -m venv .venv_tp && . .venv_tp/bin/activate
pip install "turbopress[llm]>=0.4.3" "transformers>=4.51" datasets accelerate
python run_turbopress.py --model $MODEL --bits 2,3,4
deactivate

python -m venv .venv_hqq && . .venv_hqq/bin/activate
pip install "transformers>=4.51" datasets accelerate hqq
python run_hqq.py --model $MODEL --bits 2,3,4
deactivate

python -m venv .venv_gptq && . .venv_gptq/bin/activate
pip install "transformers>=4.51" datasets accelerate gptqmodel
python run_gptq.py --model $MODEL --bits 2,3,4
deactivate

python -m venv .venv_bnb && . .venv_bnb/bin/activate
pip install "transformers>=4.51" datasets accelerate bitsandbytes
python run_bnb.py --model $MODEL
deactivate

python -m venv .venv_awq && . .venv_awq/bin/activate
pip install "transformers>=4.51" datasets accelerate autoawq
python run_awq.py --model $MODEL          # may fail on Qwen3 — see note
deactivate

python compare.py                          # merge results/ into one table
```

`run_all.sh` does exactly this loop. On **Colab**, run one script per session
and **restart the runtime** before the next (Runtime → Restart), so each method
gets clean deps; `results/` persists on disk across restarts.

## Fairness notes (read before quoting numbers)

- **Nominal vs effective bits.** bnb-NF4 / HQQ / GPTQ / AWQ carry group
  scales+zeros (~4.1–4.3 real bits at "4-bit"); TurboPress ≈ nominal+0.02.
  The `size_mb` column is the honest on-disk comparison — use it, not just bits.
- **One model, one dataset (WikiText-2).** Directional, not a paper table. For a
  real claim add multiple sizes and zero-shot tasks (lm-eval-harness).
- **Same fp16 reference + same eval slice** for every method — that part is apples-to-apples.
- **AWQ may not run on Qwen3** (archived library). If it errors, the table just
  omits it; the others are unaffected.
