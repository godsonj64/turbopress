# Quickstart

## Install

```bash
pip install "turbopress[llm]"     # torch + transformers + datasets
```

For development from a clone:

```bash
git clone https://github.com/godsonj64/turbopress
cd turbopress
pip install -e ".[dev,llm]"
```

Compression needs a single CUDA GPU with enough VRAM to hold the model in fp16
(a 4B model needs ~10 GB). Validation briefly holds eval logits on CPU.

## Compress a model

```bash
turbopress compress Qwen/Qwen3-4B --bits 3 --out ./out
```

This writes `./out/Qwen3-4B-turbopress-3bit/` containing:

| file | contents |
|------|----------|
| `turbopress_weights.pt` | packed trellis codes at the true bit-width |
| `run_quantized.py` | standalone loader/demo — no repo needed |
| `quantization_config.json` | settings + measured KL / top-1 / perplexity |
| `hf_config/`, `tokenizer/` | everything needed to run offline |
| `turbopress.log` | full run log |

…plus the same folder zipped for download.

## Run the compressed model

```bash
cd ./out/Qwen3-4B-turbopress-3bit
python run_quantized.py --prompt "The capital of France is"
python run_quantized.py --export-hf ./exported_fp16   # plain HF checkpoint
```

## Validate any two models

```bash
turbopress validate Qwen/Qwen3-4B ./out/Qwen3-4B-turbopress-3bit --out report.json
```

The `validate` subcommand is method-agnostic: the candidate can be a TurboPress
artifact, a GPTQ/AWQ checkpoint, or any HF-loadable model that shares the
reference's tokenizer.
