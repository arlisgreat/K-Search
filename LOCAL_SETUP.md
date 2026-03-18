# K-Search Local Setup

This workspace is configured to keep runtime state inside `/data1/workspace/airulan/bench/K-Search`.

## 1. Initialize the repo-local environment

```bash
cd /data1/workspace/airulan/bench/K-Search
bash scripts/setup_local_env.sh
```

The helper keeps these paths inside the repo:

- `.venv/`
- `.cache/`
- `.tmp/`
- `.ksearch-output/`
- `wandb/`
- `vendor/`
- `data/`

It also sets `HF_ENDPOINT=https://hf-mirror.com` by default.

## 2. Fetch optional external dependencies

Clone the official `flashinfer-bench-ksearch` fork into `vendor/`:

```bash
bash scripts/fetch_flashinfer_bench_ksearch.sh
```

Download the `flashinfer-trace` dataset into `data/flashinfer-trace`:

```bash
bash scripts/fetch_flashinfer_trace.sh
```

Notes:

- The existing external `flashinfer-trace` checkout under `/data1/workspace/airulan/bench/flashinfer-trace` is about `25G`.
- The helper intentionally does not copy that external dataset automatically.

## 3. Reproduce the published results

Benchmark the published GPUMode TriMul kernels:

```bash
bash results/gpumode_trimul/bench.sh
```

Benchmark the published FlashInfer kernels:

```bash
bash scripts/repro_flashinfer_results.sh --definition mla_paged_decode_h16_ckv512_kpe64_ps1
bash scripts/repro_flashinfer_results.sh --definition mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
bash scripts/repro_flashinfer_results.sh --definition gqa_paged_decode_h32_kv4_d128_ps1
bash scripts/repro_flashinfer_results.sh --definition moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
```

## 4. Re-run the search experiments

GPUMode TriMul:

```bash
export LLM_API_KEY=...
bash scripts/gpumode_trimul_wm.sh
```

FlashInfer MLA decode:

```bash
export LLM_API_KEY=...
bash scripts/mla_decode_wm.sh
```

## 5. Restore from any saved round

Every optimization round is now checkpointed under `checkpoints/<task>/runs/<run_id>/rounds/`.

List saved runs for a task:

```bash
python scripts/list_round_checkpoints.py --task-name gpumode_trimul
python scripts/list_round_checkpoints.py --task-name mla_paged_decode_h16_ckv512_kpe64_ps1
```

Resume the latest saved round from a previous run:

```bash
export CONTINUE_FROM_RUN_ID=<run_id>
bash scripts/gpumode_trimul_wm.sh
```

Resume a specific round:

```bash
export CONTINUE_FROM_RUN_ID=<run_id>
export CONTINUE_FROM_ROUND=12
bash scripts/mla_decode_wm.sh
```
