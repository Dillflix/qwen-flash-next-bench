# Qwen3.8-Flash-Next topology benchmark harness

This package runs repeatable `llama-server` comparisons for the Strix Halo + RX 7900 XT system. It is preconfigured for the existing fork, Vulkan and ROCm 10 builds, the PLE16 Vulkan-safe model, the original joined-PLE model, and the Q8 MTP sidecar.

The important default is intentionally small: `smoke` compares routed-expert placement with an 82/18 contiguous layer split. Larger placement, MTP, context, and backend sweeps are opt-in.

## What it records

For every measured completion the harness records:

- prompt-processing and decode throughput from `llama-server`;
- HTTP wall time, generated-token count, stop reason, and server startup time;
- MTP drafted/accepted token counts and acceptance ratio when available;
- a SHA-256 hash of the exact greedy output, compared with the non-MTP baseline;
- per-DRM-device busy percentage, peak VRAM/GTT use, temperature, and power when exposed by sysfs;
- AMDGPU PCIe receive/send message counters and a max-payload-size bandwidth estimate;
- process RSS and minimum host `MemAvailable`;
- the observed speed and width of PCIe bridge `c5:00.0` during the request;
- the exact server command, environment-specific device listing, git revision, OS, PCI, Vulkan, ROCm, and AMD SMI diagnostics.

Warm-ups are excluded. Measured runs use temperature zero, `top_k=1`, `cache_prompt=false`, and a fixed seed. Experiment order rotates between rounds to reduce temperature/order bias. Each completed probe is appended and flushed to JSONL immediately, so an interrupted multi-hour sweep can be resumed.

## Install on the Fedora host

Copy this whole directory to the host, for example:

```bash
sudo mkdir -p /srv/llm/bench
sudo chown "$USER:$USER" /srv/llm/bench
cp -a qwen-flash-next-bench /srv/llm/bench/
cd /srv/llm/bench/qwen-flash-next-bench
chmod +x qwen_bench.py run-bench.sh
```

The checked-in paths assume:

```text
/srv/llm/src/llama-qwen4exp
/srv/llm/models/qwen-flash-next/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX-PLE16.gguf
/srv/llm/models/qwen-flash-next/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf
/srv/llm/models/qwen-flash-next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
```

If any path differs, edit only the `variables` block at the top of `matrix.json`. No Python packages are required.

## First run

Stop any inference server or GPU-heavy job first. The harness never kills an existing process and will refuse to start if it sees another `llama-server`, `llama-cli`, or `llama-bench`, or if port 8189 is occupied.

```bash
cd /srv/llm/bench/qwen-flash-next-bench
python3 qwen_bench.py self-test
python3 qwen_bench.py preflight --tier smoke
./run-bench.sh --tier smoke --fail-fast
```

The smoke run loads two model placements, performs one excluded warm-up on each, and measures 64 greedy tokens for code and constrained JSON. Watch the console for immediate throughput; then read the generated `results/<timestamp>-smoke/summary.md`.

If a server cannot load, its complete log is retained under `logs/`. The normal behavior is to record the failure and continue to the next experiment; `--fail-fast` is useful during bring-up.

## Recommended benchmark sequence

First isolate placement without MTP:

```bash
./run-bench.sh --tier short \
  --experiments 'expert_vk_no_mtp,layer_*_vk_no_mtp'
```

This runs three rounds of the expert split and 85/15, 82/18, 80/20, 78/22, and 75/25 contiguous splits. Failed out-of-memory points are still useful because the logs establish the feasible dGPU boundary.

The expert experiments intentionally use `--split-mode layer --tensor-split 1,0`, even though they are not conventional layer splits. In llama.cpp, `--split-mode none` removes every model GPU except `--main-gpu` from the scheduler. Layer mode keeps both Vulkan/ROCm backends registered; `1,0` leaves all ordinary layers on device 0 while `--override-tensor` alone moves PLE and routed-expert tensors to device 1. The harness rejects non-CPU tensor overrides combined with split mode `none` during configuration loading.

Benchmark the fork-specific Vulkan kernel and prefill knobs on the expert placement:

```bash
./run-bench.sh --tier tuning
```

This independently tests `GGML_VK_DENSE_WAVE32=1`, `GGML_VK_MMID_WAVE32=1`, both together, ubatch 1024, ubatch 2048, and both wave32 paths with ubatch 2048. The tier stops at an 8192-token requested filler depth and uses a 16384-token context. Do not carry ubatch 2048 into the 65536-token `full` tier without separate stability testing; the fork documents compute-ring timeouts at very long context with that ubatch on related models.

Then isolate MTP depth on the expert placement:

```bash
./run-bench.sh --tier short \
  --experiments 'expert_vk_no_mtp,expert_vk_mtp_n*'
```

The non-MTP experiment remains in the selection so output hashes and speedups have a baseline. Compare both decode tokens/second and MTP acceptance; a larger draft window is not automatically faster.

Compare the best expert and layer candidate with and without MTP:

```bash
./run-bench.sh --tier short \
  --experiments 'expert_vk_no_mtp,expert_vk_mtp_n4,layer_82_18_vk_no_mtp,layer_82_18_vk_mtp_n4'
```

After pruning the shallow-context matrix, measure context sensitivity:

```bash
./run-bench.sh --tier context
```

That tests requested filler depths of 0, 4096, and 16384 tokens with a 32768-token server context. The actual tokenized prompt length—not the four-characters-per-token construction estimate—is in `timing.prompt_n`.

Compare the Q8 K/V cache used by the performance candidates against the maximum-fidelity F16 K/V control:

```bash
./run-bench.sh --tier kv
```

This covers expert and 82/18 layer placement, plus expert MTP n=4. The repeated cache flags in the expanded command are intentional: the per-experiment F16 setting comes last and therefore overrides the Q8 default.

Finally, compare backend/control cases only if useful:

```bash
./run-bench.sh --tier backend
```

This adds APU-only Vulkan, APU-only ROCm 10 with the joined PLE model and unified-memory fallback, and experimental two-device ROCm tensor routing. The mixed ROCm experiment deliberately does **not** enable global `GGML_CUDA_ENABLE_UNIFIED_MEMORY`; doing so would blur device placement and make the comparison difficult to interpret.

The `full` tier is deliberately expensive (five rounds, four prompt depths, 13 configurations). It is a template for final validation, not a sensible first run. Edit its experiment list down to the finalists before using it.

## Resume and select runs

Resume a partially completed directory:

```bash
./run-bench.sh --tier short \
  --resume /srv/llm/bench/qwen-flash-next-bench/results/20260828-013000-short
```

Completed `(round, experiment, workload, depth)` probes are skipped. Keep the same tier and matrix definition when resuming.

Review expanded commands without loading a model:

```bash
./run-bench.sh --tier short --dry-run
```

Regenerate reports after copying or inspecting `results.jsonl`:

```bash
python3 qwen_bench.py summarize results/20260828-013000-short
```

Experiment selectors accept comma-separated exact names or shell-style patterns. Quote patterns so Bash does not expand them against local filenames.

## Output layout

Each run directory contains:

```text
manifest.json          expanded config and exact command for every experiment
preflight.json         paths, file sizes, DRM mapping, memory and process checks
results.jsonl          durable machine-readable result stream
summary.csv            one median row per experiment/workload/depth
summary.md             human-readable ranking and failure list
logs/                  complete stdout/stderr for each server start
responses/             complete JSON response for each measured request
telemetry/             raw one-second telemetry samples
system/                device listings and host/software provenance
```

In `summary.csv`, `gpu_busy_mean` and `gpu_vram_max_gib` are JSON maps keyed by PCI BDF, which avoids assuming that Linux card numbering is stable. The 7900 XT should map to `0000:c7:00.0` in the current topology; verify this in `preflight.json` rather than hard-coding it in analysis.

`pcie_speed_gt_s_max=16` and `pcie_width_lanes_max=4` would confirm the expected Gen4 ×4 host uplink under load. If the request telemetry never rises above 2.5 GT/s ×1, inspect the raw telemetry and `lspci` output before drawing any conclusion about the split. The sampler first uses non-interactive `lspci`; it never prompts for `sudo`.

The `gpu_pcie_*_est_max_mib_s` fields come from AMDGPU's `pcie_bw` sysfs counters. The [kernel documentation](https://docs.kernel.org/gpu/amdgpu/driver-misc.html#pcie-accounting-information) says these are received/sent message counts for the last second plus maximum payload size, not exact byte counters. The harness therefore reports `messages × MPS` explicitly as an estimate/upper bound. It is most useful for comparing relative traffic between placements.

An output-hash mismatch is a correctness warning, not automatic proof of an invalid result: different backends can cross a greedy-token decision boundary because of floating-point differences. Inspect the saved response. MTP itself should preserve target-model sampling semantics, so systematic MTP-only mismatches deserve investigation.

## Adding custom quants or placements

Duplicate an experiment object in `matrix.json`, give it a unique name, change its `model` or arguments, and add that name to a tier. A compact variant can use `"extends": "expert_vk_no_mtp"` plus `args_append` and/or `env`; inheritance is resolved into the manifest before a run starts. Keeping the same workloads and greedy request parameters makes custom-quant comparisons directly reportable by the existing summarizer.

For a quant-quality study, this performance harness should be paired with a fixed evaluation corpus and perplexity or task-accuracy measurements. Exact output hashes catch accidental behavioral differences but are not a quality metric.
