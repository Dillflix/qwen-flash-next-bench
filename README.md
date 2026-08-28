# Qwen3.8-Flash-Next topology benchmark harness

This package runs repeatable `llama-server` comparisons for the Strix Halo + RX 7900 XT system. It is preconfigured for the existing fork, Vulkan and ROCm 10 builds, the PLE16 Vulkan-safe model, the original joined-PLE model, and the Q8 MTP sidecar.

The important default is intentionally small: `smoke` compares APU-only Vulkan with the representative 88/12 contiguous layer split. Larger strategy, MTP, context, and backend sweeps are opt-in.

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

Warm-ups are excluded. Measured throughput runs use temperature zero, `top_k=1`,
`cache_prompt=false`, `ignore_eos=true`, and a fixed seed. Ignoring EOS is required
for fixed-length decode timing: otherwise naturally short JSON/code answers report
zero or incomparable throughput. A response producing less than 95% of the requested
tokens is marked invalid, excluded from summaries, and retried when a run is resumed.
Experiment order rotates between rounds to reduce temperature/order bias. Each
completed probe is appended and flushed to JSONL immediately, so an interrupted
multi-hour sweep can be resumed.

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

Large HIP model starts can take several minutes. While waiting for `/health`, the runner prints a heartbeat every 30 seconds with elapsed time, log size, and the latest non-empty server-log line. The default startup timeout is 1200 seconds (20 minutes).

## Recommended benchmark sequence

Start with `vulkan-baseline` below. Earlier bring-up established that nominal 90/10,
88/12, 85/15, and 82/18 forward splits have effectively the same decode rate on
this host. The active tiers therefore use 88/12 as the forward representative and
spend measurement time on structurally different placements and real prefill loads.
The other ratios remain defined for deliberate boundary investigations, but are not
part of the recommended campaign.

The expert experiments intentionally use `--split-mode layer --tensor-split 1,0`, even though they are not conventional layer splits. In llama.cpp, `--split-mode none` removes every model GPU except `--main-gpu` from the scheduler. Layer mode keeps both Vulkan/ROCm backends registered; `1,0` leaves all ordinary layers on device 0 while `--override-tensor` alone moves PLE and routed-expert tensors to device 1. The harness rejects non-CPU tensor overrides combined with split mode `none` during configuration loading.

Bring up the alternative Vulkan strategies with one short pass:

```bash
./run-bench.sh --tier strategy-smoke
```

This compares APU-only, routed-expert, ordinary APU-to-dGPU contiguous layers, reverse dGPU-to-APU layers, and row splitting with intermediate results/KV on each possible main device. Row splitting is expected to be communication-heavy, but measuring both `--main-gpu` choices makes it a useful control rather than an assumption.

If those all load, collect the three-round Vulkan strategy comparison:

```bash
./run-bench.sh --tier placement
```

This compares APU-only, representative forward layer, reverse layer, expert/component,
and both row placements at base and approximately 4K prompt depths. It does not
repeat neighboring forward ratios.

### Complete Vulkan baseline

To screen the decision-relevant Vulkan configurations without invoking HIP/ROCm, run:

```bash
python3 qwen_bench.py preflight --tier vulkan-baseline
./run-bench.sh --tier vulkan-baseline
```

This is a one-round decision pass across 16 non-redundant Vulkan configurations.
The nominal 90/10 through 82/18 forward ratios produced noise-sized decode
differences on the target host, so 88/12 is retained as the representative forward
split while expert/component, reverse-layer, and both row strategies remain as
structurally different controls. Kernel, ubatch, Q8/F16 KV, and MTP variants are
applied to the representative 88/12 placement rather than the slower expert split.

Every configuration is measured with three workloads at base, approximately 4K,
and approximately 16K prompt depths inside a 32K server context, followed by 128
generated tokens. The live output reports both prefill and decode throughput plus
actual prefill token count and milliseconds. `summary.md` ranks decode and prefill
separately and provides an equal-weight geometric mean only as a convenient sorting
aid. The startup and request limits are 10 minutes. Fail-fast is intentionally
omitted so an OOM or unsupported topology is recorded and the campaign continues.

The baseline is a feasibility and ranking screen, not final proof. Use its winners
to prune the three-round placement, tuning, KV, and context tiers; do not run every
loser at long context. Completed probes can be resumed with the run directory as
described below.

### Focused F16 KV and MTP screen

To test only KV precision and MTP on the representative 88/12 Vulkan placement:

```bash
python3 qwen_bench.py preflight --tier vulkan-kv-mtp
./run-bench.sh --tier vulkan-kv-mtp
```

This runs exactly three configurations: the entire target model on the iGPU with
F16 KV; the entire target model on the iGPU with Q8 MTP n=4 on the 7900 XT; and
the entire target model on the iGPU with both F16 KV and Q8 MTP n=4 on the 7900 XT.
There is no repeated Q8/no-MTP control and no target-model layer split.

The MTP commands use `--device Vulkan1,Vulkan0 --tensor-split 1,0`: device-list
index 0 is the iGPU, so 100% of target weights remain there. Layer scheduling is
enabled only to keep Vulkan0 registered for `--spec-draft-device Vulkan0`; Vulkan0
is the 7900 XT and receives only the draft sidecar. All three configurations use
the same workloads, base/~4K/~16K prompt depths, 32K context, and 128-token output.

### Focused 7900 XT prefill screen

MTP does not accelerate target-model prefill: the target remains on the iGPU and
the draft model can add its own prompt-processing work. Test target prefill
placement separately:

```bash
python3 qwen_bench.py preflight --tier vulkan-prefill
./run-bench.sh --tier vulkan-prefill
```

This screen keeps Q8 KV and disables MTP everywhere. It compares APU-only against
one representative contiguous split, the component/expert split, reverse layers,
and both row modes. These are structurally distinct; neighboring layer ratios are
not repeated. One fixed code prompt is measured at approximately 4K and 16K depth.
Each server first processes the same 16K prompt as an excluded warm-up, preventing
cold page-cache/shader effects from masquerading as a placement result. Generation
is limited to eight forced tokens and is diagnostic only; rank this tier by prefill
tok/s and prefill milliseconds.

Bring up ROCm 10 separately before running a large cross-backend matrix:

```bash
python3 qwen_bench.py preflight --tier rocm-smoke
./run-bench.sh --tier rocm-smoke --fail-fast
```

`rocm-smoke` tests APU-only HIP with both the original joined PLE and PLE16 files, an 85/15 heterogeneous layer split, and routed-expert placement. The APU-only controls hide the 7900 XT and enable unified-memory fallback; mixed-GPU experiments expose both devices and deliberately leave the fallback disabled so placement remains measurable.

After bring-up, run the full ROCm placement sweep:

```bash
./run-bench.sh --tier rocm-placement
```

It mirrors the Vulkan ratio/order/row tests using `ROCm1` for the 8060S and `ROCm0` for the 7900 XT. Failed high-dGPU ratios remain useful for establishing the ROCm allocation boundary.

ROCm MTP is isolated in a smaller tier that reserves more dGPU headroom:

```bash
./run-bench.sh --tier rocm-mtp
```

The target uses a 90/10 APU-to-dGPU layer split and the Q8 MTP sidecar stays on `ROCm0`; n=2, n=3, and n=4 are compared against an otherwise identical non-MTP baseline. If the sidecar still exceeds available VRAM, reduce the target dGPU fraction before changing any other variable.

Benchmark the fork-specific Vulkan kernel and prefill knobs on the representative 88/12 placement:

```bash
./run-bench.sh --tier tuning
```

This independently tests `GGML_VK_DENSE_WAVE32=1`, `GGML_VK_MMID_WAVE32=1`, both together, ubatch 1024, ubatch 2048, and both wave32 paths with ubatch 2048. The tier stops at an 8192-token requested filler depth and uses a 16384-token context. Do not carry ubatch 2048 into the 65536-token `full` tier without separate stability testing; the fork documents compute-ring timeouts at very long context with that ubatch on related models.

Then isolate MTP draft depth on the representative 88/12 placement:

```bash
./run-bench.sh --tier short \
  --experiments 'layer_88_12_vk_no_mtp,layer_88_12_vk_mtp_n*'
```

The non-MTP experiment remains in the selection so output hashes and speedups have a baseline. Compare both decode tokens/second and MTP acceptance; a larger draft window is not automatically faster.

After pruning the shallow-context matrix, measure context sensitivity:

```bash
./run-bench.sh --tier context
```

That tests requested filler depths of 0, 4096, and 16384 tokens with a 32768-token server context. The actual tokenized prompt length—not the four-characters-per-token construction estimate—is in `timing.prompt_n`.

Compare the Q8 K/V cache used by the performance candidates against the maximum-fidelity F16 K/V control:

```bash
./run-bench.sh --tier kv
```

This covers APU-only and representative 88/12 layer placement, including MTP n=4.
Experiment-specific cache settings replace the Q8 defaults when commands are
expanded, so the logged command contains one unambiguous value for each cache type.

Finally, compare matched backend/control cases:

```bash
./run-bench.sh --tier backend
```

This pairs Vulkan and ROCm expert, 85/15, and 82/18 placements, then includes APU-only Vulkan and both APU-only ROCm file structures. Compare exact output hashes as well as throughput; backend numerical differences can change a greedy token even when both runs are valid.

The `full` tier is deliberately expensive (five rounds, four prompt depths, nine Vulkan configurations). It is a template for final validation, not a sensible first run. Edit its experiment list down to the finalists before using it.

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
