# Qwen3.8-Flash-Next topology benchmark harness

This package runs repeatable `llama-server` comparisons for the Strix Halo + RX 7900 XT system. It is preconfigured for the existing fork, Vulkan and ROCm 10 builds, the PLE16 Vulkan-safe model, the original joined-PLE model, and the Q8 MTP sidecar.

The important default is intentionally small: `smoke` compares APU-only Vulkan with the representative 88/12 contiguous layer split. Larger strategy, MTP, context, and backend sweeps are opt-in.

## What it records

For every measured completion the harness records:

- prompt-processing and decode throughput from `llama-server`;
- HTTP wall time, generated-token count, stop reason, and server startup time;
- synchronized multi-request aggregate prefill/decode and end-to-end throughput;
- MTP drafted/accepted token counts and acceptance ratio when available;
- a SHA-256 hash of the exact greedy output, compared with the non-MTP baseline;
- per-DRM-device busy percentage, peak VRAM/GTT use, temperature, and power when exposed by sysfs;
- AMDGPU PCIe receive/send message counters and a max-payload-size bandwidth estimate;
- process RSS and minimum host `MemAvailable`;
- process anonymous/file-backed RSS, storage-read bytes, and major/minor page faults;
- the observed speed and width of PCIe bridge `c5:00.0` during the request;
- the exact server command, environment-specific device listing, git revision, OS, PCI, Vulkan, ROCm, and AMD SMI diagnostics.

Warm-ups are excluded from rankings and recorded separately in `warmups.jsonl`.
The current hot tiers use a nonzero-depth warm-up, then erase the llama-server
slot after every request. This retains file pages and compiled kernels while
preventing target or draft KV reuse from contaminating the next prefill result.
The harness creates a per-user directory under the system temporary directory
and passes it with `--slot-save-path`, which this llama-server fork requires
before enabling slot actions. Erasing a slot does not save a KV file there.
Measured throughput runs use temperature zero, `top_k=1`, `cache_prompt=false`,
`ignore_eos=true`, and a fixed seed. Ignoring EOS is required
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

If any path differs, edit only the `variables` block at the top of `matrix.json`.
No Python packages are required for the benchmark runner. The conversion step below
uses the Python environment already required by `convert_hf_to_gguf.py`.

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

Every completed or interrupted benchmark is automatically packaged beside its run
directory as `<run>.tar.gz`, with a matching `.sha256` file. The archive contains
the summary, raw results, manifest, responses, server logs, telemetry, system
captures, preflight report, and an `archive-manifest.json` listing every payload
file with its size and SHA-256. Models and temporary slot data are never included.

Package an older run retroactively with:

```bash
python3 qwen_bench.py archive results/20260828-154636-production-capacity
```

Send the resulting `results/20260828-154636-production-capacity.tar.gz`; the small
checksum sidecar is useful for transfer verification but is not required for
analysis.

Large HIP model starts can take several minutes. While waiting for `/health`, the runner prints a heartbeat every 30 seconds with elapsed time, log size, and the latest non-empty server-log line. The default startup timeout is 1200 seconds (20 minutes).

## Build the H1 low-risk quant

H1 changes precision where quality is most likely to benefit while retaining the
fork's fast Vulkan ROCmFP4 kernels on all large compute-heavy matrices:

| Tensor family | H1 type | Performance intent |
| --- | --- | --- |
| split PLE table | Q8_0 | mmap-compatible 50.66 GiB table; no extra requantization |
| routed expert gate/up/down | Q4_0_ROCMFP4_FAST (type 101) | preserve the current fast expert kernels and traffic reduction |
| shared expert gate/up/down | Q4_0_ROCMFP4 (type 100) | dual-scale quality protection with essentially the same patched Vulkan kernel path |
| attention and linear attention | Q4_0_ROCMFP4_STRIX mixture | type 100 K/V plus type 101 for the remaining large matrices |
| MoE routers and QSA indexer Q/K | Q8_0 | spend bits only on small, routing-sensitive projections |
| token embedding | BF16 | maximum-fidelity token lookup |
| LM head | Q8_0 | protect the final logits without imposing Q8 on the transformer trunk |
| one-dimensional norms and biases | F32 | preserved automatically by the quantizer |
| MTP | existing Q8_0 sidecar | held fixed on the RX 7900 XT; not rebuilt in H1 |

The exact selective rules are in `quantization/h1.tensor-types.txt`. The base preset
is `Q4_0_ROCMFP4_STRIX`; do **not** add `--pure`, because pure mode suppresses the
preset mixture. The workflow also deliberately omits `--allow-requantize`: H1 must
not be made from the current FP4 GGUF.

### 1. Stream a quantization source

Activate the existing conversion environment, update this repository, and run:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
chmod +x run-quant.sh
python3 qwen_quant.py self-test
./run-quant.sh preflight --phase convert
./run-quant.sh convert
```

The default source is the official `Qwen/Qwen3.8-Flash-Next-FP8` release. The fork
reads its safetensors remotely, dequantizes ordinary tensors into a split BF16 GGUF,
and applies the FP8 PLE table's scalar scale before writing its 16 heads directly as
Q8_0. This is a BF16 *intermediate derived from the official FP8 checkpoint*, not a
claim that it came from the larger original BF16 checkpoint. This route is selected
because the fork's bounded-memory PLE conversion is specifically validated for that
release; pointing it at the plain BF16 repository currently bypasses that tested
scale path.

The source is expected to occupy roughly 300+ GiB, the temporary PLE map can reach
about 51 GiB during conversion, and H1 adds roughly another 125 GiB while it is being
written. The conversion preflight therefore requires 500 GiB free by default so the
source does not strand the subsequent quantization. Override the threshold only
after accounting for all three filesystems and temporary paths. Remote mode does not
cache the weight shards locally, but it does cache configuration/tokenizer files.

### 2. Generate the importance matrix

Use a representative, redistributable plain-text calibration corpus containing the
actual mix of code, JSON, technical prose, and reasoning prompts this machine will
serve. A few benchmark prompts are not enough; the default collects 200 chunks.

```bash
./run-quant.sh preflight --phase imatrix \
  --calibration-file /srv/llm/models/qwen-flash-next/calibration/h1-corpus.txt

./run-quant.sh imatrix \
  --calibration-file /srv/llm/models/qwen-flash-next/calibration/h1-corpus.txt
```

The matrix is collected with the current known-working PLE16 model at the measured
82/18 Vulkan placement. That makes the calibration run fit this host and captures
real routed-expert activations. It is an approximation to the release-weight
activations, but is preferable to spending a full quantization run with no matrix.
The quantizer refuses to proceed without it unless `--allow-no-imatrix` is explicitly
passed.

### 3. Plan, build, and verify H1

```bash
./run-quant.sh preflight --phase quantize
./run-quant.sh dry-run
./run-quant.sh quantize
./run-quant.sh verify
```

The dry run prints every final tensor decision and the estimated file size. The
build uses the fork's 512 MiB row bands and streamed GGUF writes, so the 128 GiB host
does not need to materialize the BF16 model or the 51.2B PLE tensor in memory. Every
action writes its exact command, llama.cpp revision, log, and exit status below
`/srv/llm/models/qwen-flash-next/h1-build/`. Verification checks the five selective
families, token embedding, LM head, norms/biases, attention type mixture, absence of
embedded MTP tensors, and importance-matrix provenance.

### 4. Compare performance without changing placement

```bash
python3 qwen_bench.py preflight --tier vulkan-quant-h1
./run-bench.sh --tier vulkan-quant-h1
```

This tier compares the current model with H1 at the identical 82/18 target split,
F16 KV, Q8 MTP on the 7900 XT, n=4, p-min 0.75, and ubatch 2048. It measures code,
JSON, and prose at approximately 4K, 16K, and 32K prompt depth, including both
prefill and fixed-length decode. Different output hashes are expected between two
quants; inspect saved responses and use a separate perplexity/task evaluation before
calling H1 a quality win.

## Recommended benchmark sequence

Start with `vulkan-baseline` below. Earlier bring-up established that nominal 90/10,
88/12, 85/15, and 82/18 forward splits have effectively the same decode rate on
this host. The active tiers therefore use 88/12 as the forward representative and
spend measurement time on structurally different placements and real prefill loads.
The other ratios remain defined for deliberate boundary investigations, but are not
part of the recommended campaign.

The expert experiments intentionally use `--split-mode layer --tensor-split 1,0`, even though they are not conventional layer splits. In llama.cpp, `--split-mode none` removes every model GPU except `--main-gpu` from the scheduler. Layer mode keeps both Vulkan/ROCm backends registered; `1,0` leaves all ordinary layers on device 0 while `--override-tensor` alone moves PLE and routed-expert tensors to device 1. The harness rejects non-CPU tensor overrides combined with split mode `none` during configuration loading.

The original expert strategy does not move the shared experts. To directly test
PLE plus routed and shared experts on the iGPU, with attention and remaining target
tensors on the 7900 XT, run:

```bash
python3 qwen_bench.py preflight --tier vulkan-expert-shared
./run-bench.sh --tier vulkan-expert-shared
```

This is a two-cell F16-KV A/B, not another broad placement sweep. The control moves
only PLE and `ffn_*_exps`; the candidate additionally moves `ffn_*_shexp` and
`ffn_gate_inp_shexp`. Both use identical code/prose prompts at approximately 4K
and 16K depth, an excluded 16K warm-up, and 128 forced output tokens.

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

### F16 KV: MTP device and target-placement decision

The earlier MTP tests always put the draft sidecar on Vulkan0 (the 7900 XT); they
did not compare it with MTP on Vulkan1 (the iGPU). Run the direct six-cell matrix:

```bash
python3 qwen_bench.py preflight --tier vulkan-f16-mtp-placement
./run-bench.sh --tier vulkan-f16-mtp-placement
```

This holds F16 K/V cache, prompts, and output length constant while crossing two
target placements (entire target on the iGPU, or the proven 82/18 iGPU/dGPU layer
split) with three draft choices (no MTP, MTP on the iGPU, or MTP on the 7900 XT).
It measures code, JSON, and prose at approximately 4K and 16K prompt depth, with
one excluded 16K warm-up and 512 forced output tokens. The longer output makes
total request time, decode throughput, draft acceptance, and prefill overhead
directly comparable. Fail-fast is intentionally omitted: the 82/18 target plus
dGPU MTP cell may exceed the 7900 XT's available VRAM, and that failure is useful
capacity data while the remaining cells continue.

The first placement pass selected the 82/18 target split with F16 K/V cache and
MTP on the 7900 XT. Once that result is reproduced on a host, tune only the two
remaining speculative-decoding controls:

```bash
python3 qwen_bench.py preflight --tier vulkan-mtp-tuning
./run-bench.sh --tier vulkan-mtp-tuning
```

This retains a no-MTP control and makes three isolated decisions: n=3 versus n=4
at `--spec-draft-p-min 0.75`, p-min 0.50 versus 0.75 at n=4, and ubatch 512 versus
1024 versus 2048 on the n=4/p-min-0.75 winner. Target placement, F16 cache, draft
device, prompts, 16K warm-up, and 512-token output are otherwise identical. The
ubatch sweep is not multiplied across the MTP parameter candidates, and the tier
does not repeat device-placement or layer-ratio tests.

Ubatches above 2048 require a matching logical batch size. The measured 8192 cell
lost the Vulkan device and is no longer defined. To decide whether 4096's larger
batch can pay off only at much longer prompts, compare 2048 and 4096 directly at
approximately 32K requested depth:

```bash
python3 qwen_bench.py preflight --tier vulkan-ubatch-32k
./run-bench.sh --tier vulkan-ubatch-32k
```

The server context is 65536 so the actual tokenized prompt plus 512 forced output
tokens fits safely. One code workload, two rounds, a 4K excluded warm-up, and
rotated experiment order isolate the ubatch decision without rerunning the earlier
workload matrix. The 4096 cell explicitly sets both batch and ubatch to 4096; the
2048 control uses llama.cpp's 2048 logical batch with ubatch 2048.

### PLE n-gram mmap/SSD residency

`--load-mode mmap` is already the harness default, but `=CPU` alone does not prove
that the n-gram table is being served from SSD rather than resident RAM. The exact
tensor name also differs between the original joined GGUF and this repository's
Vulkan-compatible PLE16 file. Run the hot two-cell residency and throughput test:

```bash
python3 qwen_bench.py preflight --tier vulkan-ple-ssd
./run-bench.sh --tier vulkan-ple-ssd
```

The control uses PLE16 automatic placement. The candidate applies
`^ple_ngram_embd\.[0-9]+\.weight$=CPU` to all PLE16 shards. Everything else stays
at 82/18, F16 KV, dGPU MTP n=4, p-min 0.75, and ubatch 2048. Each server receives
an excluded 4K warm-up; its slot is erased before measured 4K, 16K, and 32K
prefills. Physical reads and faults in each measured row verify whether the page
working set was actually hot.

The original joined GGUF is no longer part of the tier. Its matched CPU override
still makes Vulkan reserve a roughly 55 GiB compute buffer on the iGPU, including
a single roughly 6.8 GiB allocation that exceeds the backend limit. PLE16 is the
required split representation for this Vulkan path.

Cold first-request behavior is intentionally kept out of the hot throughput
ranking. Measure it separately, with one prompt depth per server:

```bash
python3 qwen_bench.py preflight --tier vulkan-ple-ssd-cold-4k
./run-bench.sh --tier vulkan-ple-ssd-cold-4k

python3 qwen_bench.py preflight --tier vulkan-ple-ssd-cold-16k
./run-bench.sh --tier vulkan-ple-ssd-cold-16k
```

These tiers do not globally drop Linux caches. Treat a sample as physically cold
only when its storage-read and major-fault counters confirm it; "server-cold" is
not necessarily "page-cache-cold." Run the 4K and 16K tiers independently after
the file pages have been reclaimed; running them back-to-back normally makes the
second one hot. Each cold tier uses one round with the CPU/mmap candidate first.
The summary reports file-backed and anonymous RSS, host available/cached memory,
per-device VRAM/GTT peaks, storage bytes read, and major faults. File-backed cache
is reclaimable, so capacity conclusions should use `MemAvailable` and device
allocations rather than process RSS alone.

### Recommended two-user production deployment

The production candidate keeps the split PLE16 table CPU-mapped, puts the Q8 MTP
sidecar on the RX 7900 XT, uses F16 target KV, and gives each of two independent
slots 262,144 tokens by setting total context to 524,288 with `--parallel 2`.
The preferred target split is 88/12; 90/10 is the lower-VRAM fallback. Both use
batch/ubatch 2048, continuous batching, separate per-slot KV, no prompt-response
RAM cache, and no context shifting.

Do not start with a near-full prompt. First prove that both complete server
allocations fit:

```bash
python3 qwen_bench.py preflight --tier production-capacity
./run-bench.sh --tier production-capacity
```

This tier loads 90/10 and 88/12 separately at the complete two-slot allocation,
waits for steady telemetry, records startup time and memory/device residency, and
stops without submitting a completion. Use 88/12 only if it loads with useful
7900 XT headroom; otherwise use 90/10. A load success is necessary but does not
prove simultaneous-request stability.

Next run two real requests together. The selector avoids loading both candidates
again after the capacity result has chosen one:

```bash
./run-bench.sh --tier production-concurrency-smoke \
  --experiments prod_88_12_vk_mtp_n4_f16kv_dgpu_ub2048_ple16_cpu \
  --fail-fast
```

For the fallback, replace `prod_88_12` with `prod_90_10` in that experiment name.
The two workers synchronize before sending independent code and prose prompts.
The runner reports per-user latency plus aggregate prefill, aggregate decode, and
end-to-end throughput in `concurrency.csv` and `summary.md`; this is not two
serial requests against a parallel server.

The default production candidate uses `--no-kv-unified` for isolated per-slot KV.
If the server log shows one prompt monopolizing batching while the other slot
barely advances—or a decode-fairness ratio far below 1.0—test the matched unified
KV candidate before changing precision or batch size:

```bash
./run-bench.sh --tier production-concurrency-unified-smoke --fail-fast
```

The harness removes the inherited `--no-kv-unified` flag, so the command contains
only `--kv-unified`. Conservative aggregate rates divide total work by the sum of
per-lane phase times; separate overlap-upper-bound columns are reported because
per-request timing alone cannot prove that two prefill or decode phases overlapped.
The slow-lane decode rate and fairness ratio are the primary admission criteria.

If unified KV does not improve fairness, keep separate KV and isolate scheduler
admission from memory pressure:

```bash
./run-bench.sh --tier production-concurrency-scheduler-memory-smoke --fail-fast
```

This tier first raises only the logical batch from 2048 to 4096 while keeping
the physical ubatch at 2048. A logical batch of 2048 can be consumed by one
slot's 2048-token prompt chunk; the larger logical batch tests whether both
slots can be admitted without requesting the unstable 4096-token physical
ubatch. The second experiment changes only target KV from F16 to Q8_0. Compare
slow-lane decode, fairness, host cached GiB, storage-read GiB, and major faults.
Q8 KV is a production candidate only if it restores residency and passes output
quality validation; it is not assumed equivalent to F16.

Only after smoke passes, validate the actual context target:

```bash
./run-bench.sh --tier production-concurrency-full \
  --experiments prod_88_12_vk_mtp_n4_f16kv_dgpu_ub2048_ple16_cpu \
  --fail-fast
```

That tier runs synchronized pairs at exactly fitted prompt budgets of 32,768,
131,072, and 253,952 tokens per lane. The `/tokenize` endpoint is used before
measurement, so the near-full case retains margin for 256 generated tokens inside
each 262,144-token slot. Slot KV is erased after every pair without dropping the
model mapping or Linux page cache.

After the chosen split passes, install the checked-in systemd deployment:

```bash
sudo cp deployment/qwen-flash-next.env.example /etc/qwen-flash-next.env
sudo cp deployment/qwen-flash-next.service /etc/systemd/system/
sudo chmod 640 /etc/qwen-flash-next.env
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-flash-next.service
systemctl status qwen-flash-next.service
```

Edit `/etc/qwen-flash-next.env` before starting if capacity selected 90/10, the
paths differ, or the server must listen somewhere other than
`127.0.0.1:8080`. The launcher refuses any non-loopback bind without `API_KEY`.
Keep the service on loopback behind an authenticated reverse proxy or Tailscale
when possible. Metrics are available from llama-server's `/metrics` endpoint.

### Focused 7900 XT prefill screen

MTP does not accelerate target-model prefill: the target remains on the iGPU and
the draft model can add its own prompt-processing work. Test target prefill
placement separately:

```bash
python3 qwen_bench.py preflight --tier vulkan-prefill
./run-bench.sh --tier vulkan-prefill
```

This screen keeps Q8 KV and disables MTP everywhere. It compares APU-only against
one maximum-known-to-load 82/18 contiguous split, the component/expert split, reverse layers,
and both row modes. These are structurally distinct; neighboring layer ratios are
not repeated. One fixed code prompt is measured at approximately 4K and 16K depth.
Each server first processes the same 16K prompt as an excluded warm-up, preventing
cold page-cache/shader effects from masquerading as a placement result. Generation
is limited to eight forced tokens and is diagnostic only; rank this tier by prefill
tok/s and prefill milliseconds.

### ROCm 10 recovery: prove kernels before loading the model

The old HIP result is not a meaningful backend comparison. The current branch
contains standalone ROCmFP4 source, but the observed build did not establish that
types 100/101 dispatch into compiled HIP matrix kernels. Running the same 121 GB
model again would only repeat the catastrophic scalar/fallback path.

Start with the static audit. It deliberately exits nonzero on the currently known
bad build and writes a detailed report instead of loading a model:

```bash
python3 qwen_bench.py rocm-audit
```

The audit counts ROCmFP4 and ROCmFP4_FAST integration only under the compiled
`ggml/src/ggml-cuda` or `ggml/src/ggml-hip` runtime trees. A standalone
`ggml/rocmfp4` file is explicitly insufficient. After porting the accelerated HIP
kernels into this Qwen/PLE16 branch, build the operation test binary and run the
functional gate:

```bash
cmake --build /srv/llm/src/llama-qwen4exp/build-hip10 \
  --target llama-server test-backend-ops -j16
python3 qwen_bench.py rocm-audit --run-ops
```

The functional audit filters `MUL_MAT` and `MUL_MAT_ID` for Q8_0,
Q4_0_ROCMFP4, and Q4_0_ROCMFP4_FAST on both `ROCm0` and `ROCm1`. It rejects a
zero-test match, any numerical failure, or a stale server/HIP-library fingerprint.
Every ROCm model tier is gated on that saved proof, so an expensive full-model run
cannot begin accidentally after an unverified rebuild.

Once the audit passes, use the clean APU-only diagnostic:

```bash
python3 qwen_bench.py preflight --tier rocm-smoke
./run-bench.sh --tier rocm-smoke --fail-fast
```

This path hides the 7900 XT with `ROCR_VISIBLE_DEVICES=1` (therefore the physical
8060S becomes logical `ROCm0`), disables managed-memory fallback, keeps the split
PLE table CPU-mapped, and uses neither MTP nor heterogeneous placement. It tests
only shallow 0/512-token depths so kernel correctness and basic throughput are
isolated first.

If shallow performance is sane, localize the previously reported context cliff:

```bash
./run-bench.sh --tier rocm-depth --fail-fast
```

That single-variable tier stays APU-only and sweeps 0, 512, 1024, 1536, 2048,
4096, 8192, and 32768 requested tokens. Do not proceed to `rocm-placement` or
`rocm-mtp` until this curve remains sane. Only then should multi-GPU placement be
reintroduced, followed last by the Q8 MTP sidecar on the 7900 XT.

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
warmups.jsonl          excluded warm-up timings, I/O, faults, and slot erasure
summary.csv            one median row per experiment/workload/depth
summary.md             human-readable ranking and failure list
logs/                  complete stdout/stderr for each server start
responses/             complete JSON response for each measured request
telemetry/             raw one-second telemetry samples
system/                device listings and host/software provenance
```

In `summary.csv`, `gpu_busy_mean`, `gpu_vram_max_gib`, and `gpu_gtt_max_gib` are
JSON maps keyed by PCI BDF, which avoids assuming that Linux card numbering is
stable. The Markdown summary also includes a dedicated residency/capacity table.
The 7900 XT should map to `0000:c7:00.0` in the current topology; verify this in
`preflight.json` rather than hard-coding it in analysis.

`pcie_speed_gt_s_max=16` and `pcie_width_lanes_max=4` would confirm the expected Gen4 ×4 host uplink under load. If the request telemetry never rises above 2.5 GT/s ×1, inspect the raw telemetry and `lspci` output before drawing any conclusion about the split. The sampler first uses non-interactive `lspci`; it never prompts for `sudo`.

The `gpu_pcie_*_est_max_mib_s` fields come from AMDGPU's `pcie_bw` sysfs counters. The [kernel documentation](https://docs.kernel.org/gpu/amdgpu/driver-misc.html#pcie-accounting-information) says these are received/sent message counts for the last second plus maximum payload size, not exact byte counters. The harness therefore reports `messages × MPS` explicitly as an estimate/upper bound. It is most useful for comparing relative traffic between placements.

An output-hash mismatch is a correctness warning, not automatic proof of an invalid result: different backends can cross a greedy-token decision boundary because of floating-point differences. Inspect the saved response. MTP itself should preserve target-model sampling semantics, so systematic MTP-only mismatches deserve investigation.

## Adding custom quants or placements

Duplicate an experiment object in `matrix.json`, give it a unique name, change its `model` or arguments, and add that name to a tier. A compact variant can use `"extends": "expert_vk_no_mtp"` plus `args_append` and/or `env`; inheritance is resolved into the manifest before a run starts. Keeping the same workloads and greedy request parameters makes custom-quant comparisons directly reportable by the existing summarizer.

For a quant-quality study, this performance harness should be paired with a fixed evaluation corpus and perplexity or task-accuracy measurements. Exact output hashes catch accidental behavioral differences but are not a quality metric.
