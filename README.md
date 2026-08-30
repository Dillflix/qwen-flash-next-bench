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

## Vision bring-up

Qwen3.8-Flash-Next is multimodal, but llama.cpp stores its vision encoder/projector
separately from the language-model GGUF. The `kingjones777` ROCmFP4 repository does
not include that projector. This harness uses the matching BF16 projector from
Unsloth as the quality reference and the matching Q8 projector published by
`ggml-org` as a memory/performance control.

Prepare both checksum-pinned projectors and three deterministic local PNG fixtures:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
python3 qwen_vision.py self-test
python3 qwen_vision.py prepare
```

The download is about 1.5 GB total. `prepare` writes only under
`/srv/llm/models/qwen-flash-next`, verifies SHA-256, and generates synthetic cases
for colored shapes plus text, invoice OCR, and chart reasoning. It is safe to rerun;
valid existing files are not downloaded again. To fetch only the smaller projector:

```bash
python3 qwen_vision.py prepare --projectors q8
```

Run the one-image bring-up first:

```bash
python3 qwen_bench.py preflight --tier vision-smoke
./run-bench.sh --tier vision-smoke --fail-fast
```

This compares three isolated server starts:

1. BF16 projector on CPU, which is the correctness reference;
2. BF16 projector on the RX 7900 XT;
3. Q8 projector on the RX 7900 XT.

The dGPU experiments set `MTMD_BACKEND_DEVICE=Vulkan0` explicitly. `--main-gpu`
does not select the projector device. All experiments keep the target model on the
88/12 iGPU/dGPU split, PLE16 on CPU mmap, Q8 target KV, and a 2048-token batch.

Run the complete fixture set before selecting a projector. The measured 7900 XT
result chose BF16: all six samples passed, projector prefill was about 7.1x the
CPU reference, and Q8 repeatedly mistranscribed `UNSLOTH 42` as `UN5LOTH 42` while
saving only about 0.3 GiB of VRAM. Q8 is therefore not a production candidate for
this model/build. Test MTP only with the passing BF16 projector:

```bash
./run-bench.sh --tier vision
./run-bench.sh --tier vision-mtp
```

Vision results add a dedicated summary table with known-fact anchor score, image
encode/decode/total milliseconds parsed from the server log, visual prompt tokens,
prompt time, HTTP wall time, GPU residency, and the existing exact output hashes.
The anchor score is the primary correctness gate because numerically equivalent
backends can produce differently worded answers. Review the archived response JSON
before accepting Q8 or Vulkan for production.

Vision tiers pass `chat_template_kwargs.enable_thinking=false` so a fixed decode
budget measures the answer rather than being consumed by Qwen's default internal
reasoning. A response with reasoning tokens but an empty final answer is marked
degenerate and excluded from rankings.

Every vision tier also declares a minimum anchor score. Responses below that gate
are visibly marked as quality failures and excluded from performance rankings, so
a fast but incomplete description cannot win. An experiment with any measured
anchor failure is also marked `DISQUALIFIED` in the overall table; its passing-cell
throughput remains visible for diagnosis. The shapes fixture requests a compact
one-line answer to ensure all known facts fit inside the fixed decode budget.

The MTP tier is deliberately separate. Multimodal plus MTP has had model- and
backend-specific failures in llama.cpp, so a successful projector-only run is a
hard prerequisite; `vision-mtp` is a compatibility gate, not an assumption that
the combination is safe.

The Vulkan tiers above remain useful historical controls. ROCm smoke run
`20260829-211352-rocm-vision-smoke` proved that the native joined model and BF16
projector are correct on ROCm. The CPU projector passed with 16.060 s image time,
66.67 prompt tok/s, and 29.22 decode tok/s. Explicitly placing the same projector
on the iGPU (`MTMD_BACKEND_DEVICE=ROCm1`) also passed, reducing image time to
2.368 s while reaching 406.60 prompt tok/s and 32.39 decode tok/s. The iGPU BF16
projector is therefore the production placement; spending 7900 XT VRAM on it is
both unnecessary and contrary to the fixed MTP/target placement.

The combined iGPU-projector plus n=3 MTP cell did not run out of memory. It
finished the target image decode, immediately tried to decode the same raw image
embedding batch on the Qwen4Exp MTP draft context, and aborted at
`qwen4exp MTP requires token input`. This is the same unsupported speculative
multimodal boundary tracked by llama.cpp
[issue #19712](https://github.com/ggml-org/llama.cpp/issues/19712) and the
MTP-specific failure in
[issue #22867](https://github.com/ggml-org/llama.cpp/issues/22867).

The first narrow workaround skips only the invalid raw-image decode on a
target-conditioned draft context. The target context still decodes the image and
the existing MTP boundary-resync path seeds the next draft boundary. Run
`20260829-215638-rocm-vision-mtp-resync-smoke` proved that this removes the crash,
but it did **not** prove correctness: both warm-up and measured responses read
`UNSLOTH 1` instead of `UNSLOTH 42` (6/7 anchors), despite 84.1% MTP acceptance.
The result was also slower than the earlier target-only control on this short
answer (24.39 versus 32.39 decode tok/s).

Because the requests are greedy, an ungrounded draft should reduce acceptance,
not change the target answer. The remaining divergence is therefore tested at
the target verifier. Qwen4Exp is recurrent, but the pinned server's existing
single-row exact-verification path recognized HY3 only. The first strict attempt
correctly selected that path but immediately asserted at
`n_ubatch > n_keep_tail`: n=3 bounded rollback requires four trailing rows to
remain in one microbatch, so it cannot also perform one-row verification.

The checkpoint follow-up resolves that incompatibility explicitly. The harness
passes `--spec-mtp-strict-qwen4exp-vision`, which disables bounded recurrent
rollback only for this Qwen4Exp vision context and uses the server's full-state
speculative checkpoints instead. Target verification can then run one row at a
time. `LLAMA_CKPT_FORCE_HOST=1` keeps both control and MTP checkpoint state in
host/unified RAM rather than spending 7900 XT VRAM. Text-only MTP and every
non-vision topology remain unchanged. Rebuild,
re-audit, and run the matched A/B:

```bash
./build-rocm10-dual.sh
python3 qwen_bench.py rocm-audit --run-ops
python3 qwen_bench.py preflight --tier rocm-vision-mtp-strict-ab
./run-bench.sh --tier rocm-vision-mtp-strict-ab --fail-fast
```

The A/B uses two fresh-server rounds of the same image, target model, iGPU BF16
projector, sampler, and 64-token budget. The target-only arm is the explicit output-hash
baseline. The MTP arm must emit the runtime resync, single-row verification, and
checkpoint-backed rollback markers; 100% anchors and hash agreement are the
correctness criteria. Compare decode speed only after those pass. If
strict verification restores correctness but loses to target-only, production
vision should automatically omit MTP rather than accept a quality regression.
Both arms explicitly set the prompt-checkpoint cadence to 32768 tokens. That
keeps the host-checkpoint safety gate reproducible without creating an unrelated
periodic prompt checkpoint in this 8192-token qualification context; the MTP
arm's per-draft rollback checkpoint is still forced into host/unified memory.

The completed strict A/B restored 100% vision anchors, but proved the strict
checkpoint path unsuitable for production: 13.99 tok/s median decode versus
32.52 tok/s for target-only, a 57% regression. Prefill and image processing were
effectively unchanged. A first per-request-disable A/B then showed that
`"speculative.n_max": 0` suppressed all draft tokens and preserved 100% anchors,
but did not make the loaded sidecar inert: median decode was 24.93 tok/s versus
31.90 tok/s for target-only, a 21.8% regression. Prefill and image processing
again matched, isolating the cost to decode-time target hidden-state export and
speculative post-processing that were still enabled by the global sidecar.

The first follow-up server patch made request-level disabling complete. When the
request's effective `speculative.n_max` was zero it suppressed speculative target
embeddings, pre-normalized hidden-state output, and post-processing; restored
backend sampling; and prevented enabled and disabled requests from sharing the
same target batch. MTP state is still cleared safely on slot reset. At that
point text requests with n=3 were intentionally unchanged; later long agentic
testing showed that path was not production-safe. The resulting vision A/B
proved the target-only path, but requiring every caller to know this backend
detail is not a production API.

Version 1.43 moves the decision into the server scheduler under the production
`LLAMA_AUTO_DISABLE_SPEC_MULTIMODAL=1` policy. It detects actual media chunks in
the incoming prompt, forces that request's draft budget to zero before
sampler/graph setup, and emits an automatic
bypass marker. This remains useful to experimental MTP servers; the production
launcher now disables MTP for all requests. The A/B deliberately sends no
`speculative.n_max` field, so it fails if client knowledge is still required:

```bash
./build-rocm10-dual.sh
python3 qwen_bench.py rocm-audit --run-ops
python3 qwen_bench.py preflight --tier rocm-vision-mtp-auto-bypass-ab
./run-bench.sh --tier rocm-vision-mtp-auto-bypass-ab --fail-fast
```

The harness fails the candidate if the response reports any drafted tokens or
the request log lacks both automatic-detection and target-work-bypass markers.
Production can use one loaded server for fast text MTP and exact target-only
vision only if this arm matches the control hash and target-only speed.

The patched A/B passed. Across two fresh-server rounds, target-only median decode
was 32.17 tok/s and the MTP-loaded/request-disabled median was 32.20 tok/s.
Median prefill was 399.13 versus 397.69 tok/s (-0.36%), and median image time was
2370.5 versus 2378.5 ms (+0.34%). Both arms retained 100% anchors; the candidate
reported zero drafted tokens and emitted the required bypass marker. The former
21.8% decode penalty is gone.

The final 262144-token tier keeps the sidecar loaded on the 7900 XT for n=3 text
MTP while sending ordinary vision requests with no speculative override. It
tests all three fixtures against a matched target-only control. The remaining
production topology stays fixed: 88/12 target split, ubatch 1536, CPU-mmap PLE,
host checkpoints, iGPU BF16 projector, and F16 target/draft KV. Run it after the
automatic-bypass A/B passes:

```bash
python3 qwen_bench.py preflight --tier rocm-vision
./run-bench.sh --tier rocm-vision --fail-fast
```

Run `20260830-024815-rocm-vision` qualified the underlying explicit-bypass
behavior:
all 12 measured responses passed their anchors, and the MTP-loaded/bypassed arm
matched the target-only output hash in every cell. Overall it was 1.005x decode,
0.990x prefill, and 0.998x balanced versus target-only. The bypass marker was
present on every candidate request and the draft engine reported zero calls and
zero generated or accepted draft tokens. At the native 256K allocation the
loaded sidecar used 18.97 GiB of 7900 XT VRAM versus 12.43 GiB without MTP, while
minimum host-available memory remained about 54 GiB.

That archive also revealed that the old vision warm-up covered only the first
fixture and only 32 output tokens. Consequently the nominally hot measured
`invoice` and `chart` requests still read roughly 10.5 and 6.3 GiB from storage;
even `shapes` read about 1 GiB after its partial warm-up. Version 1.42 changes
the full vision tier to a full-budget, per-cell warm-up immediately before each
measurement. The next run now has two purposes: storage-hot characterization
and qualification of the new server-side automatic media routing. Model quality,
placement, and the zero-draft execution path are already qualified.

Both ROCm tiers explicitly use F16 target KV and F16 draft KV. Q8 draft KV is not
present. The Q8 projector is also excluded because the earlier Vulkan run failed
the OCR quality gate; that is independent of the draft-KV decision. Strict
checkpoint-backed vision MTP remains available only as a diagnostic: production
vision is target-only execution inside the shared MTP-loaded server.

The projector is deliberately not placed on the 7900 XT: its scarce VRAM is reserved
for MTP and the selected target tensors. CPU versus iGPU projector placement remains
an explicit correctness/performance comparison.

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

### Qualified ROCm production deployment

The checked-in launcher reproduces the configuration that passed 256K
allocation, near-full-context, and vision qualification. It now defaults to
**target-only text generation** because a real long agentic prompt produced a
complete, coherent response with `speculative.n_max=0` but structurally corrupt
output with n=3. The earlier throughput and short-prompt MTP runs did not prove
lossless generation and are not sufficient production qualification.

- joined ROCmFP4 model with its 51.2B PLE tensor CPU-mapped;
- ROCm devices ordered `ROCm0,ROCm1`, with an 88/12 target split and routed
  experts forced to the iGPU;
- one 262,144-token slot, F16 target KV, batch 2048, and ubatch 1536;
- MTP disabled by default, leaving the Q8_0 sidecar unloaded;
- eight host-resident prompt checkpoints at 32K-token intervals;
- BF16 projector on the iGPU; and
- no prompt-response RAM cache and no context shifting.

This is deliberately a **one-slot** service. Multiple clients may connect, but
requests queue behind the active slot. Two simultaneous 256K slots have not been
qualified and are not silently enabled by the production launcher.

For a foreground launch, provide authentication either as an argument or through
llama-server's native environment variable:

```bash
./deployment/run-production.sh --api-key 'replace-with-a-long-random-key'

# Equivalent, and avoids putting the key in llama-server's command line:
LLAMA_API_KEY='replace-with-a-long-random-key' \
  ./deployment/run-production.sh

# Validate paths, placement, authentication, and scalar settings without loading:
LLAMA_API_KEY='replace-with-a-long-random-key' \
  ./deployment/run-production.sh --check
```

The command-line form is converted to `LLAMA_API_KEY` before `exec`, so the
secret is not present in the resulting llama-server argument list. It can still
be retained by shell history; use the environment or an API-key file for normal
operation. `QWEN_API_KEY` and the old `API_KEY` name are accepted as compatibility
aliases. The launcher refuses a non-loopback bind without an API key or key file
and rejects target splits other than the qualified 88/12 layout. It also pins
the process to `/opt/rocm-10.0.0` by default so systemd cannot resolve a different
host ROCm installation.

`LLAMA_SLOT_SAVE_PATH` is empty by default. The strict MTP diagnostic requires
an existing writable absolute directory there because it erases slot 0 before
every request. Do not set it for normal production: the qualified topology
does not import or persist slot state.

`LLAMA_MTP_MODE=off` is the safe default in the checked-in environment file.
There is an explicit experimental `LLAMA_MTP_MODE=strict` mode which loads the
Q8_0 sidecar on the RX 7900 XT, enables n=3 with F16 draft KV, and requires the
patched boundary-safe Qwen4Exp verification marker. Do not enable it in the
systemd environment until its output matches the target-only control on the
same long OpenWebUI/tool-calling prompts; acceptance rate and token throughput
are not correctness tests.

`LLAMA_MTP_MODE=checkpoint-diagnostic` loads the same sidecar and placement but
replaces bounded recurrent rollback and multi-row target verification with the
already compiled single-row/full-checkpoint path. The internal llama-server
flag retains `vision` in its name, but its single-row decision is based on
speculative verifier rows rather than request media, so it also applies to this
text diagnostic while mmproj is loaded. This is an isolation arm, not a proposed
production setting: if it restores exact n=0/n=1 equivalence, the remaining
fault is in bounded rollback or batched target verification; if it still
diverges, those two mechanisms are not the cause.
`LLAMA_CONTEXT_SIZE` remains fixed at `262144` in production and bounded strict
mode. Only `checkpoint-diagnostic` may lower it; the short isolation command
uses 8192 because its prompt is about 2106 tokens and every known divergence is
before generated token 142. Its n=0 control is run in the same process, so this
reduces full-state checkpoint cost without weakening the within-run equality
test.

Disabling MTP is containment, not the proposed backend fix. The candidate source
fix is `rocmfpx-qwen4exp-mtp-state-correctness.patch`; it repairs rollback-aware
PLE token history, Qwen4Exp recurrent-convolution snapshots, complete verifier
hidden rows, and stale verifier-ring replacement.
The strict boundary mode is layered on top of those repairs and is not a fix by
itself. `qwen_mtp_diag.py` is the token-level qualification gate. The ordinary
text benchmark cannot perform that job: it calls `/completion`, whereas the
observed failure came through OpenAI chat completions with thinking, tool
schemas, and streaming.

This candidate deliberately targets the qualified production topology: one
target context/slot, `--cache-ram 0`, and `--no-context-shift`. Its compact PLE
history is reconstructed or truncated inside the live process; it is not yet a
general serialized llama-state feature. Do not extend strict MTP to persisted
slot-state restore, sequence copy/add, context shifting, or multiple target
contexts until those PLE lifecycle operations have dedicated tests and state
serialization support.

The diagnostic sends one fixed agentic request through the complete 16-cell
matrix:

- `temperature` 0 and 1 with seed 1234;
- streaming and non-streaming OpenAI chat completions; and
- per-request `speculative.n_max` 0, 1, 2, and 3.

Before every request it erases slot 0 and requires the server to report zero
cached prompt tokens. It saves the exact request, raw JSON or SSE body, parsed
SSE events, canonical reasoning/content/tool calls, exact generated token IDs,
byte sequences, available logprobs/top candidates for non-stream requests, HTTP
headers, and a request-scoped server-log slice. Exact IDs and bytes are the hard
identity gate. Missing score/top-candidate detail is reported separately and
cannot make an otherwise comparable row disappear: current llama-server leaves
those arrays empty for tokens emitted by its speculative loop (`TODO: set
result.probs`). This distinction prevents an incomplete-probability warning from
turning a real MTP token divergence into a vacuous PASS.
The API key is never written to the result directory. Exact key occurrences
are redacted from caller-supplied server logs, response bodies/headers, parsed
artifacts, manifests, and result rows before those artifacts are saved or
hashed.
The checked-in fixture validates a *first* agentic turn: nontrivial reasoning is
required, while either a normal answer or a syntactically valid tool request is
allowed. A first-turn `tool_calls` finish is not itself a failure when tools were
offered. Final qualification must additionally replay the actual tool results
and validate the completed multi-turn answer; the synthetic fixture cannot
substitute for that captured exchange.

The checked-in fixture is a sanitized synthetic reproduction with the exact
user prompt and substantial agentic tool schemas. For final qualification,
replace only its `request` object with the exact OpenWebUI POST body captured
from the failing request. Keep that private if its messages or tool definitions
are sensitive; a diagnostic archive contains the complete request by design.
The runner deliberately removes any request-level `reasoning_format` override,
leaving llama-server's automatic Qwen handling in control.

For the diagnostic server, set `LLAMA_TRACE=1` and
`LLAMA_LOG_VERBOSITY=4` (or pass `-lv 4`) and tee stdout/stderr to the file
given to `--server-log`. The trace records accepted-draft blocks while debug
logging records sampled IDs and state transitions, allowing the external first
token mismatch to be correlated with the responsible verification cycle. The
runner parses each request-scoped canonical `accepted X/Y draft tokens` trace
line into structured evidence. It excludes the later debug-only
`new n_tokens` summary for that same cycle, so event counts are not doubled.
The diagnostic also erases slot 0 before every matrix cell. Create a private
temporary directory and set `LLAMA_SLOT_SAVE_PATH` when launching either
diagnostic server; the launcher then passes the required `--slot-save-path`
without enabling slot persistence in normal production:

```bash
install -d -m0700 /tmp/qwen-mtp-diag-slots
export LLAMA_SLOT_SAVE_PATH=/tmp/qwen-mtp-diag-slots
```

Run a one-pass smoke against a foreground server whose stdout/stderr is also
being written to a file:

```bash
read -rsp "API key: " QWEN_TEST_API_KEY
echo
export QWEN_TEST_API_KEY

python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label strict \
  --server-log /tmp/qwen-mtp-strict-server.log

unset QWEN_TEST_API_KEY
```

During source debugging, cap the same matrix at 256 generated tokens. The known
n=1/2/3 greedy divergences all occur before token 142, so this catches the fault
without allowing broken arms to run to the fixture's 4096-token ceiling:

```bash
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label strict-short \
  --server-log /tmp/qwen-mtp-strict-server.log \
  --max-tokens 256
```

Version 1.2 incorrectly marked MTP non-stream rows as errors when only their
top-candidate arrays were absent, then reported equivalence from an empty set of
comparisons. Reclassify such an existing run without any model requests:

```bash
python3 qwen_mtp_diag.py reclassify results/REPLACE-mtp-diagnostic-run
```

This writes `comparisons-reclassified.json` and `summary-reclassified.md` while
preserving the original evidence and report.

To run the first rollback-versus-batch-shape discriminator, start the diagnostic
server exactly as before except for the mode override:

```bash
sudo systemctl stop qwen-flash-next
tmux kill-session -t qwen-mtp-strict 2>/dev/null || true
install -d -m0700 /tmp/qwen-mtp-diag-slots
: > /tmp/qwen-mtp-checkpoint-server.log

tmux new-session -d -s qwen-mtp-checkpoint \
  "bash -lc 'cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench; set -a; source /etc/qwen-flash-next.env; set +a; export LLAMA_MTP_MODE=checkpoint-diagnostic LLAMA_CONTEXT_SIZE=8192 LLAMA_TRACE=1 LLAMA_LOG_VERBOSITY=4 LLAMA_SLOT_SAVE_PATH=/tmp/qwen-mtp-diag-slots; exec ./deployment/run-production.sh >>/tmp/qwen-mtp-checkpoint-server.log 2>&1'"

for _ in {1..60}; do
  grep -q "listening on" /tmp/qwen-mtp-checkpoint-server.log && break
  tmux has-session -t qwen-mtp-checkpoint 2>/dev/null || {
    tail -n 100 /tmp/qwen-mtp-checkpoint-server.log
    exit 1
  }
  sleep 5
done
grep -q "listening on" /tmp/qwen-mtp-checkpoint-server.log || {
  tail -n 100 /tmp/qwen-mtp-checkpoint-server.log
  echo "checkpoint diagnostic server did not become ready within 300 seconds" >&2
  exit 1
}
```

Run the 256-token matrix and package it:

```bash
read -rsp "API key: " QWEN_TEST_API_KEY
echo
export QWEN_TEST_API_KEY
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label checkpoint-short \
  --server-log /tmp/qwen-mtp-checkpoint-server.log \
  --max-tokens 256
unset QWEN_TEST_API_KEY

RUN=$(ls -dt results/*-mtp-diagnostic-checkpoint-short | head -1)
python3 qwen_bench.py archive "$RUN"

tmux kill-session -t qwen-mtp-checkpoint
sudo systemctl start qwen-flash-next
```

That checkpoint run localized the earliest failure more tightly than rollback:
greedy `n_max=1` diverged at zero-based output token 37 after 23 complete 1/1
acceptance events, while the first rejected draft occurred only after at least
46 speculative outputs. The mismatch therefore predates both bounded rollback
and full-checkpoint restore. The selected token's real top-candidate score also
already changes at output token 0 when MTP is enabled (`-0.427770` to
`-0.273570`), before any draft can be verified. The next test must isolate the
target hidden-state-export graph and the verifier's logical two-row decode; it
must not repeat the full matrix.

Version 1.4 adds a reduced `greedy-n01` profile and three opt-in server
diagnostics. None is enabled in production:

- `LLAMA_MTP_DIAG_FORCE_TARGET_EXPORT=1` makes the per-request `n_max=0`
  control export the same target pre-norm hidden row as an MTP request while
  still generating no drafts.
- `LLAMA_MTP_DIAG_OUTER_SERIAL=1` replaces the existing one-logical-batch,
  `n_ubatch=1` verifier with independent one-token outer decode calls, then
  reconstructs the logits and pre-norm rows consumed by the unchanged verifier.
- `LLAMA_MTP_DIAG_TRACE_TARGET_LOGITS=1` logs each target row's raw top-two
  token IDs, logits, and margin before sampling.

Version 1.5 distinguishes MTP divergence from a non-repeatable target baseline.
If greedy `n_max=0` changes across fixed-seed repeats, cross-`n_max`
equivalence is reported as `INCONCLUSIVE` instead of attributing the mismatch
to MTP. The `--prime-requests` option runs archived, request-identical greedy
requests before the measured matrix. This isolates the Qwen4Exp cold-start
signature where the first long prompt has different raw target logits from
every later request despite the server's empty model warm-up.

After the unprimed forced-export arm, run the same matrix with one exact-prompt
prime:

```bash
read -rsp "API key: " QWEN_TEST_API_KEY
echo
export QWEN_TEST_API_KEY

python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label export-n01-primed \
  --server-log /tmp/qwen-mtp-export-server.log \
  --matrix-profile greedy-n01 \
  --prime-requests 1 \
  --repeats 3 \
  --max-tokens 40

unset QWEN_TEST_API_KEY
RUN=$(ls -dt results/*-mtp-diagnostic-export-n01-primed | head -1)
python3 qwen_bench.py archive "$RUN"
```

The priming response and its request-scoped server log are retained and indexed
by `priming.json`, but excluded from measured equivalence and repeatability
counts. A passing primed arm proves whether forced target export restores
n=0/n=1 parity independently of the startup-only target-state defect; it does
not excuse that defect for production.

The forced-export arm proves graph-path parity only by making the n=0 control
use the hidden-state-export graph. It is not a production fix: ordinary target
decoding must remain the reference. The next candidate changes only Qwen4Exp's
target graph construction order. Target logits are registered as the primary
graph output before the pre-norm hidden row is appended for MTP. This tests
whether registering the hidden view first changed ROCm fusion, allocation, or
in-place scheduling and therefore changed target logits.

After rebuilding and passing `rocm-audit --run-ops`, start the diagnostic server
with `LLAMA_MTP_DIAG_TRACE_TARGET_LOGITS=1` but **without**
`LLAMA_MTP_DIAG_FORCE_TARGET_EXPORT`. Run the same primed `greedy-n01` profile.
Exact n=0/n=1 parity in that arm qualifies the output-order candidate; failure
means the hidden row needs a non-aliasing copy or a separate scheduling fix.

The `20260830-180244-mtp-diagnostic-export-order-n01-primed` result rejected
the output-order change as sufficient. The explicit n=0 prime and first measured
n=0 request had the same `2a717ae6184e` token trace and first-token logprob
`-0.427770`. The first n=1 request changed to trace `bb3db103854a` and logprob
`-0.273570`; every subsequent n=0 request retained that second trace. This is
not a cold-start outlier. The first MTP request permanently changed the target
process's numerical regime.

Request logs expose the cause: n=0 produced 40 target-logit fingerprints for
one prompt output plus 39 generated positions, while every n=1 arm produced
2,145 fingerprints: all 2,106 prompt positions plus 39 generated positions.
The server marked every prompt token as a vocabulary-logit output merely to
obtain MTP's unmasked pre-norm rows. That unnecessarily enlarged the target
output graph and graph allocator; later n=0 requests then used the altered
allocation regime. The `rocmfpx-mtp-prompt-logit-mask.patch` candidate leaves
prompt logits last-token-only. Unmasked pre-norm extraction already copies all
token rows through its separately sized buffer, so it does not require
`batch.logits` on every prompt token.

After rebuilding, repeat the primed `greedy-n01` matrix without forced target
export. A qualified run must meet all three conditions:

- the prime, every n=0 repeat, and every n=1 repeat have one exact token trace;
- n=1 request logs contain one prompt target-logit row, not 2,106 rows;
- the first-token target distribution stays on the ordinary n=0 fingerprint
  rather than transitioning after the first MTP request.

Rebuild and rerun the ROCm audit first because this diagnostic adds code to both
`libllama.so` and `llama-server`:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
git pull --ff-only
python3 qwen_mtp_diag.py self-test
./build-rocm10-dual.sh
python3 qwen_bench.py rocm-audit --run-ops
```

First run the hidden-export arm without true outer serialization:

```bash
sudo systemctl stop qwen-flash-next
tmux kill-session -t qwen-mtp-export 2>/dev/null || true
install -d -m0700 /tmp/qwen-mtp-diag-slots
: > /tmp/qwen-mtp-export-server.log

tmux new-session -d -s qwen-mtp-export \
  "bash -lc 'cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench; set -a; source /etc/qwen-flash-next.env; set +a; export LLAMA_MTP_MODE=checkpoint-diagnostic LLAMA_CONTEXT_SIZE=8192 LLAMA_TRACE=1 LLAMA_LOG_VERBOSITY=4 LLAMA_SLOT_SAVE_PATH=/tmp/qwen-mtp-diag-slots LLAMA_MTP_DIAG_FORCE_TARGET_EXPORT=1 LLAMA_MTP_DIAG_TRACE_TARGET_LOGITS=1; exec ./deployment/run-production.sh >>/tmp/qwen-mtp-export-server.log 2>&1'"

for _ in {1..120}; do
  grep -q "listening on" /tmp/qwen-mtp-export-server.log && break
  tmux has-session -t qwen-mtp-export 2>/dev/null || { tail -n 100 /tmp/qwen-mtp-export-server.log; exit 1; }
  sleep 5
done
grep -q "listening on" /tmp/qwen-mtp-export-server.log || { tail -n 100 /tmp/qwen-mtp-export-server.log; exit 1; }

read -rsp "API key: " QWEN_TEST_API_KEY
echo
export QWEN_TEST_API_KEY
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label export-n01 \
  --server-log /tmp/qwen-mtp-export-server.log \
  --matrix-profile greedy-n01 \
  --repeats 2 \
  --max-tokens 40
unset QWEN_TEST_API_KEY

RUN=$(ls -dt results/*-mtp-diagnostic-export-n01 | head -1)
python3 qwen_bench.py archive "$RUN"
tmux kill-session -t qwen-mtp-export
for _ in {1..60}; do
  pgrep -f '/build-hip10-dual/bin/llama-server' >/dev/null || break
  sleep 1
done
pgrep -f '/build-hip10-dual/bin/llama-server' >/dev/null && {
  echo "export diagnostic server did not stop within 60 seconds" >&2
  exit 1
}
```

Then repeat with genuine independent target decode calls:

```bash
: > /tmp/qwen-mtp-outer-server.log
tmux kill-session -t qwen-mtp-outer 2>/dev/null || true
tmux new-session -d -s qwen-mtp-outer \
  "bash -lc 'cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench; set -a; source /etc/qwen-flash-next.env; set +a; export LLAMA_MTP_MODE=checkpoint-diagnostic LLAMA_CONTEXT_SIZE=8192 LLAMA_TRACE=1 LLAMA_LOG_VERBOSITY=4 LLAMA_SLOT_SAVE_PATH=/tmp/qwen-mtp-diag-slots LLAMA_MTP_DIAG_FORCE_TARGET_EXPORT=1 LLAMA_MTP_DIAG_OUTER_SERIAL=1 LLAMA_MTP_DIAG_TRACE_TARGET_LOGITS=1; exec ./deployment/run-production.sh >>/tmp/qwen-mtp-outer-server.log 2>&1'"

for _ in {1..120}; do
  grep -q "listening on" /tmp/qwen-mtp-outer-server.log && break
  tmux has-session -t qwen-mtp-outer 2>/dev/null || { tail -n 100 /tmp/qwen-mtp-outer-server.log; exit 1; }
  sleep 5
done
grep -q "listening on" /tmp/qwen-mtp-outer-server.log || { tail -n 100 /tmp/qwen-mtp-outer-server.log; exit 1; }

read -rsp "API key: " QWEN_TEST_API_KEY
echo
export QWEN_TEST_API_KEY
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label outer-n01 \
  --server-log /tmp/qwen-mtp-outer-server.log \
  --matrix-profile greedy-n01 \
  --repeats 2 \
  --max-tokens 40
unset QWEN_TEST_API_KEY

RUN=$(ls -dt results/*-mtp-diagnostic-outer-n01 | head -1)
python3 qwen_bench.py archive "$RUN"
tmux kill-session -t qwen-mtp-outer
for _ in {1..60}; do
  pgrep -f '/build-hip10-dual/bin/llama-server' >/dev/null || break
  sleep 1
done
pgrep -f '/build-hip10-dual/bin/llama-server' >/dev/null && {
  echo "outer-serial diagnostic server did not stop within 60 seconds" >&2
  exit 1
}
sudo systemctl start qwen-flash-next
```

The 40-token cap includes the known zero-based `n=1` divergence at token 37 and
stays below the baseline run's conservative lower bound of 46 outputs before
the first rejection. The classifier verifies rejection timing independently in
each new arm because the diagnostic itself could change acceptance timing; do
not exclude rollback unless that arm reports either a pre-rejection divergence
or no rejected draft in the captured request. If forced target export makes
`n=0` and `n=1` match, the original token-0 drift comes from the target export
graph. If the export arm still differs but the outer-serial arm matches, token
37 is caused by shared logical-batch target memory/PLE preparation. If both arms
still differ before any rejection, the next instrumentation target is accepted-
path recurrent/PLE state or MTP hidden-state handoff—not rollback.

Do not combine the bounded strict and checkpoint flags; the launcher
deliberately selects exactly one.

For final qualification, first run the same three-repeat matrix against a
target-only server and retain its result directory. Then run the strict
candidate with that target directory as the no-sidecar reference:

```bash
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label target \
  --server-log /tmp/qwen-mtp-target-server.log \
  --repeats 3

# Restart with LLAMA_MTP_MODE=strict before running the candidate.
python3 qwen_mtp_diag.py run \
  --url http://127.0.0.1:8080 \
  --server-label strict \
  --server-log /tmp/qwen-mtp-strict-server.log \
  --reference-run results/REPLACE-target-run \
  --repeats 3 \
  --require-pass
```

Use `--repeats 3 --require-pass` only for the final candidate. The pass gate
refuses to run without `--reference-run`, and it requires every target-only
n=0/non-stream temperature-and-repeat cell to have the identical request-body
SHA-256, be successful, and retain a complete nonempty exact token trace. At
temperature zero, n=1/2/3 must match n=0 token-for-token while reporting
nonzero drafted tokens; the report gives the exact first mismatching token ID,
byte sequence, logprob, and preceding prefix. A
loaded-sidecar n=0 mismatch against the target-only reference implicates the
request bypass or hidden-state export.
An n=1 greedy mismatch rules out a fault that requires two or more draft
tokens, but its target verifier still evaluates a two-row batch. It therefore
implicates one-token rollback, recurrent/PLE state, hidden-state handoff, or
batched target verification. An n=1 match with n=2/3 failures narrows the fault
to the deeper draft/rollback paths or the 256-cell attention boundary. In
addition, `--require-pass` requires the
n=3 arms collectively to contain 0/3, 1/3, and 2/3 partial-acceptance events in
their `LLAMA_TRACE` slices. Those events exercise rollback distances three, two,
and one respectively; merely drafting tokens or observing full 3/3 acceptance
does not qualify the rollback repair. The summary reports counts and case IDs
for each required event and names any missing path. Streaming must reconstruct
to the same reasoning, answer, tool calls, and finish reason as its paired
non-stream arm.

llama-server rejects `logprobs` together with tools and streaming. Consequently,
the exact token trace comes from each paired non-stream request, while the raw
SSE arm independently tests transport and chat parsing. Temperature-1
cross-window token differences are recorded but are not an exact-equivalence
failure because speculative sampling may consume RNG differently; fixed-seed
repeatability and stream/non-stream parity remain hard gates.

Rebuild and audit the server normally. The target-only launcher does not require
the experimental MTP marker; strict mode will refuse to start without it:

```bash
./build-rocm10-dual.sh
python3 qwen_bench.py rocm-audit --run-ops
```

Then install the launcher, configuration, and hardened systemd unit:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
sudo install -Dm0755 deployment/run-production.sh \
  /usr/local/libexec/qwen-flash-next
sudo install -Dm0644 deployment/qwen-flash-next.service \
  /etc/systemd/system/qwen-flash-next.service
sudo install -Dm0640 -o root -g jdillman \
  deployment/qwen-flash-next.env.example /etc/qwen-flash-next.env
sudoedit /etc/qwen-flash-next.env
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-flash-next.service
systemctl status qwen-flash-next.service --no-pager
journalctl -u qwen-flash-next.service -n 100 --no-pager
```

For the service, either set `LLAMA_API_KEY` in the root-managed environment file
or put one key per line in a separate mode-0640 file and set
`LLAMA_ARG_API_KEY_FILE` to its path. The key-file form is preferred. Leave the
unused setting empty. The unit validates the complete configuration with
`--check` before every start and does not restart-loop on configuration or
missing-file exit codes.

To prepare the preferred key file without placing the key in a command line:

```bash
sudo install -m0640 -o root -g jdillman /dev/null \
  /etc/qwen-flash-next.keys
sudoedit /etc/qwen-flash-next.keys
# Then set LLAMA_ARG_API_KEY_FILE=/etc/qwen-flash-next.keys in the environment file.
```

Keep `LLAMA_HOST=127.0.0.1` behind an authenticated TLS reverse proxy or a trusted
Tailscale path when possible. An API key authenticates requests but does not
encrypt traffic. If binding directly to another interface, set both
`LLAMA_HOST` and authentication before starting the unit. Remote clients cannot
connect while the log says the server is listening on `127.0.0.1`; use the
host's LAN/Tailscale address or `0.0.0.0` when direct remote access is intended.

Vision clients send normal OpenAI-compatible multimodal requests. With the safe
production default, both text and vision are target-only, so callers do not send
`speculative.n_max` or need to know that an experimental MTP mode exists.
Metrics remain available at `/metrics`.

### Historical, unqualified two-user Vulkan work

The following experiments document the earlier concurrency investigation. They
are retained for reproducibility, but they are not the recommended deployment
and the old Vulkan topology must not be substituted into the service above.

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

Once Q8 KV has eliminated storage reads, sweep the remaining greedy-prefill time
slice without reintroducing the F16 memory bottleneck:

```bash
./run-bench.sh --tier production-concurrency-q8-timeslice --fail-fast
```

This compares matched batch/ubatch sizes 2048, 1024, 512, and 256. Smaller
batches return control to the shared slot scheduler more frequently, which may
protect a user's decode while the other user is prefilling, at the cost of some
peak prompt throughput. Choose by slow-lane decode and end-to-end latency first,
then conservative prefill; aggregate throughput alone can hide an unusable
interactive lane.

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

The historical install instructions were removed. Use the qualified ROCm
launcher and systemd procedure in the preceding section.

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

### ROCm 10 recovery: prove the runtime before loading the model

The old sub-1 tok/s HIP result is not a backend comparison. That build could parse
the ROCmFP4 GGUF types, but its compiled HIP runtime did not contain the custom
ROCmFP4 dispatch. The likely result was unsupported work falling back out of the
accelerated matrix path. Do not launch the 121 GB model with that binary again.

First package the old source and build evidence. This reads source, CMake metadata,
binary code-object markers, linked libraries, and device information; it does not
load a model:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
git pull --ff-only
python3 qwen_rocm.py self-test
python3 qwen_rocm.py collect \
  --llama-dir /srv/llm/src/llama-qwen4exp \
  --build-dir /srv/llm/src/llama-qwen4exp/build-hip10 \
  --rocm /opt/rocm-10.0.0
```

The collector creates one `preflight/rocm-forensics-*.tar.gz` plus a SHA-256 file.
Its Git-remote output is credential-redacted, and it does not include model files.

The GGUF publisher identifies `kingjones30/ROCmFPX` as the required runtime. Use a
separate checkout so the working Vulkan/Qwen tree and its build remain unchanged:

```bash
git clone https://github.com/kingjones30/ROCmFPX.git \
  /srv/llm/src/ROCmFPX-qwen4exp
git -C /srv/llm/src/ROCmFPX-qwen4exp checkout \
  36e9acd40e10a87cd3c3ef8ec734668757dc8520
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
./build-rocm10-dual.sh
```

Revision `36e9acd40e10a87cd3c3ef8ec734668757dc8520` is pinned so a moving fork cannot
silently change the experiment. The build script refuses another revision, the
wrong source family, an incompatible existing CMake cache, or a library that does
not contain both required code objects. It also applies
`patches/rocmfpx-qwen4exp-mtp.patch`, its hidden-state scheduling follow-up,
`patches/rocmfpx-mtp-vision-resync.patch`,
`patches/rocmfpx-qwen4exp-vision-strict.patch`,
`patches/rocmfpx-qwen4exp-vision-strict-checkpoint.patch`, and
`patches/rocmfpx-host-checkpoints.patch`, plus the opt-in
`patches/rocmfpx-mtp-target-isolation.patch` diagnostics idempotently. This also upgrades a source
tree that already has either earlier MTP patch. Version 1.36.1 also detects and
removes the malformed zero-context host-checkpoint patch shipped in commit
`dc18127` before applying its contextual replacement. The first two patches add the missing
Qwen4Exp MTP sidecar loader/graph integration. The vision-resync patch prevents a
target-conditioned draft from decoding tokenless projector embeddings directly;
the strict pair serializes greedy target verification and replaces incompatible
bounded rollback with full-state checkpoints only when the explicit Qwen4Exp
vision flag is present. These remain experimental until the focused vision A/B
passes. The final patch adds an opt-in
`LLAMA_CKPT_FORCE_HOST=1` path that clears the device-storage flag only for prompt
checkpoint save/restore, retaining those checkpoints in host/unified RAM instead
of discrete-GPU VRAM. None replaces the ROCmFP4 kernels or changes Vulkan. The
script rejects source drift
if an exact patch can neither be applied nor recognized as already applied. It configures an isolated HIP-only build
against `/opt/rocm-10.0.0` with:

- Qwen4Exp plus ROCmFP4/ROCmFP4_FAST runtime dispatch;
- forced MMQ and no experimental rocWMMA flash-attention path;
- `gfx1100;gfx1151` in both `CMAKE_HIP_ARCHITECTURES` and `GPU_TARGETS`;
- `test-backend-ops` and compile-command metadata enabled;
- a compiled Qwen4Exp MTP marker in `libllama.so`;
- a compiled MTP multimodal-resync marker in `llama-server`;
- a compiled Qwen4Exp vision single-row verification marker in `llama-server`;
- a compiled Qwen4Exp checkpoint-backed strict-verification marker in `llama-server`;
- a compiled host-checkpoint marker in `libllama-common.so`.
- compiled target-export, true outer-serial verifier, and raw-logit diagnostic
  markers in `llama-server` and `libllama.so`.

Collect the new build evidence, then run the numerical gate:

```bash
python3 qwen_rocm.py collect \
  --llama-dir /srv/llm/src/ROCmFPX-qwen4exp \
  --build-dir /srv/llm/src/ROCmFPX-qwen4exp/build-hip10-dual \
  --rocm /opt/rocm-10.0.0

python3 qwen_bench.py rocm-audit --run-ops
```

Both commands create one `.tar.gz` plus a SHA-256 file automatically. The audit
archive is written even when a gate fails, so failed numerical output is preserved.
Any rebuild changes the server/HIP/libllama/libllama-common fingerprints, so the audit must be rerun
before model benchmarks.

The audit has two independent functional gates on both GPUs:

1. ordinary Q8_0 `MUL_MAT` and `MUL_MAT_ID`, proving ROCm 10 and the generic HIP
   backend before any custom quant is involved;
2. ROCmFP4 and ROCmFP4_FAST `MUL_MAT`/`MUL_MAT_ID`, proving the actual tensor
   formats used by the model.

It rejects a zero-test match, any numerical failure, missing `gfx1100`/`gfx1151`
coverage, or a stale server/HIP/libllama fingerprint. The MTP tier has one additional
gate: both the Qwen4Exp MTP source markers and the compiled `libllama.so` marker must
be present. A production experiment that sets `LLAMA_CKPT_FORCE_HOST` has another
gate: the source must clear `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, and the matching
`libllama-common.so` must contain the compiled marker. Every ROCm model tier is
gated on the saved proof, so a full-model run
cannot begin after an unverified rebuild.

Once the audit passes, use the clean APU-only diagnostic:

```bash
python3 qwen_bench.py preflight --tier rocm-smoke
./run-bench.sh --tier rocm-smoke --fail-fast
```

This path hides the 7900 XT with `ROCR_VISIBLE_DEVICES=1` (therefore the physical
8060S becomes logical `ROCm0`), uses the original joined GGUF with its native mmap
placement, and uses neither MTP nor heterogeneous placement. It first runs with
flash attention disabled, then repeats with it enabled. Both experiments test only
shallow 0/512-token depths. Before recording either depth, the harness processes the
same 512-token prompt once and erases its KV slot. This warms the relevant joined-PLE
file pages and kernels without reusing prompt KV. A depth-0-only warm-up is invalid
here: the first larger prompt would otherwise include cold random reads from the
50.66 GiB mmap table. If only the enabled experiment collapses after this warm-up,
the remaining defect is isolated to the attention path rather than storage or the
ROCmFP4 matrix kernels.

The pinned ROCmFPX revision uses the older `--mmap`/`--no-mmap` interface. The
harness explicitly translates the Vulkan fork's `--load-mode mmap` into ROCmFPX's
`--mmap`; it does not rely on an implicit backend default. An unknown load mode is
a hard configuration error rather than being silently removed.

The APU-only tier above is fault isolation, not a Vulkan/ROCm performance A/B.
After it passes, run the semantically matched control:

```bash
./run-bench.sh --tier backend-smoke-matched --fail-fast
```

That tier holds the quantized weights, F16 KV, batch 2048, ubatch 512, context,
prompts, decode length, and flash-attention state constant. Each backend uses the
PLE representation it actually supports: Vulkan uses the PLE16 GGUF with the split
tables CPU-mapped, while ROCm uses the original joined GGUF with native mmap
placement. PLE16 exists only to work around Vulkan's per-buffer allocation limit;
it is not a separate quality variant and must not be forced into the ROCm path.

If shallow performance is sane, localize the previously reported context cliff:

```bash
./run-bench.sh --tier rocm-depth --fail-fast
```

That single-variable tier stays APU-only, pre-warms the full 32768-token prompt,
erases its KV slot, then sweeps 0, 512, 1024, 1536, 2048, 4096, 8192, and 32768
requested tokens. Once this curve remains sane, run the focused multi-GPU placement
test:

```bash
python3 qwen_bench.py preflight --tier rocm-placement
./run-bench.sh --tier rocm-placement --fail-fast
```

This is not the old ratio sweep. It tests only four F16-KV, batch-2048 deployments
on the code workload at approximately 4K and 32K depth after an excluded 32K
warm-up: APU-only; the established 82/18 contiguous layer split; routed experts on
the iGPU with remaining eligible tensors on the 7900 XT; and routed plus shared
experts on the iGPU with the remainder on the 7900 XT. In the unrestricted ROCm
device list, `ROCm0` is the RX 7900 XT and `ROCm1` is the Radeon 8060S. The joined
PLE tensor remains on its native file-backed mmap path in every cell.

The additional layer and row definitions remain available for deliberate follow-up,
but are excluded from the recommended campaign. The first ROCm MTP attempt failed
before serving because the pinned runtime treated the 4.1 GB sidecar as a complete
49-layer target and demanded `blk.0.hc_attn_norm.weight`. Identical failure on both
draft devices rules out capacity and placement. Rebuild the same pinned ROCmFPX tree
with the harness patch, refresh the numerical audit, and only then rerun the tier:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
git pull --ff-only
python3 qwen_bench.py self-test
./build-rocm10-dual.sh
python3 qwen_bench.py rocm-audit --run-ops
python3 qwen_bench.py preflight --tier rocm-mtp
./run-bench.sh --tier rocm-mtp
```

The patch keeps the 48-layer trunk optional when loading an MTP-only GGUF, loads the
single appended draft block, passes the target's four-stream hidden state to it,
uses an attention-only KV cache for that block, and enables recurrent-state rollback
during speculative verification. The hidden-state handoff is registered as an
explicit graph output so the scheduler assigns it a backend before the speculative
driver copies it. Preflight now refuses an older unpatched binary,
so the previous two-minute startup failures cannot recur unnoticed.

The ROCm MTP tier fixes the target at the routed-expert placement: the joined PLE
stays file-backed, routed experts stay on the iGPU, and shared experts plus remaining
eligible target tensors stay on the 7900 XT. It compares no MTP against the Q8
sidecar at n=4/p-min 0.75 first on the iGPU and then on the 7900 XT, using the same
F16 KV, batch 2048, 4K/32K prompts, and excluded 32K warm-up. Do not add
`--fail-fast`: the dGPU cell intentionally runs last because its 20 GiB capacity may
be insufficient after the winning target placement, and an OOM is itself a useful
result that should not discard the completed control and iGPU cells.

The first successful run (`20260829-002156-rocm-mtp`) establishes the 7900 XT as
the MTP device. Relative to no MTP, it raises decode from 31.18 to 42.44 tok/s at
4K depth (+36%) and from 29.70 to 50.98 tok/s at 32K depth (+72%). It is also 17%
to 20% faster at decode than iGPU MTP, with effectively identical acceptance.
MTP is not free during prompt processing: the target exposes its 10,240-wide hidden
state and the draft block processes the prompt to build its own attention KV. The
7900 XT result therefore lowers prefill from 590.29 to 518.15 tok/s at 4K (-12%)
and from 484.80 to 427.75 tok/s at 32K (-12%). For the tier's 256-token response,
estimated prefill-plus-decode time is 17.37 versus 16.47 seconds at 4K (MTP wins)
and 89.44 versus 96.62 seconds at 32K (no MTP wins). The approximate MTP break-even
is 150 generated tokens at 4K and 767 at 32K; prompt/KV reuse lowers the practical
break-even substantially on subsequent turns.

Tune only the successful dGPU MTP topology. The draft-window tier deliberately uses
only the 4K prompt so n=2 and n=3 do not repeat an identical expensive 32K prefill:

```bash
python3 qwen_bench.py preflight --tier rocm-mtp-window
./run-bench.sh --tier rocm-mtp-window
```

Run `20260829-113303-rocm-mtp-window` selected n=3: 51.89 tok/s with 90.4%
acceptance, versus 49.06 tok/s at n=2 and 46.91 tok/s at n=4. Prefill remained
within 0.6% across the three cells. The next tier therefore uses n=3 exclusively.

Test the knobs that can affect prompt processing at both 4K and 32K: ubatch
512/1024/2048 and Q8 draft KV. Target KV stays F16 in every cell, so the Q8 result
isolates only the sidecar cache:

```bash
python3 qwen_bench.py preflight --tier rocm-mtp-prefill
./run-bench.sh --tier rocm-mtp-prefill
```

These are separate tiers because draft-window length changes decode work but not the
sidecar's prompt prefill. Combine the winning window, ubatch, and draft-KV settings
only after both focused sweeps complete.

Run `20260829-114537-rocm-mtp-prefill` selected ubatch 2048 for end-to-end
throughput. Relative to ubatch 512, it increased prefill by 25.7% at 4K and 24.7%
at 32K. Its 256-token estimated prefill-plus-decode time improved from 15.55 to
14.08 seconds at 4K and from 97.40 to 78.99 seconds at 32K. Q8 draft KV was
orthogonally useful at ubatch 512: prefill was unchanged while decode improved by
6.6% at 4K and 2.7% at 32K. The combination still needs to be measured; ubatch
1024 is excluded because its 32K decode and acceptance collapsed.

The finalist tier combines ubatch 2048 with Q8 draft KV, adds matched no-MTP
controls, and probes batch/ubatch 4096 because prefill had not plateaued at 2048.
It uses two rounds and 512 generated tokens. Do not use `--fail-fast`: a failed
4096 probe must not discard the validated 2048 results.

```bash
python3 qwen_bench.py preflight --tier rocm-mtp-finalists
./run-bench.sh --tier rocm-mtp-finalists
```

Run `20260829-131616-rocm-mtp-finalists` makes the production decision. At
batch/ubatch 2048, F16 draft KV reached 51.23 tok/s decode and 652.08 tok/s
prefill at 4K, and 50.11/535.16 tok/s at 32K, with about 88% MTP acceptance.
Q8 draft KV saved only 0.07 GiB of 7900 XT VRAM at 4K and 0.36 GiB at 32K while
slightly reducing throughput and acceptance, so it is excluded from every
production ROCm tier.

The no-MTP batch/ubatch-4096 control improved prefill by roughly 2-3%, but the
MTP-4096 probe exhausted 7900 XT memory while allocating a 4,080 MiB compute
buffer even with the smaller Q8 draft cache. F16 draft KV needs at least as much
memory, so repeating the same topology with F16 would not be a meaningful test.
The production MTP configuration is therefore n=3, F16 target and draft KV, and
batch/ubatch 2048. A future 4096 MTP experiment must first move more target
tensors off the 7900 XT or reduce another allocation.

Validate joined-PLE SSD backing separately from the MTP decision:

```bash
python3 qwen_bench.py preflight --tier rocm-ple-ssd
./run-bench.sh --tier rocm-ple-ssd
```

This compares automatic placement with a candidate that keeps the routed experts
on `ROCm1` and additionally applies
`^per_layer_token_embd\.weight$=CPU` under `--mmap`. The 51.2B-parameter PLE is
then file-backed through the Linux page cache: hot pages can occupy RAM, but they
are reclaimable and can be faulted from SSD instead of consuming a permanent GTT
allocation. This is SSD backing, not direct I/O or a promise that every access
hits the SSD.

Both candidates receive the same excluded 32K warm-up, followed by slot erasure,
before measured 4K and 32K requests. Compare prefill/decode, file versus anonymous
RSS, GTT/VRAM, `MemAvailable`, physical read bytes, and major faults. The explicit
CPU candidate is successful only if it materially reduces device/GTT residency
without an unacceptable hot-throughput regression.

That older tier is a placement test, not a strict SSD-versus-RAM test. The precise
production storage A/B keeps the joined PLE explicitly on CPU in both arms and
changes only the loader mode. The SSD arm uses `--mmap`, so PLE pages occupy the
reclaimable Linux page cache and may be faulted from the model file. The RAM arm
uses `--no-mmap`, so the same CPU tensor becomes resident in non-file-backed
unified RAM (reported primarily as `RssShmem` on this ROCm system). MTP,
F16 target/draft KV, 88/12 target split, ubatch 1536, host checkpoints, and every
tensor override remain identical.

Start with the 64K screen; do not begin with two near-full 256K prompts:

```bash
python3 qwen_bench.py preflight --tier rocm-ple-storage-screen
./run-bench.sh --tier rocm-ple-storage-screen
```

The `20260829-195434-rocm-ple-storage-screen` run made the production tradeoff
clear. After an excluded 32K warm-up, mmap and no-mmap produced identical output
hashes and 70.3% MTP acceptance. Mmap delivered 470.69 prefill and 33.07 decode
tok/s; no-mmap delivered 469.03 and 33.43 tok/s. The remaining hot mmap request
read only 0.14 GiB from storage, but retained 55.54 GiB minimum host availability.
No-mmap retained only 3.43 GiB and placed 54.27 GiB in `RssShmem`. In other words,
resident PLE provides no material hot-path speed benefit while consuming roughly
52.1 GiB of the memory margin needed by native 256K context.

If the RAM arm loads and serves coherently, prove its native-256K startup allocation:

```bash
python3 qwen_bench.py preflight --tier rocm-ple-ram-capacity
./run-bench.sh --tier rocm-ple-ram-capacity --fail-fast
```

Only if both gates pass, run the near-full A/B:

```bash
./run-bench.sh --tier rocm-ple-storage-full
```

To avoid repeating the approximately 20-minute mmap baseline after a matching
host-checkpoint `rocm-256k-fit-full` result already exists, select only
`prod_hip_256k_tail_88_12_ub1536_ple_ram` in the last command and compare its
archive with that baseline. This capacity is deliberately not assumed: the first
successful no-checkpoint 253952-token mmap run retained 52.46 GiB minimum host
availability, only modestly more than the 50.66 GiB PLE, before accounting for
host-resident checkpoints and other non-file-backed memory.

### Native 256K one-slot placement campaign

First inventory the exact storage span of every GGUF tensor and aggregate those
bytes into placement-relevant families. The parser is dependency-free and includes
alignment padding, so all family spans add exactly to each GGUF data section:

```bash
python3 qwen_inventory.py self-test
python3 qwen_inventory.py scan \
  /srv/llm/models/qwen-flash-next/Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf \
  /srv/llm/models/qwen-flash-next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --include-tensors \
  --output preflight/qwen-gguf-inventory.json \
  --archive
```

Upload `preflight/qwen-gguf-inventory.tar.gz` for placement analysis; the matching
`.sha256` sidecar is optional.

The measured inventory for the current files is:

| Target family | Exact data span |
| --- | ---: |
| PLE | 50.66 GiB |
| routed experts | 59.77 GiB |
| attention projections | 1.07 GiB |
| token embedding | 0.49 GiB |
| LM head | 0.49 GiB |
| linear-attention state weights | 0.29 GiB |
| routers | 0.23 GiB |
| shared experts | 0.12 GiB |

The Q8_0 MTP sidecar has a 3.84 GiB data section, dominated by 2.49 GiB of
routed-expert weights plus two 0.63 GiB embedding/output matrices.

The first 64K screen held Q8_0 MTP on `ROCm0`, draft depth 3, F16 target and
draft KV, batch/ubatch 2048, routed experts on `ROCm1`, CPU-mapped PLE, and
`--cache-ram 0`:

```bash
python3 qwen_bench.py preflight --tier rocm-256k-placement-screen
./run-bench.sh --tier rocm-256k-placement-screen
```

Run `20260829-143406-rocm-256k-placement-screen` disqualifies the cumulative
tensor-family strategy. Token embedding migration reduced decode by about 11.5%
without reducing peak 7900 XT residency. Shared experts saved only about 0.1 GiB,
reduced 32K decode by 26% and prefill by 14%, and caused the 32K response to follow
text embedded in the reference corpus instead of the requested coding task. The
LM-head and full-attention candidates inherited that correctness failure. Linear
extrapolation of measured request peaks estimates 20.94 GiB at 253952 tokens for
the base, 19.43 GiB for the shared-plus-output candidate, and 18.90 GiB for the
largest migration. These are planning estimates, not capacity proofs.

The follow-up therefore leaves token embedding, shared experts, and LM head alone.
It tests dGPU-first contiguous tail-layer splits. This is the important distinction
from the old APU-first 82/18 test: `--device ROCm0,ROCm1` keeps most target layers
on the 7900 XT, while a small tail moves to the iGPU. Routed experts remain
explicitly on the iGPU, PLE remains CPU-mapped, and MTP remains explicitly on
`ROCm0`. A successful tail split can free both static target tensors and target KV
for the moved layers with one contiguous boundary:

```bash
python3 qwen_bench.py preflight --tier rocm-256k-tail-screen
./run-bench.sh --tier rocm-256k-tail-screen
```

Do not jump directly from a successful 64K run to the near-full prompt. Select the
least expensive tail candidate that leaves credible 7900 XT headroom, and prove
that a single 262144-token slot starts first:

```bash
python3 qwen_bench.py preflight --tier rocm-256k-capacity \
  --experiments 'prod_hip_256k_tail_88_12,prod_hip_256k_tail_84_16'
./run-bench.sh --tier rocm-256k-capacity \
  --experiments 'prod_hip_256k_tail_88_12,prod_hip_256k_tail_84_16'
```

Runs `20260829-151825-rocm-256k-tail-screen` and
`20260829-153929-rocm-256k-capacity` establish that the coarse endpoints are not
production finalists. The 88/12 split retained base-class 32K throughput and task
following, but 256K initialization failed while ROCm0 attempted to reserve a
3565232128-byte MTP compute buffer. The 84/16 split initialized in 67.1 seconds,
but consumed 19.79 GiB of the 19.98 GiB 7900 XT allocation, reduced 32K decode
from 50.65 to 30.14 tok/s, dropped MTP acceptance to 68.9%, and answered an
unrelated question from the reference corpus. Allocation success does not override
that quality failure.

The next screen searches both sides of that tradeoff: 87/13, 86/14, and 85/15
retain ubatch 2048, while three 88/12 candidates reduce only ubatch to 1792, 1536,
or 1024. The latter preserve the known-good layer boundary and test whether a
smaller prompt/MTP graph reserve is cheaper than moving another target layer:

```bash
python3 qwen_bench.py preflight --tier rocm-256k-fit-capacity
./run-bench.sh --tier rocm-256k-fit-capacity
```

Do not use `--fail-fast`: expected OOMs must not suppress later candidates.
Run `20260829-160347-rocm-256k-fit-capacity` found that all six candidates
initialized, but allocation alone is not the acceptance criterion. Exact peak
7900 XT headroom was only 144 MiB for 87/13, 146 MiB for 86/14, 202 MiB for
85/15, and 262 MiB for 88/12 at ubatch 1792. Those are boundary diagnostics, not
production margins. The same 88/12 placement retained 1118 MiB at ubatch 1536
and 2837 MiB at ubatch 1024.

The initial quality tier kept all three 88/12 ubatch variants so 1792 could
record the speed ceiling while excluding the three low-headroom layer splits:

```bash
python3 qwen_bench.py preflight --tier rocm-256k-fit-quality
./run-bench.sh --tier rocm-256k-fit-quality
```

Run `20260829-163847-rocm-256k-fit-quality` invalidated a simple throughput or
MTP-acceptance ranking. At 32K, ubatch 1792 still answered the requested
`merge_intervals` task and accepted 94.8% of draft tokens. Ubatch 1536 answered
an unrelated Chinese llama-server question at 83.4%, while ubatch 1024 emitted
incoherent unrelated code at 69.7%. The result is deterministic task/state
divergence, not an ordinary acceptance-rate fluctuation. SSD traffic, page
faults, and temperature did not correlate with correctness.

Do not run the final 256K tier yet. First remove MTP from the equation and
fingerprint the target model's next-token distribution across the same ubatches:

```bash
python3 qwen_bench.py preflight --tier rocm-ubatch-target-fingerprint
./run-bench.sh --tier rocm-ubatch-target-fingerprint --fail-fast
```

This runs two fresh-server rounds at 32K for ubatch 2048, 1792, 1536, 1024, and 512.
It generates only four tokens, requests the top 20 probabilities, stores a
compact first-token view plus a distribution hash in `results.jsonl`, and retains
the complete response. Decode speed and answer quality are intentionally not
measured here: `n_probs` adds overhead, and four tokens cannot reach a function
signature after a reasoning preamble. If the no-MTP probability fingerprints
diverge, the target's chunked Gated DeltaNet prefill/state path is implicated.

The interrupted v1.33.1 run `20260829-171001-rocm-ubatch-target-fingerprint`
already captured one complete round with 32 probability-bearing tokens. Ubatch
2048, 1536, and 1024 produced byte-identical text; ubatch 1792 differed only by
leading whitespace. Nevertheless, the first-token probability distributions
changed materially: the selected newline had log probability -0.349 at ubatch
2048, -0.262 at 1536, and -0.692 at 1024. At ubatch 1792, three spaces narrowly
won a near tie at -1.049 versus -1.052 for an empty control token and -1.226 for
newline. This proves ubatch-dependent target-logit drift without yet proving
semantic corruption. Exact probability hashes are deliberately strict and must
be interpreted together with token rank, margin, and the full-output control.

Quantify functional quality with the deterministic target-only screen, not an
output hash and not MTP acceptance:

```bash
python3 qwen_bench.py preflight --tier rocm-ubatch-quality-target-screen
./run-bench.sh --tier rocm-ubatch-quality-target-screen --fail-fast
```

This uses exact 32768-token prompts and eight machine-verifiable tasks: passkeys
at early/middle/late positions, ordered retrieval, latest-revision selection,
grounded arithmetic, exact JSON retrieval, and distractor rejection. Each runs
twice from fresh servers at ubatch 2048, 1792, 1536, 1024, and 512. Requests are
greedy but respect EOS; forcing `ignore_eos=true` would turn a correct short
answer into an artificial exact-match failure. The 256-token ceiling prevents a
truncated reasoning preamble from being confused with task corruption.

`summary.md` and `quality.csv` report exact pass count, pass-rate delta from the
ubatch-2048 reference, 95% Wilson interval, paired regressions/improvements,
two-sided exact McNemar p-value, and median prefill/decode throughput over every
scored response. `quality-cases.csv` identifies which retrieval position or task
failed. With 16 trials per experiment, 16/16 has a Wilson interval of roughly
80.6%–100%; this is a deterministic correctness screen, not a claim about broad
language-model benchmark quality. An eligible target ubatch needs 16/16 and zero
paired regressions. A repeat of the finalist is appropriate before production if
all candidates pass.

The ubatch-2048 arm is the numerical/performance reference at 64K, not an implied
256K deployment candidate. Capacity headroom is evaluated separately.

Only after the target-only screen identifies safe target ubatches, run MTP on
that subset. Keep n=3 on the 7900 XT and F16 target/draft KV fixed:


```bash
python3 qwen_bench.py preflight --tier rocm-ubatch-quality-mtp-screen
./run-bench.sh --tier rocm-ubatch-quality-mtp-screen \
  --experiments 'COMMA_SEPARATED_TARGET_QUALIFIED_MTP_EXPERIMENTS' \
  --fail-fast
```

Apply the same 16/16 and zero-regression gate, then use decode throughput and MTP
acceptance only to rank candidates that passed. Acceptance is conditional on the
target stream and cannot itself measure quality. Historical run
`20260829-131616-rocm-mtp-finalists` produced identical hashes and acceptance in
both ubatch-2048 rounds, which argues for deterministic configuration-dependent
state divergence rather than random ROCm instability.

The older `rocm-ubatch-target-correctness` and
`rocm-ubatch-mtp-repeatability` tiers remain available for reproducing the single
`merge_intervals` symptom, but their anchor score is too narrow to quantify
quality and they are no longer the production qualification path.

The symptom closely parallels open llama.cpp correctness reports for hybrid
Gated DeltaNet models: [issue #27237](https://github.com/ggml-org/llama.cpp/issues/27237)
reports batch/ubatch-dependent garbage, including with MTP disabled, and
[issue #27556](https://github.com/ggml-org/llama.cpp/issues/27556) reports
deterministic HIP context loss on gfx1151 while individual backend-op tests pass.

The 1792 arm still is not eligible for production because its 262 MiB startup
margin is too small. The broader deterministic screen subsequently showed that
the earlier single-task failures were not a useful general quality gate. Proceed
with ubatch 1536 as the performance candidate; its measured startup margin is
1118 MiB. Run only the decisive exact 253952-token prompt, leaving 8192 tokens
of the native 262144 window for generation and server overhead:

```bash
./run-bench.sh --tier rocm-256k-fit-full \
  --experiments 'prod_hip_256k_tail_88_12_ub1536' \
  --fail-fast
```

If—and only if—that populated-KV request fails or loses the device, repeat with
`prod_hip_256k_tail_88_12_ub1024`, which retains 2837 MiB measured startup
headroom at a modest prefill cost. Do not spend time running both when 1536
passes.

The startup-only tier proves allocation, not operational stability. Only the
near-full request proves populated-KV operation. A candidate also needs practical
headroom; merely reaching the health endpoint at nearly 100% VRAM is not a
production pass.

The first ubatch-1536 near-full attempt exposed a separate hidden VRAM consumer,
not a microbatch failure. Although `--cache-ram 0` disabled the inter-request
prompt cache, llama-server still created per-slot context checkpoints every 8192
tokens using its independent default of 32. After serialized checkpoints of
141.4, 157.6, and 173.7 MiB, the 32K warm-up aborted when checkpoint handling
requested another 114.5 MiB allocation on ROCm0.

Those checkpoints are useful and are not disabled in the production experiments.
The patched ROCmFPX build runs them with `LLAMA_CKPT_FORCE_HOST=1`, which preserves
the checkpoint byte vectors in unified host RAM and prevents their auxiliary
storage from consuming 7900 XT VRAM. This is the pinned-fork adaptation of the
host-checkpoint workaround validated in
[llama.cpp issue #23719](https://github.com/ggml-org/llama.cpp/issues/23719).
Production bounds the subsystem to eight
checkpoints with `--ctx-checkpoints 8 --checkpoint-every-n-tokens 32768`. A near-full
253952-token prompt crosses seven such
boundaries, so no useful checkpoint is evicted in the one-slot 256K proof. The
server log records both the environment setting and, at the first checkpoint,
`forcing checkpoint state to host memory`; absence of that line makes the run
invalid. `--ctx-checkpoints 0` is now reserved for capacity diagnosis only, not
the production topology. This subsystem remains independent of `--cache-ram`.

The in-memory prompt-response cache is disabled in every candidate with
`--cache-ram 0`. Disk-backed prompt caching remains valuable, but `--slot-save-path`
only enables explicit slot actions; it is not an automatic bounded SSD cache. This
ROCmFPX build now advertises `--cache-disk PATH`, so that specific implementation is
the future A/B subject after native 256K capacity is established. Its separate
acceptance criteria are tracked in
[`FUTURE_TESTS.md`](FUTURE_TESTS.md).

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
