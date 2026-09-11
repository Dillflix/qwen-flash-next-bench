# Isolated Strix cache + vision smoke

This follows the successful gfx1151-only 256K allocation check. It does not
change production configuration, build the runtime, or quantize weights.

The trial fixes target, Q8_0 MTP n=3, and BF16 vision projector placement to
`ROCm1` (gfx1151 on the measured host). Both KV caches are F16. Ordinary model
weights use `--load-mode none`; PLE remains SSD-backed with direct I/O and a
256 MiB row-cache cap. The context allocation is 262144, with one slot,
ubatch 1536, batch 2048, an 8192 MiB prompt backing-cache cap, and up to eight
checkpoints at a minimum 32768-token spacing. This fork uses
`--checkpoint-min-step`, not the older production fork's spelling.

The five measured requests are:

1. A: the saved, chat-templated 32768-token baseline, 32 generated tokens.
2. A again: live-prefix reuse and repeated greedy output comparison.
3. B: a short, unrelated, properly templated request.
4. A again: backing-cache selection, reused tokens beyond A/B's common prefix,
   and repeated greedy output comparison.
5. One deterministic shapes image with MTP enabled; basic color/shape/OCR checks.

There is no full 256K prompt. The backing cache is not filled to its cap.
The text cap is intentional: this is a state-reuse smoke, not an answer-quality
benchmark or a statistically meaningful performance comparison. The vision
keyword check does not prove general vision quality or MTP correctness.
If prefix reuse fails, more than one request may require a full 32K prefill.

## Run inside tmux

Start a new session once:

```bash
tmux new -s strix-service-smoke
```

Inside it, after the corresponding harness commit has been published:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench &&
git pull --ff-only &&
python3 -m unittest discover -s tests -p 'test_strix_service_smoke.py' &&
python3 qwen_strix_service_smoke.py --run
```

Omit `--run` to print the plan without touching services. Detach with Ctrl-B,
then D; reconnect with `tmux attach -t strix-service-smoke`.

The script requires the pinned trial checkout
`5f851647fe5ed795dfd6c0a3fba543114879e874`, the existing target/MTP/projector
files, and the exact baseline token file under
`trial-results/hip-templated-32k.xhe6lmo5/prompt-tokens.json`. It checks these
before stopping production. Do not run other GPU benchmarks simultaneously.

It stops production only if it was active, owns and terminates only its trial
child process, then starts production again only if it was active initially.
It reports that start command's outcome; this is not proof that the production
model has finished loading. If production was already inactive, it stays so.
Ctrl-C and ordinary request/startup errors go through cleanup and packaging;
power loss, SIGKILL, and unrecoverable filesystem errors cannot be handled.

Inherited `LLAMA_*`, `GGML_*`, and GPU visibility overrides are removed for the
trial child. The trial does not source production credentials or old-fork MTP
patch settings. It listens unauthenticated on loopback port 8189 only. No API
key is needed. This tests the pinned fork's own multimodal/MTP behavior.

## Results

The script prints a single `.tar.gz` and writes its `.sha256` alongside it
under the trial checkout's `trial-results`. It includes request/response JSON,
server logs, `/props`, command/revision, memory samples, and cache/vision verdicts.
An error or failed smoke produces a nonzero exit status, not a false pass.
Outputs use only synthetic fixture content; inspect the archive before sharing.

A backing-cache PASS requires both logged selection of a stored prompt and
actual token reuse beyond B's live common prefix. Merely setting `cache_prompt`
or getting a successful HTTP response does not pass. Changed greedy output is
flagged for investigation, not automatically attributed to one specific bug.
At least one text probe must report actual draft tokens. Vision separately
reports whether drafting occurred: successful vision with a native bypass is
not evidence that speculative image-conditioned decoding was exercised.
Passing this short test does not qualify a fully populated 8 GiB cache, long
conversations, multiple active slots, or near-full-context vision inference.

## Focused uncached repeat / cache / MTP comparison

The initial smoke demonstrated cache reuse and correct shapes/OCR, but the
first cold output differed from both identical cached outputs. To separate
uncached repeatability from cache-path drift, use:

```bash
python3 qwen_strix_service_smoke.py --repeat-ab --run
```

This runs **six requests**, all with the same saved 32768-token prompt and a
32-token generation cap:

1. Fresh server with MTP n=3: uncached, uncached, cached.
2. Fresh server without the draft model or speculative options: uncached,
   uncached, cached.

Each uncached request explicitly uses `cache_prompt: false`; reported
`cache_n=0` and `prompt_n=32768` must confirm full processing. The cached request
must demonstrate reuse. The test checks observed draft activity in both arms.
Target settings, SSD PLE, vision loading, 256K allocation, and cache settings
are unchanged. Removing MTP also removes its memory allocation and changes
verification batch shapes; this is a diagnostic contrast, not proof of a
specific fault. No image requests, diversion requests, or full-256K prompts
are repeated. Four full 32K prefills are unavoidable in this comparison;
using the recent timings, allow roughly six minutes including two model loads,
or longer if the cached requests miss or the host is busy.

Both arms run even if the first reports output drift. A crash, timeout, or
malformed response stops the run and is archived. Production is stopped and
conditionally restored once around the entire pair, with one combined archive
under `trial-results/hip-cache-repeat-ab.*.tar.gz`.

The per-arm verdict distinguishes:

- `PASS`: both uncached responses and the cached response match exactly.
- `INCONCLUSIVE_UNCACHED_DRIFT`: the uncached responses differ; do not attribute
  the difference specifically to caching.
- `CACHE_PATH_DRIFT`: the uncached pair matches but cached output differs.
- `INVALID`: cache or MTP activity did not satisfy the diagnostic controls.

Non-PASS results return a nonzero status. Token divergence positions and IDs,
full outputs, and per-request prefill/decode timings are retained. A
cross-arm comparison of the second uncached responses is recorded separately
in `results.json`; passing per-arm tests is not a general MTP equivalence or
quality certification. Cache-path drift alone still cannot distinguish floating
point/batch-shape effects from incomplete checkpoint restoration.

## Candidate repair: MTP checkpoint carry state

The pinned MTP driver keeps `pending_h` outside the target/draft KV contexts.
Its `get_state`/`set_state` methods were inherited no-ops, even though the server
calls these hooks when it saves and restores prompt checkpoints. The candidate
patch saves the deferred hidden row and restores it with the contexts. It clears
batch-local verification rows and indices from the abandoned continuation;
the next `process()` rebuilds them. Draft sampling already resets on each draft.
This patch does not change quantization, kernels, acceptance rules, or KV precision.

The payload has a version marker and embedding width. Missing, truncated, or
incompatible payloads fail explicitly rather than leaving stale state. Start
fresh processes for the test; do not import old saved slots into the patched
runtime. The serialized row is 10248 bytes at width 2560 (10240 bytes plus an
8-byte header), per checkpoint, not another multi-GiB model buffer.

This is a local candidate repair for the measured gfx1151 deployment, not an
upstream submission or a proven fix for token divergence. Bad draft state
alone should normally be rejected by correct target verification. The same
focused A/B diagnostic is still required after the patch.

After publication, inside tmux with no Strix trial server running:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench &&
git pull --ff-only &&
python3 -m unittest discover -s tests -p 'test_strix_service_smoke.py' &&
bash ./build-strix-mtp-checkpoint.sh &&
python3 qwen_strix_service_smoke.py --repeat-ab --require-mtp-state-patch --run
```

The build helper checks the exact source revision and refuses to rebuild while
a matching trial server is running. It applies only
`patches/strix-mtp-checkpoint-state.patch`, runs a CPU-only regression fixture
compiled with the existing cached C++ compiler, and incrementally rebuilds the
trial's existing `build-hip10-dual` server. It reuses the CMake cache rather than
changing HIP, compiler, architecture, or kernel options. Running it twice is
safe: an already-applied patch is detected. Unrelated source edits are not reset.
The old `/srv/llm/src/ROCmFPX-qwen4exp` production build is not modified.

The test fixture compiles the actual patched methods extracted from
`common/speculative.cpp`, and checks bitwise round-trip (including signed zero),
sequence isolation, stale-state clearing, and malformed payload rejection.
It does not instantiate real model contexts or prove GPU correctness. CI also
applies the patch to the pin and builds the CPU server.

`--require-mtp-state-patch` checks the compiled marker before stopping production,
records binary hashes, and requires an executed restore-hook log in the MTP-on
arm. A stale or unused patch cannot pass the diagnostic merely because output
happens to match. Results still go into one archive, and production restoration
retains the same conditional behavior described above.
