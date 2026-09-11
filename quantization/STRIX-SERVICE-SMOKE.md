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
