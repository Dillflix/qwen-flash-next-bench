# Strix Q5_K gate/up + IQ4_NL down

This is a screening candidate, not a qualified production model. It does not
change H1, the benchmark matrix, deployment settings, or the running service.

Recipe: all 96 routed gate/up tensors Q5_K; all 48 routed down tensors IQ4_NL;
joined PLE Q8_0; token embedding/output Q8_0; routers and indexer projections
preserve source precision (normally F32 and BF16 respectively). Other eligible tensors use the Q8_0 preset, with the quantizer's protected
norm/state tensors preserved. MTP and mmproj remain separate existing files.
The output inventory records every actual tensor type.

Estimated target weights: 77.8 GiB excluding PLE, plus 50.7 GiB PLE on disk.
Runtime allocation, 256K F16 caches, compute buffers, backing cache, MTP, and vision
are additional. Use --ngram-on-disk when benchmarking; do not load PLE onto the GPU.

## First: find the source

Do not requantize ROCmFP4 or another low-bit GGUF. The earlier H1 conversion was
deferred; its source file must not be assumed to exist. This helper accepts an
F16/BF16/F32 target GGUF (or its first shard), with joined PLE, without the MTP
block. GGUF file sharding is supported; splitting PLE into 16 tensors is not.
Model architecture metadata is required in the first shard only; later shards
may omit it, as the converter normally does. A conflicting architecture on any
later shard is rejected. If an older helper reports `Not a qwen4exp GGUF` for
shard 2 after successful conversion, update the helper and rerun the plan using
the existing first shard; do not repeat conversion for that preflight error.
Header precision checks cannot detect a file previously dequantized from low bit:
source provenance must also be known.

```bash
find /srv/llm/models -maxdepth 5 -type f \
  \( -iname '*Flash*BF16*.gguf' -o -iname '*Flash*F16*.gguf' \
     -o -name 'model.safetensors.index.json' \) -printf '%p\n'
df -h /srv/llm/models
```

If none exists, prepare an original-source conversion first. Do not use the old
H1 run-quant.sh: it targets the older PLE16 converter and ROCmFP4 recipe. No
conversion/download is automated here. The correct conversion route depends on
whether the available HF source is BF16 or FP8; this must be established first.

## Build and plan (no quantization yet)

Quantization runs on CPU. The Vulkan build provides a quantizer without the HIP
runtime-library dependency encountered during the trial. It still produces the
same standard GGUF formats for either backend.

```bash
cmake --build /srv/llm/src/strix-llama-trial-5f851647/build-vulkan \
  --target llama-quantize -j 4
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench
python3 -m unittest discover -s tests -p 'test_strix_quant.py'
read -erp 'Original-source F16/BF16 GGUF (first shard): ' SOURCE_GGUF
OUTPUT_DIR=/srv/llm/models/qwen-flash-next/strix-q5k-iq4nl-trial1
python3 qwen_strix_quant.py --source "$SOURCE_GGUF" --output-dir "$OUTPUT_DIR"
```

The inspected Strix revision is enforced. Planning reads headers and does not
write model weights. It checks the exact expert shapes and rejects quantized
input, missing shards, extra routed MTP tensors, and PLE16 layouts.

## Quantize in tmux

Use `tmux new -s strix-quant` before setting SOURCE_GGUF and OUTPUT_DIR above, or
set them again in that shell. Add --run to the successful plan command:

```bash
python3 qwen_strix_quant.py --source "$SOURCE_GGUF" \
  --output-dir "$OUTPUT_DIR" --run
```

Optional: append `--imatrix /absolute/path/to/matching-imatrix.gguf` to both
commands. Without it this is explicitly an uncalibrated screening candidate;
Q5_K and IQ4_NL support this mode. An imatrix must correspond to the same target
weights/tensor layout and a representative calibration corpus. Neither mode is
a quality certification.

At least 145 GiB of free output disk space is required, in addition to the source.
Eight CPU threads and a 1024 MiB quantization input buffer limit are defaults;
the buffer limit is not a total-process RAM cap. CPU and storage contention can
affect a running service even though this command does not stop/reconfigure it.

The helper runs the quantizer dry-run first, then writes into a new directory.
It refuses existing output directories. Failures retain partial files and logs;
do not deploy them. Success includes a complete inventory plus verification of
tensor names, shapes, explicit recipe types, token embedding, and output type.
No GPU run, quality evaluation, or performance claim is implied by this check.

If quantization completed but post-verification failed, retain the output. To
recheck it without running the quantizer or writing anything:

```bash
python3 qwen_strix_quant.py --source "$SOURCE_GGUF" \
  --output-dir "$OUTPUT_DIR" --verify-only
```

Shape comparisons accept trailing size-one axes being omitted by the C++ GGUF
writer (for example `[2560, 1]` becomes `[2560]`). Non-unit axes, axis order,
recipe types, and protected precision remain checked. A quantizer fallback
warning is separate: inspect `run.log` to identify the tensor and actual type.

Next qualification: gfx1151-only HIP and Vulkan, matched exact 32K prefill and
decode, followed by MTP n=3 and quality checks. Full 256K allocation comes after
the candidate passes these cheaper screens. No production switch is automatic.
