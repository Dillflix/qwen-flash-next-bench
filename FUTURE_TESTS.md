# Future benchmark subjects

## General Qwen4Exp PLE/MTP state lifecycle

The current state-correctness candidate is intentionally scoped to the qualified
one-slot service: one target context, `--cache-ram 0`, `--no-context-shift`, and
no imported slot state. Its full PLE token journal can truncate an in-process
speculative or prompt-checkpoint rewind exactly, but that journal is still owned
by the model rather than llama context memory.

Before enabling MTP outside that topology, move or serialize PLE history with the
rest of per-sequence state and test sequence copy/add/keep, context shifting,
multiple contexts sharing a model, explicit slot save/load, and restore after a
server restart. A separate stop-boundary test must force EOG inside a multi-token
accepted draft and verify that target KV, recurrent state, PLE history, and the MTP
boundary are rolled back to the last token actually emitted to the caller.

## Bounded SSD-backed prompt-response cache

Status: deferred until native 256K capacity is proven. Production controls use
`--cache-ram 0`; an 8 GiB anonymous-RAM cache is not an acceptable substitute on
the 128 GiB unified-memory host.

The feature is valuable if repeated prompt state can live on SSD and be restored
without permanently consuming unified RAM. `--slot-save-path` by itself only gives
the server a directory for explicit slot save/restore actions. It does not establish
automatic cache lookup, eviction, persistence, or an 8 GiB disk budget.

The current ROCmFPX server log explicitly advertises `--cache-disk PATH` alongside
`--cache-ram N`. That makes its native disk cache the implementation to test; it is
no longer necessary to wait for a new cache implementation. Before adding a matrix
tier, capture `llama-server --help` for its companion size/eviction flags and verify
whether the cache includes both target and MTP state.

The native implementation is eligible for production only after it proves:

- a bounded on-disk store with automatic lookup and eviction;
- cache keys covering model/build identity, sampler-relevant prompt bytes, target
  and MTP state, chat template, and vision inputs;
- compatible target KV and draft/MTP state restoration, or a documented safe
  target-only fallback;
- atomic writes plus rejection of truncated, stale, or version-incompatible entries;
- metrics for hits, misses, bytes read/written, restore time, and evictions;
- no persistent increase in 7900 XT VRAM, iGPU GTT, or anonymous host RSS.

The acceptance matrix must compare `--cache-ram 0` against disk-cache cold miss,
first hit, repeated hit, restart hit, and eviction pressure at 64K and near 256K.
Record time to first generated token, total prefill time, physical storage reads and
writes, major faults, `MemAvailable`, output hash, and MTP acceptance. It becomes a
production option only if cache hits materially reduce latency while the near-full
one-slot capacity proof still passes.
