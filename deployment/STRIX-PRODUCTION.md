# gfx1151 production migration

This migrates the existing `qwen-flash-next.service` to the tested Strix runtime.
It does not rebuild, requantize, replace the original production launcher, edit
credentials, change firewall rules, or alter the existing bind address/port.
Paths and service user `jdillman` match this installation deliberately.

## Profile

- Target mixed Q5_K/IQ4_NL with Q8_0 PLE, Q8_0 MTP n=3, and BF16 vision on ROCm1.
- One slot, 262144 context, F16 target/draft KV, batch 2048, ubatch 1536.
- Ordinary weights: `--load-mode none`; PLE: SSD direct I/O plus row cache.
- Backing cache: **12288 MiB**; eight checkpoints, minimum spacing 32768.
- API alias: `Qwen/Qwen3.8-Flash-Next-think`.

The inference command matches the tested retention profile except host/port,
the explicit API alias, and log verbosity reduced from 5 to 3. The 7900 XT may
be enumerated, but target/draft/projector weights are placed on gfx1151 only.
Legacy `LLAMA_*`, `GGML_*` and GPU-visibility overrides are stripped, then native
authentication is restored. Old 16 GiB cache, strict-mode, vision-bypass,
dual-GPU and HIP graph-disable settings cannot silently alter this launch.

## Deploy

After publication, inside tmux with no other Strix trial running:

```bash
cd /srv/llm/src/llama-qwen4exp/qwen-flash-next-bench &&
git pull --ff-only &&
python3 -m unittest discover -s tests -p 'test_strix_production.py' &&
sudo bash deployment/install-strix-production.sh
```

No rebuild is needed. The installer requires the existing service with
User=jdillman and enabled/disabled state. It refuses conflicting trial processes
and symlinked deployment targets. It creates a private root-owned backup at
`/var/backups/qwen-strix.XXXXXX` and prints its rollback command before switching.

It installs `/usr/local/libexec/qwen-strix-production.py` and the systemd drop-in
`90-strix-runtime.conf`. This replaces ExecStartPre/ExecStart/ExecStartPost, while
retaining existing hardening, credentials and networking. The original launcher,
unit, environment/key files and backing-cache drop-in remain intact. The service
is enabled at boot on success. Failure during deployment triggers rollback;
rollback failure is reported explicitly rather than claimed successful.

ExecStartPre requires readable models, nonempty credentials and successful
near-full MTP retention evidence at:

`/srv/llm/src/strix-llama-trial-5f851647/trial-results/hip-near-full-retention.p0ia0m_5`

It compares the three recorded hashes: `llama-server`, `libllama-common.so` and
`libllama-server-impl.so`. This is not a complete HIP dependency fingerprint or
a rehash of model weights. Do not replace other libraries/models behind these
validated paths. Future evidence can be selected using `STRIX_VALIDATION_DIR`
in the existing service environment file; changed recorded binaries fail closed.

ExecStartPost waits up to 600 seconds for health, checks authenticated model
listing/alias, verifies 256K allocation and vision availability, and issues one
short authenticated text completion requiring MTP drafting. `systemctl restart`
waits for this check. There is no long prompt or full vision suite at startup.

## Authentication and endpoint

Existing EnvironmentFile entries supply `LLAMA_HOST`, `LLAMA_PORT` and credentials.
An unset host defaults to loopback; this migration never silently opens a new
interface. Keep the current working LiteLLM endpoint. No new caller-specific
MTP/vision-bypass fields are required by this launcher.

Authentication is mandatory, including on loopback. Inputs, in priority order:

1. `--api-key KEY` or `--api-key-file PATH` for manual launches.
2. `LLAMA_API_KEY` or `LLAMA_ARG_API_KEY_FILE` from the existing environment.
3. `QWEN_API_KEY` / `API_KEY` compatibility aliases.

Explicit command-line credentials override environment credentials. Do not set
both environment forms. A key file must have at least one nonempty line.
Credentials reach llama-server through native environment variables, not argv.
Prefer the protected key file: a CLI key can briefly appear in the launcher's
arguments or shell history. The launcher and readiness check do not print keys
or completion text. Environment files are never sourced as shell code.

## Verify and rollback

```bash
systemctl status qwen-flash-next.service --no-pager
sudo journalctl -u qwen-flash-next.service -n 60 --no-pager
```

Look for `Strix production ready` and the gfx1151-only launch summary. Test a
normal conversation and an image through LiteLLM/Open WebUI; the startup check
does not itself verify those external applications.

Use the exact backup path printed during installation:

```bash
sudo bash deployment/install-strix-production.sh --rollback /var/backups/qwen-strix.XXXXXX
```

Rollback restores previous Strix launcher/drop-in files, or removes just those
two files if newly created. It restores the prior active/inactive and
enabled/disabled service state. If old production was inactive, it remains
inactive after rollback; start it explicitly if desired. Backups remain and no
model/runtime tree is deleted.

## Memory limitation

Near-full retention passed with a 9.80 GiB saved state under this 12 GiB limit,
but available RAM dipped to **2.68 GiB during saving/diversion**. The fully
occupied 8 GiB test separately passed near-full inference. A fully occupied
12 GiB cache plus another large save and long-term endurance remain untested.
The limit is not preallocated. Do not treat remaining unified RAM as freely
available for other workloads, especially during large conversation switches.
The isolated test's memory-abort guard is not installed as a production
watchdog. Monitor available RAM and the journal during rollout.
