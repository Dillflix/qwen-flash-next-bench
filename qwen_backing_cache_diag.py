#!/usr/bin/env python3
"""Isolated one-slot strict-MTP backing-cache A/B; no production settings changed."""
from __future__ import annotations

import argparse
import json
import hashlib
import mmap
import os
import pathlib
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time

import qwen_bench as bench
import qwen_mtp_diag as mtp
import qwen_prefix_diag as prefix

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = "1.1.0"


def payload(family: str, revision: int, blocks: int) -> tuple[dict, dict]:
    codes = [f"{family}{i:03d}" for i in range(12)]
    expected = {"ledger": f"{family}_LEDGER_314159", "revision": revision, "codes": codes}
    body = {
        "model": "Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX.gguf",
        "messages": [
            {"role": "system", "content": f"Conversation {family}. Output only the requested JSON object."},
            {"role": "user", "content": prefix.make_prefix(family, expected["ledger"], blocks)
             + "\nThe ordered code list is: " + json.dumps(codes)},
            {"role": "user", "content": f"Return the ledger key, revision {revision}, and ordered codes. "
             "Use exactly the fields ledger, revision, codes."},
        ],
        "temperature": 0, "seed": 1234, "max_tokens": 192,
        "stream": False, "cache_prompt": True,
        "speculative.n_max": 3,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True, "top_logprobs": 1,
    }
    return body, expected


def snapshot(pid: int) -> dict:
    row = {"ts": mtp.utc_now(), "pid": pid}
    for name, path in (("meminfo", "/proc/meminfo"), ("status", f"/proc/{pid}/status"),
                       ("io", f"/proc/{pid}/io")):
        try:
            values = {}
            for line in pathlib.Path(path).read_text().splitlines():
                key, _, value = line.partition(":")
                parts = value.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * (1024 if parts[-1] == "kB" else 1)
            row[name] = values
        except OSError:
            pass
    row["gpu"] = {}
    for card in pathlib.Path("/sys/class/drm").glob("card[0-9]*"):
        if "-" in card.name:
            continue
        dev = card / "device"
        gpu = {}
        for field in ("mem_info_vram_used", "mem_info_gtt_used", "gpu_busy_percent"):
            try:
                gpu[field] = int((dev / field).read_text())
            except (OSError, ValueError):
                pass
        if gpu:
            row["gpu"][dev.resolve().name] = gpu
    return row


def record_memory(pid: int, path: pathlib.Path, done: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as out:
        while not done.is_set():
            out.write(json.dumps(snapshot(pid)) + "\n")
            out.flush()
            done.wait(1)


def check_runtime_patch() -> dict:
    server = pathlib.Path(os.environ.get("LLAMA_SERVER") or
        "/srv/llm/src/ROCmFPX-qwen4exp/build-hip10-dual/bin/llama-server")
    library = server.parent / "libllama.so"
    if not library.is_file():
        library = server.parent.parent / "lib/libllama.so"
    result = {}
    for name, path, marker in (
        ("server", server, b"prompt cache checkpoint candidate: source=ram"),
        ("llama_library", library, b"Qwen4Exp PLE snapshot version mismatch"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing {path}; rebuild with ./build-rocm10-dual.sh")
        with path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                if data.find(marker) < 0:
                    raise RuntimeError(f"{path} lacks the backing-cache patch; rebuild with ./build-rocm10-dual.sh")
                result[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    return result


def arm(run_dir: pathlib.Path, label: str, mib: int, args, api_key: str) -> list[dict]:
    # The shell runner loads the existing paths/runtime first. Authentication is
    # overridden with an ephemeral key, and the server only binds to loopback.
    env = dict(os.environ)
    env.update({
        "LLAMA_API_KEY": api_key, "LLAMA_ARG_API_KEY_FILE": "", "QWEN_API_KEY": "", "API_KEY": "",
        "LLAMA_HOST": "127.0.0.1", "LLAMA_PORT": str(args.port),
        "LLAMA_MTP_MODE": "strict", "LLAMA_PARALLEL": "1", "LLAMA_CONTEXT_SIZE": "262144",
        "LLAMA_MULTI_SLOT_DIAGNOSTIC": "0", "LLAMA_SLOT_SAVE_PATH": "",
        "LLAMA_DISABLE_HIP_GRAPHS": "1", "LLAMA_BACKING_CACHE_DIAGNOSTIC": "1",
        "LLAMA_CACHE_RAM_MIB": str(mib),
    })
    for key in list(env):
        if key.startswith("LLAMA_MTP_DIAG_") or key.startswith("LLAMA_ARG_CACHE_DISK"):
            env.pop(key)
    base_url = f"http://127.0.0.1:{args.port}"
    log = run_dir / f"server-{label}.log"
    rows = []
    process = None
    done = threading.Event()
    monitor = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(["bash", str(ROOT / "deployment/run-production.sh")],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            monitor = threading.Thread(target=record_memory,
                args=(process.pid, run_dir / f"memory-{label}.jsonl", done), daemon=True)
            monitor.start()
            deadline = time.monotonic() + args.startup_timeout
            next_update = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} server exited with {process.returncode}; see {log}")
                try:
                    health = mtp.http_request("GET", base_url + "/health", None, api_key, 2)
                    if health["status"] == 200:
                        break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{label} startup timeout; see {log}")
                if time.monotonic() > next_update:
                    print(f"{label}: waiting for model load", flush=True)
                    next_update = time.monotonic() + 30
                time.sleep(1)

            props = mtp.http_request("GET", base_url + "/props", None, api_key, 10)
            if props["status"] != 200 or json.loads(props["body"]).get("total_slots") != 1:
                raise RuntimeError("Diagnostic requires a confirmed single server slot")
            command = pathlib.Path(f"/proc/{process.pid}/cmdline").read_bytes().decode().split("\0")
            if "--cache-ram" not in command or command[command.index("--cache-ram") + 1] != str(mib):
                raise RuntimeError("Running process did not receive the intended cache capacity")
            mtp.atomic_json(run_dir / f"process-{label}.json", {
                "command": mtp.redact_secret_value(command, api_key),
                "cache_ram_mib": mib, "props": mtp.redact_secret_value(json.loads(props["body"]), api_key),
            })
            # A fresh process per arm avoids accidental backing-cache hits in
            # the cache-off reference. Never erase between A/B/A requests.
            sequence = [("prime", "P", 0, 96), ("a-establish", "A", 0, args.blocks),
                        ("b-displace", "B", 0, args.blocks),
                        ("a-resume", "A", 1, args.blocks),
                        ("b-resume", "B", 1, args.blocks)]
            for name, family, revision, blocks in sequence:
                body, expected = payload(family, revision, blocks)
                identifier = f"{label}-{name}"
                print(f"{identifier}: MTP n=3, backing cache {mib} MiB", flush=True)
                before = snapshot(process.pid)
                row = prefix.run_request(run_dir, base_url, {"case_id": identifier},
                                         body, api_key, args.request_timeout, log)
                row.update({"arm": label, "case": name, "cache_ram_mib": mib,
                            "memory_before": before, "memory_after": snapshot(process.pid)})
                try:
                    row["contract_pass"] = (json.loads(row.get("content", "")) == expected
                                             and row.get("finish_reason") == "stop")
                except ValueError:
                    row["contract_pass"] = False
                if row.get("parsed_file"):
                    parsed = json.loads((run_dir / row["parsed_file"]).read_text())
                    row["timings"] = parsed["timings"]
                    row["completion_tokens"] = parsed["usage"].get("completion_tokens")
                with (run_dir / "results.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")
                rows.append(row)
                if row["status"] != "ok":
                    raise RuntimeError(row.get("error", "request failed"))
                timing = row.get("timings", {})
                print(f"  cache_n={prefix.cache_count(row)}, prompt_ms={timing.get('prompt_ms')}, "
                      f"decode={timing.get('predicted_per_second')} tok/s, "
                      f"JSON={'PASS' if row['contract_pass'] else 'FAIL'}", flush=True)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            done.set()
            if monitor:
                monitor.join(timeout=5)
        if log.exists():
            log.write_bytes(mtp.redact_secret_bytes(log.read_bytes(), api_key))
    return rows


def checkpoint_evidence(run_dir: pathlib.Path, row: dict) -> dict:
    path = run_dir / "request-logs" / f"{row['case_id']}.log" if row.get("case_id") else None
    text = path.read_text(errors="replace") if path and path.is_file() else ""
    candidates = re.findall(r"prompt cache checkpoint candidate: source=ram lcp=(\d+) reusable=(\d+)", text)
    restored = re.findall(r"prompt cache checkpoint rollback: lcp=(\d+).*?n_past=(\d+) spec_state=(\d+)", text)
    count = prefix.cache_count(row)
    for lcp, n_past, spec_bytes in restored:
        if (lcp, n_past) in candidates and int(spec_bytes) > 0 and count == int(n_past) and count > 0:
            return {"verified": True, "lcp": int(lcp), "reused": count}
    return {"verified": False}


def report(run_dir: pathlib.Path, rows: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in rows}
    comparisons = []
    for case in ("a-resume", "b-resume"):
        cold = by_id.get(f"off-{case}", {})
        cached = by_id.get(f"ram-{case}", {})
        comparison = prefix.exact_output_comparison(run_dir, cached, cold)
        count = prefix.cache_count(cached)
        total = cached.get("usage_prompt_n") or 0
        checkpoint = checkpoint_evidence(run_dir, cached)
        comparison.update({"case": case, "cache_n": count,
                           "checkpoint_restore": checkpoint,
                           "cache_hit_pass": count is not None and total > 0 and 0 < count <= total
                               and (count / total >= 0.5 or checkpoint["verified"]),
                           "token_evidence_present": all(r.get("token_count", 0) > 0
                               and r.get("token_count") == r.get("completion_tokens")
                               and r.get("token_evidence", {}).get("identity_complete") is True
                               for r in (cold, cached)),
                           "mtp_on_resume": bool(cold.get("draft_n") and cached.get("draft_n")),
                           "cold_reference_pass": prefix.cache_count(cold) is not None
                               and (cold.get("usage_prompt_n") or 0) > 0
                               and prefix.cache_count(cold) < cold["usage_prompt_n"] * 0.1,
                           "contract_pass": cold.get("contract_pass", False) and cached.get("contract_pass", False)})
        comparisons.append(comparison)
    measured = [row for row in rows if row["case"] != "prime"]
    result = {"schema": 1, "comparisons": comparisons,
              "mtp_observed_in_both_arms": all(any(r.get("draft_n", 0) and r["arm"] == a for r in measured) for a in ("off", "ram")),
              "all_contracts_pass": len(measured) == 8 and all(r.get("contract_pass") for r in measured)}
    result["passed"] = bool(result["mtp_observed_in_both_arms"] and result["all_contracts_pass"]
        and all(c["passed"] and c["cache_hit_pass"] and c["token_evidence_present"]
                and c["mtp_on_resume"] and c["cold_reference_pass"] for c in comparisons))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument("--blocks", type=int, default=96)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--output-root", type=pathlib.Path, default=ROOT / "results")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run on the Linux Qwen host")
    if not 1 <= args.blocks <= 512 or not 1 <= args.port <= 65535:
        parser.error("blocks must be 1..512 and port 1..65535")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    run_dir = args.output_root / f"{mtp.stamp()}-backing-cache-one-slot"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results: {run_dir}", flush=True)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    api_key = secrets.token_urlsafe(32)
    rows = []
    failure = None
    mtp.atomic_json(run_dir / "manifest.json", {"version": VERSION, "arms_mib": [0, 8192],
        "parallel": 1, "context": 262144, "mtp_n": 3, "hip_graphs": False,
        "blocks": args.blocks, "note": "Short prompts; 256K allocation, not 256K prefill. No slot erasure. Fixed order off then RAM."})
    try:
        mtp.atomic_json(run_dir / "runtime-fingerprint.json", check_runtime_patch())
        for label, mib in (("off", 0), ("ram", 8192)):
            rows.extend(arm(run_dir, label, mib, args, api_key))
    except (Exception, KeyboardInterrupt) as exc:
        failure = mtp.redact_secret_text(repr(exc), api_key)
        print(f"ERROR: {failure}", file=sys.stderr)
    finally:
        # Reload incremental rows to retain a partially completed arm on failure.
        path = run_dir / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        result = report(run_dir, rows)
        result["error"] = failure
        result["passed"] = result["passed"] and failure is None
        mtp.atomic_json(run_dir / "report.json", result)
        summary = ["# One-slot backing-cache A/B", "",
            f"Result: {'PASS' if result['passed'] else 'FAIL'}", "",
            "| Arm | Request | Cached tokens | Prefill ms | Decode tok/s | Wall ms | JSON |",
            "|---|---|---:|---:|---:|---:|---|"]
        for row in rows:
            t = row.get("timings", {})
            summary.append(f"| {row['arm']} | {row['case']} | {prefix.cache_count(row)} | "
                f"{t.get('prompt_ms')} | {t.get('predicted_per_second')} | "
                f"{row.get('http_wall_ms')} | {row.get('contract_pass')} |")
        summary += ["", "See report.json for exact-output, cache-hit and MTP gates. "
                    "Fixed off-then-RAM order; memory telemetry includes PLE page-cache effects."]
        if failure:
            summary += ["", f"Error: {failure}"]
        (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        archive = bench.archive_results(run_dir)
        print(f"Backing-cache restoration: {'PASS' if result['passed'] else 'FAIL'}\nArchive: {archive}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
