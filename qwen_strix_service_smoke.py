#!/usr/bin/env python3
"""Isolated gfx1151 256K allocation + prefix-cache + vision smoke.

Default: print a plan. --run temporarily stops an active production service,
starts only the pinned trial, and restores the service and archives on exit.
Does not modify production configuration. --capacity-validation replaces the
short smoke with occupied backing-cache and near-full-context validation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import os

from qwen_vision import shapes_fixture

PIN = "5f851647fe5ed795dfd6c0a3fba543114879e874"
TRIAL = pathlib.Path("/srv/llm/src/strix-llama-trial-5f851647")
MODELS = pathlib.Path("/srv/llm/models/qwen-flash-next")
SERVICE = "qwen-flash-next.service"
URL = "http://127.0.0.1:8189"
STATE_MARKER = b"MTP checkpoint carry restored:"


def state_patch_artifacts(server_path):
    server = pathlib.Path(server_path)
    found = []
    for path in (server, server.parent / "libllama-common.so", server.parent / "libllama-server-impl.so"):
        if not path.is_file():
            continue
        digest, tail, marked = hashlib.sha256(), b"", False
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                marked = marked or STATE_MARKER in tail + chunk
                tail = chunk[-len(STATE_MARKER):]
        found.append({"path": str(path), "sha256": digest.hexdigest(), "mtp_state_marker": marked})
    return found


def command_for(trial, models):
    return [str(trial / "build-hip10-dual/bin/llama-server"),
            "-m", str(models / "strix-bf16-joined.wPbjNP/quantized/Qwen3.8-Flash-Next-Q5_K-IQ4_NL-PLE-Q8_0.gguf"),
            "--host", "127.0.0.1", "--port", "8189", "--ctx-size", "262144",
            "--parallel", "1", "--device", "ROCm1", "--split-mode", "none",
            "--n-gpu-layers", "999", "--fit", "off", "--load-mode", "none",
            "--ngram-on-disk", "--ngram-direct-io", "--ngram-cache", "256",
            "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--batch-size", "2048", "--ubatch-size", "1536", "--threads", "16",
            "--cache-ram", "8192", "--ctx-checkpoints", "8", "--checkpoint-min-step", "32768",
            "--no-kv-unified", "--cache-idle-slots", "--no-context-shift", "--jinja", "-lv", "5",
            "-md", str(models / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"),
            "--spec-type", "draft-mtp", "--spec-draft-device", "ROCm1",
            "--spec-draft-ngl", "999", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.75",
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
            "--mmproj", str(models / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
            "--mmproj-offload", "--mmproj-device", "ROCm1",
            "--image-min-tokens", "1024", "--image-max-tokens", "2240"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def without_mtp(command):
    result, index = [], 0
    while index < len(command):
        if command[index] == "-md" or command[index].startswith("--spec-"):
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


def output_comparison(first, second):
    a, b = first.get("tokens", []), second.get("tokens", [])
    shared = lcp(a, b)
    equal = bool(a) and a == b
    return {"tokens_match": equal, "text_matches": first.get("content") == second.get("content"),
            "first_divergence_zero_based": None if equal else shared,
            "first_token_id": a[shared] if shared < len(a) else None,
            "second_token_id": b[shared] if shared < len(b) else None}


def repeat_verdict(results, prompt_length, mtp):
    errors = []
    for name in ("uncached_1", "uncached_2"):
        timing = results[name].get("timings", {})
        if timing.get("cache_n") != 0 or timing.get("prompt_n") != prompt_length:
            errors.append(f"{name}: not a full uncached {prompt_length}-token prefill")
    timing = results["cached"].get("timings", {})
    if timing.get("cache_n", 0) <= 0 or timing.get("cache_n", 0) + timing.get("prompt_n", 0) != prompt_length:
        errors.append("cached: prefix reuse was not demonstrated for the full input")
    for name in ("uncached_1", "uncached_2", "cached"):
        drafting = results[name].get("timings", {}).get("draft_n", 0) > 0
        if drafting != mtp:
            errors.append(f"{name}: MTP activity does not match the requested arm")
    uncached = output_comparison(results["uncached_1"], results["uncached_2"])
    cached = output_comparison(results["uncached_2"], results["cached"])
    if errors:
        status = "INVALID"
    elif not uncached["tokens_match"] or not uncached["text_matches"]:
        status = "INCONCLUSIVE_UNCACHED_DRIFT"
    elif not cached["tokens_match"] or not cached["text_matches"]:
        status = "CACHE_PATH_DRIFT"
    else:
        status = "PASS"
    return {"status": status, "errors": errors, "uncached_repeat": uncached,
            "cached_vs_second_uncached": cached,
            "note": "Cache-path drift does not distinguish numerical changes from a state-restoration bug."}


def repeat_probes(out, tokens, results, mtp):
    for label, cache_prompt in (("uncached_1", False), ("uncached_2", False), ("cached", True)):
        body = {"prompt": tokens, "n_predict": 32, "temperature": 0, "seed": 1234,
                "cache_prompt": cache_prompt, "return_tokens": True, "stream": False}
        write_json(out / f"{label}-request.json", body)
        print(f"{out.name}/{label}: {len(tokens)} input tokens; cache_prompt={cache_prompt}", flush=True)
        response = post("/completion", body)
        write_json(out / f"{label}-response.json", response)
        results[label] = response
        print(json.dumps({"timings": response.get("timings"), "content": response.get("content")}), flush=True)
        text_result(response)
    results["verdict"] = repeat_verdict(results, len(tokens), mtp)
    write_json(out / "repeat-summary.json", results["verdict"])
    print(json.dumps(results["verdict"], indent=2), flush=True)


def stop_server(server):
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)


def wait_ready(server):
    start, next_update = time.monotonic(), 0
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {server.returncode}")
        elapsed = time.monotonic() - start
        if elapsed > 600:
            raise TimeoutError("Startup exceeded 600 seconds")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                if response.status == 200:
                    print(f"READY after {elapsed:.1f}s", flush=True)
                    return
        except OSError:
            pass
        if elapsed >= next_update:
            print(f"Startup: {elapsed:.0f}s", flush=True)
            next_update = elapsed + 30
        time.sleep(2)


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def text_result(result):
    if not result.get("content", "").strip() or result.get("timings", {}).get("predicted_n", 0) < 2:
        raise ValueError("Degenerate text response; retained raw response")
    if not result.get("tokens"):
        raise ValueError("Server omitted generated token IDs; cannot check repeat equivalence")


def compare_cache(first, later, shared_with_current=0):
    cached = later.get("timings", {}).get("cache_n")
    return {"cached_tokens": cached,
            "reused_beyond_current_prefix": isinstance(cached, (int, float)) and cached > shared_with_current,
            "output_tokens_match": bool(first.get("tokens")) and first.get("tokens") == later.get("tokens"),
            "output_text_matches": first.get("content") == later.get("content")}


def evaluate(results):
    failures = []
    cache = results.get("cache_summary", {})
    for name in ("live", "after_b"):
        item = cache.get(name, {})
        if not item.get("reused_beyond_current_prefix"):
            failures.append(f"{name}: no demonstrated prefix reuse")
        if not item.get("output_tokens_match") or not item.get("output_text_matches"):
            failures.append(f"{name}: repeated greedy output changed; investigate state/numerics")
    if not cache.get("backing_restore_selection_logged"):
        failures.append("Backing-cache selection was not observed in the return-request log")
    if not any(results.get(name, {}).get("timings", {}).get("draft_n", 0) > 0
               for name in ("a_cold", "a_live", "a_return")):
        failures.append("No MTP drafting was reported on the text probes")
    vision = results.get("vision_summary", {})
    if not vision.get("content") or vision.get("missing_anchors"):
        failures.append("Vision fixture did not satisfy its keyword smoke check")
    return {"status": "FAIL" if failures else "PASS", "failures": failures,
            "scope": "Short cache/vision smoke at 256K allocation, not full-context qualification"}


def trial_environment():
    # Do not inherit production request-bypass, load, graph or device-remapping settings.
    removed = [key for key in os.environ if key.startswith(("LLAMA_", "GGML_")) or key in
               {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}]
    env = {key: value for key, value in os.environ.items() if key not in removed}
    env["LD_LIBRARY_PATH"] = "/opt/rocm-10.0.0/lib:/opt/rocm-10.0.0/lib64:" + env.get("LD_LIBRARY_PATH", "")
    return env, sorted(removed)


def post(endpoint, body, timeout=300):
    request = urllib.request.Request(URL + endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def memory_monitor(stop, samples):
    while not stop.is_set():
        row = {"timestamp": time.time()}
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
                row[key + "_GiB"] = int(value.split()[0]) / 1048576
        samples.append(row)
        stop.wait(1)


def probes(out, tokens, results):
    def text_probe(label, prompt):
        body = {"prompt": prompt, "n_predict": 32, "temperature": 0, "seed": 1234,
                "stream": False, "cache_prompt": True, "return_tokens": True}
        # No id_slot: the server's automatic cache save/load path must run.
        write_json(out / f"{label}-request.json", body)
        print(f"{label}: {len(prompt)} input tokens", flush=True)
        result = post("/completion", body)
        write_json(out / f"{label}-response.json", result)
        results[label] = result
        print(json.dumps({"label": label, "timings": result.get("timings"), "content": result.get("content")}), flush=True)
        text_result(result)
        return result

    b_chat = {"messages": [{"role": "user", "content":
              "Explain why 17 is a prime number in three complete sentences."}],
              "chat_template_kwargs": {"enable_thinking": False}}
    rendered = post("/apply-template", b_chat)["prompt"]
    b = post("/tokenize", {"content": rendered, "add_special": False, "parse_special": True})["tokens"]
    shared = lcp(tokens, b)
    write_json(out / "prompt-tokens.json", tokens)
    first = text_probe("a_cold", tokens)
    live = text_probe("a_live", tokens)
    text_probe("b_diversion", b)
    restore_log_start = (out / "server.log").stat().st_size
    returned = text_probe("a_return", tokens)
    with (out / "server.log").open("rb") as handle:
        handle.seek(restore_log_start)
        restore_log = handle.read().decode(errors="replace")
    results["cache_summary"] = {
        "live": compare_cache(first, live),
        "after_b": compare_cache(first, returned, shared),
        "a_b_shared_prefix_tokens": shared,
        "backing_restore_selection_logged": "found better prompt with f_keep" in restore_log,
        "cache_cap_MiB": 8192, "cache_filled_to_cap": False,
    }
    write_json(out / "cache-summary.json", results["cache_summary"])

    png = shapes_fixture()
    (out / "shapes.png").write_bytes(png)
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "Answer in one compact line: left=<color> <shape>; center=<color> <shape>; right=<color> <shape>; bottom text=<exact transcription>. Do not explain."}]}],
        "max_tokens": 128, "temperature": 0, "seed": 1234, "stream": False,
        "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    write_json(out / "vision-request.json", body)
    print("vision: shapes fixture with MTP still enabled", flush=True)
    vision = post("/v1/chat/completions", body)
    write_json(out / "vision-response.json", vision)
    results["vision"] = vision
    content = vision["choices"][0]["message"].get("content") or ""
    anchors = ["red", "circle", "blue", "square", "green", "triangle", "UNSLOTH 42"]
    results["vision_summary"] = {"content": content,
        "missing_anchors": [a for a in anchors if a.lower() not in content.lower()],
        "mtp_drafting_reported": (vision.get("timings") or {}).get("draft_n", 0) > 0,
        "timings": vision.get("timings"), "note": "Keyword smoke only; not a general vision qualification"}
    write_json(out / "vision-summary.json", results["vision_summary"])
    print(json.dumps(results["vision_summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=pathlib.Path, default=TRIAL)
    parser.add_argument("--model-dir", type=pathlib.Path, default=MODELS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeat-ab", action="store_true",
                        help="Replace cache/vision smoke with uncached/uncached/cached, in fresh MTP-on/off servers")
    parser.add_argument("--capacity-validation", action="store_true",
                        help="Saturate the backing cache, then run one exact 253952-token prompt")
    parser.add_argument("--require-mtp-state-patch", action="store_true",
                        help="Require the compiled state patch and its execution on the cached MTP request")
    args = parser.parse_args()
    capacity = getattr(args, "capacity_validation", False)
    require_state = getattr(args, "require_mtp_state_patch", False) or capacity
    if capacity and args.repeat_ab:
        parser.error("--capacity-validation and --repeat-ab are mutually exclusive")
    if require_state and not (args.repeat_ab or capacity):
        parser.error("--require-mtp-state-patch requires --repeat-ab or --capacity-validation")
    command = command_for(args.trial_dir, args.model_dir)
    print(json.dumps({"command": command, "repeat_ab": args.repeat_ab,
                      "capacity_validation": capacity,
                      "probes": (["Fill 8 GiB cache to >=95% with capacity eviction", "Verify saturated backing restore",
                                  "One exact 253952-token prefill and <=128-token decode", "Observe near-full state retention"]
                                 if capacity else ["MTP on: uncached / uncached / cached", "MTP off: uncached / uncached / cached"]
                                 if args.repeat_ab else ["A cold 32K", "A live repeat", "B diversion", "A backing restore", "vision shapes"])}, indent=2), flush=True)
    if not args.run:
        print("Plan only. --run is an isolated trial; production settings are never edited.")
        return 0
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run the trial on the Linux GPU host")
    revision = subprocess.check_output(["git", "-C", str(args.trial_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected pinned trial revision {PIN}; found {revision}")
    for flag in ("-m", "-md", "--mmproj"):
        path = pathlib.Path(command[command.index(flag) + 1])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {flag} file: {path}")
    if not os.access(command[0], os.X_OK):
        raise FileNotFoundError(f"Missing/non-executable server: {command[0]}")
    artifacts = state_patch_artifacts(command[0])
    if require_state and not any(item["mtp_state_marker"] for item in artifacts):
        raise RuntimeError("Compiled MTP state marker missing; run build-strix-mtp-checkpoint.sh first")
    baseline = args.trial_dir / "trial-results/hip-templated-32k.xhe6lmo5/prompt-tokens.json"
    tokens = [] if capacity else json.loads(baseline.read_text())
    if not capacity and (not isinstance(tokens, list) or len(tokens) != 32768 or not all(type(t) is int for t in tokens)):
        raise ValueError(f"Invalid exact-token baseline: {baseline}")
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", 8189)) == 0:
            raise RuntimeError("Port 8189 is occupied; stop the previous test first")
    subprocess.run(["sudo", "-v"], check=True)
    prefix = "hip-cache-full-context." if capacity else "hip-cache-repeat-ab." if args.repeat_ab else "hip-cache-vision-256k."
    out = pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=args.trial_dir / "trial-results"))
    write_json(out / "command.json", command)
    write_json(out / "revision.json", {"trial_revision": revision})
    write_json(out / "binary-fingerprint.json", artifacts)
    print(f"Results: {out}", flush=True)
    server = log = monitor = None
    restore = False
    stop = threading.Event()
    samples, results = [], {}
    status = 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        restore = subprocess.run(["systemctl", "is-active", "--quiet", SERVICE]).returncode == 0
        if restore:
            subprocess.run(["sudo", "systemctl", "stop", SERVICE], check=True)
        monitor = threading.Thread(target=memory_monitor, args=(stop, samples), daemon=True)
        monitor.start()
        env, removed = trial_environment()
        write_json(out / "environment-isolation.json", {"removed_variable_names": removed})
        arms = [("mtp_on", command), ("mtp_off", without_mtp(command))] if args.repeat_ab else [("smoke", command)]
        for name, arm_command in arms:
            arm_out = out / name if args.repeat_ab else out
            arm_out.mkdir(exist_ok=True)
            arm_results = {} if args.repeat_ab else results
            if args.repeat_ab:
                results[name] = arm_results
            write_json(arm_out / "command.json", arm_command)
            print(f"Starting {name}", flush=True)
            log = (arm_out / "server.log").open("wb")
            server = subprocess.Popen(arm_command, stdout=log, stderr=subprocess.STDOUT, env=env)
            wait_ready(server)
            with urllib.request.urlopen(URL + "/props", timeout=10) as response:
                props = json.load(response)
            write_json(arm_out / "props.json", props)
            if not props.get("modalities", {}).get("vision") or props.get("default_generation_settings", {}).get("n_ctx") != 262144:
                raise RuntimeError("Server did not confirm vision plus 262144 context")
            if capacity:
                from qwen_strix_capacity import capacity_probes
                capacity_probes(arm_out, arm_results, server)
            elif args.repeat_ab:
                repeat_probes(arm_out, tokens, arm_results, mtp=name == "mtp_on")
                if require_state and name == "mtp_on":
                    restored = STATE_MARKER.decode() in (arm_out / "server.log").read_text(errors="replace")
                    arm_results["verdict"]["mtp_state_restore_logged"] = restored
                    if not restored:
                        arm_results["verdict"]["status"] = "INVALID"
                        arm_results["verdict"].setdefault("errors", []).append("Patched MTP state restore was not observed")
                    write_json(arm_out / "repeat-summary.json", arm_results["verdict"])
                    print(json.dumps(arm_results["verdict"], indent=2), flush=True)
            else:
                probes(arm_out, tokens, arm_results)
                arm_results["verdict"] = evaluate(arm_results)
                print(json.dumps(arm_results["verdict"], indent=2), flush=True)
            status = max(status, int(arm_results["verdict"]["status"] != "PASS"))
            stop_server(server)
            server = None
            log.close()
            log = None
        if args.repeat_ab:
            results["cross_arm_second_uncached"] = output_comparison(
                results["mtp_on"]["uncached_2"], results["mtp_off"]["uncached_2"])
            write_json(out / "comparison.json", {name: value["verdict"] for name, value in results.items()
                                                 if name in ("mtp_on", "mtp_off")})
    except BaseException as error:
        status = 1
        results["error"] = f"{type(error).__name__}: {error}"
        print("ERROR: " + results["error"], file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        stop_server(server)
        if log is not None:
            log.close()
        stop.set()
        if monitor is not None:
            monitor.join(timeout=2)
        results["minimum_host_available_GiB"] = min((s["MemAvailable_GiB"] for s in samples), default=None)
        results["production_was_active"] = restore
        if restore:
            print("Restoring production...", flush=True)
            results["production_start_succeeded"] = subprocess.run(["sudo", "systemctl", "start", SERVICE]).returncode == 0
            print(f"Production start succeeded: {results['production_start_succeeded']}", flush=True)
            if not results["production_start_succeeded"]:
                status = 1
        else:
            print("Production was not active initially; leaving it unchanged.", flush=True)
        write_json(out / "memory-samples.json", samples)
        write_json(out / "results.json", results)
        archive = pathlib.Path(str(out) + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(out, arcname=out.name)
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        pathlib.Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
        print(f"Archive: {archive}", flush=True)
    print(f"Trial exit status: {status}. No production qualification is implied.", flush=True)
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
