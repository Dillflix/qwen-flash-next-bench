#!/usr/bin/env python3
"""Isolated gfx1151 256K allocation + prefix-cache + vision smoke.

Default: print a plan. --run temporarily stops an active production service,
starts only the pinned trial, and restores the service and archives on exit.
Does not modify production configuration or fill the backing cache to its cap.
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
    args = parser.parse_args()
    command = command_for(args.trial_dir, args.model_dir)
    print(json.dumps({"command": command, "probes": ["A cold 32K", "A live repeat", "B diversion", "A backing restore", "vision shapes"]}, indent=2), flush=True)
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
    baseline = args.trial_dir / "trial-results/hip-templated-32k.xhe6lmo5/prompt-tokens.json"
    tokens = json.loads(baseline.read_text())
    if not isinstance(tokens, list) or len(tokens) != 32768 or not all(type(t) is int for t in tokens):
        raise ValueError(f"Invalid exact-token baseline: {baseline}")
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", 8189)) == 0:
            raise RuntimeError("Port 8189 is occupied; stop the previous test first")
    subprocess.run(["sudo", "-v"], check=True)
    out = pathlib.Path(tempfile.mkdtemp(prefix="hip-cache-vision-256k.", dir=args.trial_dir / "trial-results"))
    write_json(out / "command.json", command)
    write_json(out / "revision.json", {"trial_revision": revision})
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
        log = (out / "server.log").open("wb")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
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
                        break
            except OSError:
                pass
            if elapsed >= next_update:
                print(f"Startup: {elapsed:.0f}s", flush=True)
                next_update = elapsed + 30
            time.sleep(2)
        print(f"READY after {elapsed:.1f}s", flush=True)
        with urllib.request.urlopen(URL + "/props", timeout=10) as response:
            props = json.load(response)
        write_json(out / "props.json", props)
        if not props.get("modalities", {}).get("vision") or props.get("default_generation_settings", {}).get("n_ctx") != 262144:
            raise RuntimeError("Server did not confirm vision plus 262144 context")
        probes(out, tokens, results)
        results["verdict"] = evaluate(results)
        print(json.dumps(results["verdict"], indent=2), flush=True)
        status = int(results["verdict"]["status"] != "PASS")
    except BaseException as error:
        status = 1
        results["error"] = f"{type(error).__name__}: {error}"
        print("ERROR: " + results["error"], file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
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
