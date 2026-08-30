#!/usr/bin/env python3
"""Prime the qualified Qwen4Exp strict-MTP graph after llama-server starts."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request


VERSION = "1.0.0"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def positive_number(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def api_key() -> str:
    direct = (
        os.environ.get("LLAMA_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    ).strip()
    key_file = os.environ.get("LLAMA_ARG_API_KEY_FILE", "").strip()
    if direct and key_file:
        raise ValueError("configure either LLAMA_API_KEY or LLAMA_ARG_API_KEY_FILE, not both")
    if direct:
        key = direct.split(",", 1)[0].strip()
        if not key:
            raise ValueError("LLAMA_API_KEY contains no usable key")
        return key
    if not key_file:
        return ""
    path = pathlib.Path(key_file)
    if not path.is_file():
        raise ValueError(f"API-key file is not readable: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        key = line.strip()
        if key:
            return key
    raise ValueError(f"API-key file contains no usable key: {path}")


def endpoint() -> str:
    override = os.environ.get("LLAMA_STARTUP_PRIME_URL", "").strip()
    if override:
        return override.rstrip("/")
    host = os.environ.get("LLAMA_HOST", "127.0.0.1").strip()
    port = os.environ.get("LLAMA_PORT", "8080").strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("LLAMA_PORT must be an integer from 1 through 65535")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def request(url: str, key: str, *, method: str = "GET", body: dict | None = None,
            timeout: float = 30.0) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    response = opener.open(
        urllib.request.Request(url, data=data, headers=headers, method=method),
        timeout=timeout,
    )
    with response:
        return response.status, response.read()


def wait_ready(base_url: str, key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server has not answered"
    while time.monotonic() < deadline:
        try:
            status, _body = request(f"{base_url}/health", key, timeout=10.0)
            if status == 200:
                return
            last_error = f"health returned HTTP {status}"
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"llama-server did not become healthy within {timeout:g}s: {last_error}")


def prime_payload() -> dict:
    # More than one 1536-token ubatch, without performing a near-context prefill.
    calibration = " warmup" * 2048
    return {
        "model": "Qwen/Qwen3.8-Flash-Next-think",
        "messages": [{
            "role": "user",
            "content": (
                "Read this synthetic startup calibration text. Do not summarize it."
                f"{calibration}\nReply only with READY."
            ),
        }],
        "max_tokens": 16,
        "temperature": 0,
        "seed": 1234,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "speculative.n_max": 3,
    }


def run_prime(base_url: str, key: str, timeout: float) -> None:
    status, raw = request(
        f"{base_url}/v1/chat/completions",
        key,
        method="POST",
        body=prime_payload(),
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"startup prime returned HTTP {status}")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("startup prime returned invalid JSON") from exc
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("startup prime returned no completion choice")
    timings = result.get("timings") or {}
    draft_n = timings.get("draft_n")
    accepted = timings.get("draft_n_accepted")
    if not isinstance(draft_n, (int, float)) or draft_n <= 0:
        raise RuntimeError("startup prime did not exercise MTP drafting")
    prompt_n = timings.get("prompt_n", "unknown")
    print(
        "Strict MTP startup prime complete: "
        f"prompt_n={prompt_n}, draft_accepted={accepted}/{draft_n}",
        flush=True,
    )


def self_test() -> None:
    global request
    old = dict(os.environ)
    original_request = request
    try:
        os.environ.clear()
        assert endpoint() == "http://127.0.0.1:8080"
        os.environ.update({"LLAMA_HOST": "0.0.0.0", "LLAMA_PORT": "8189"})
        assert endpoint() == "http://127.0.0.1:8189"
        os.environ["LLAMA_HOST"] = "::"
        assert endpoint() == "http://[::1]:8189"
        os.environ["LLAMA_API_KEY"] = "first,second"
        assert api_key() == "first"
        assert env_flag("LLAMA_STARTUP_PRIME", True) is True
        payload = prime_payload()
        assert payload["speculative.n_max"] == 3
        assert payload["max_tokens"] == 16
        assert payload["messages"][0]["content"].count(" warmup") == 2048
        request = lambda *_args, **_kwargs: (200, json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "READY"}}],
            "timings": {"prompt_n": 2050, "draft_n": 3, "draft_n_accepted": 2},
        }).encode("utf-8"))
        run_prime("http://127.0.0.1:8080", "test-key", 30)
    finally:
        request = original_request
        os.environ.clear()
        os.environ.update(old)
    print(f"prime-production.py {VERSION}: self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without network access")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    enabled = env_flag("LLAMA_STARTUP_PRIME", True)
    mode = os.environ.get("LLAMA_MTP_MODE", "off").strip()
    if mode not in {"off", "strict", "checkpoint-diagnostic"}:
        raise ValueError("LLAMA_MTP_MODE must be off, strict, or checkpoint-diagnostic")
    key = api_key()
    base_url = endpoint()
    startup_timeout = positive_number("LLAMA_STARTUP_PRIME_TIMEOUT", 1800)
    request_timeout = positive_number("LLAMA_STARTUP_PRIME_REQUEST_TIMEOUT", 300)
    if mode == "strict" and not enabled:
        raise ValueError("qualified strict MTP production requires LLAMA_STARTUP_PRIME=1")
    if args.check:
        print(
            f"Startup prime configuration valid: mode={mode}, enabled={int(enabled)}, "
            f"endpoint={base_url}, auth={'yes' if key else 'no'}"
        )
        return 0
    if mode != "strict":
        print(f"Strict MTP startup prime skipped: LLAMA_MTP_MODE={mode}", flush=True)
        return 0
    wait_ready(base_url, key, startup_timeout)
    run_prime(base_url, key, request_timeout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
