#!/usr/bin/env python3
"""Token-level OpenAI chat diagnostic for Qwen MTP equivalence.

This runner is intentionally separate from the throughput harness.  It sends a
fixed OpenAI chat request through all 16 combinations of temperature, streaming,
and per-request MTP window while preserving raw HTTP evidence.  Non-streamed
requests ask llama-server for token logprobs, whose entries include the exact
generated token IDs and bytes needed to locate the first greedy divergence.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import pathlib
import re
import ssl
import sys
import tempfile
import time
import urllib.parse
from typing import Any


VERSION = "1.2.0"
DEFAULT_FIXTURE = pathlib.Path(__file__).resolve().parent / "diagnostics" / "mtp-agentic-openwebui.json"
API_KEY_REDACTION = "[REDACTED_API_KEY]"

# LLAMA_TRACE emits the first two forms below.  At debug verbosity the server
# also emits a later "..., new n_tokens = ..." summary for the same verification
# cycle; requiring end-of-line here prevents that duplicate from inflating the
# acceptance histogram.
DRAFT_ACCEPTANCE_RE = re.compile(
    r"\baccepted\s+(?P<accepted>\d+)\s*/\s*(?P<draft_total>\d+)\s+draft tokens"
    r"(?P<checkpoint>\s+\(restore checkpoint\))?\s*$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def redact_secret_text(value: str, secret: str) -> str:
    return value.replace(secret, API_KEY_REDACTION) if secret else value


def redact_secret_bytes(value: bytes, secret: str) -> bytes:
    if not secret:
        return value
    return value.replace(secret.encode("utf-8"), API_KEY_REDACTION.encode("utf-8"))


def redact_secret_value(value: Any, secret: str) -> Any:
    """Recursively remove an exact API-key value before evidence is persisted."""
    if not secret:
        return value
    if isinstance(value, str):
        return redact_secret_text(value, secret)
    if isinstance(value, bytes):
        return redact_secret_bytes(value, secret)
    if isinstance(value, list):
        return [redact_secret_value(item, secret) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secret_value(item, secret) for item in value)
    if isinstance(value, dict):
        return {
            redact_secret_text(key, secret) if isinstance(key, str) else key:
                redact_secret_value(item, secret)
            for key, item in value.items()
        }
    return value


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_fixture(path: pathlib.Path) -> tuple[str, str, dict[str, Any], dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("fixture must be a JSON object")
    if "request" in value:
        request = value.get("request")
        name = str(value.get("name", path.stem))
        description = str(value.get("description", ""))
        validation = value.get("validation", {})
    else:
        request = value
        name = path.stem
        description = ""
        validation = {}
    if not isinstance(request, dict):
        raise ValueError("fixture.request must be a JSON object")
    if not isinstance(request.get("messages"), list) or not request["messages"]:
        raise ValueError("fixture request requires a non-empty messages array")
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise ValueError("fixture request requires a model string")
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("MTP agentic diagnostic fixture requires a non-empty tools array")
    kwargs = request.get("chat_template_kwargs")
    if not isinstance(kwargs, dict) or kwargs.get("enable_thinking") is not True:
        raise ValueError("fixture must set chat_template_kwargs.enable_thinking=true")
    if not isinstance(validation, dict):
        raise ValueError("fixture.validation must be an object")
    return name, description, copy.deepcopy(request), copy.deepcopy(validation), sha256_bytes(raw)


def resolve_api_key(args: argparse.Namespace) -> tuple[str, str]:
    if args.api_key_file:
        lines = [line.strip() for line in pathlib.Path(args.api_key_file).read_text(encoding="utf-8").splitlines()]
        keys = [line for line in lines if line and not line.startswith("#")]
        if not keys:
            raise ValueError(f"API-key file has no usable key: {args.api_key_file}")
        return keys[0], "file"
    if args.api_key_env:
        value = os.environ.get(args.api_key_env, "")
        if value:
            return value, f"environment:{args.api_key_env}"
    return "", "none"


def http_request(
    method: str,
    url: str,
    body: bytes | None,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no host: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            host, port, timeout=timeout_s, context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    headers = {
        "Accept": "text/event-stream, application/json",
        "Accept-Encoding": "identity",
        "User-Agent": f"qwen-mtp-diag/{VERSION}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    started = time.monotonic()
    try:
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return {
            "status": response.status,
            "reason": response.reason,
            "headers": {key.lower(): value for key, value in response.getheaders()},
            "body": raw,
            "wall_ms": elapsed_ms,
        }
    finally:
        connection.close()


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def erase_slot(base_url: str, slot: int, api_key: str, timeout_s: float) -> dict[str, Any]:
    result = http_request(
        "POST", join_url(base_url, f"slots/{slot}?action=erase"), None, api_key,
        min(30.0, timeout_s),
    )
    text = result["body"].decode("utf-8", errors="replace")
    try:
        decoded = json.loads(text) if text else None
    except json.JSONDecodeError:
        decoded = {"raw": text}
    if result["status"] != 200:
        safe_decoded = redact_secret_value(decoded, api_key)
        raise RuntimeError(f"slot {slot} erase returned HTTP {result['status']}: {safe_decoded}")
    return {
        "status": result["status"],
        "response": redact_secret_value(decoded, api_key),
        "wall_ms": round(result["wall_ms"], 3),
    }


def parse_sse(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="strict")
    # SSE permits CRLF and multiple data lines per event.  llama-server emits
    # one compact JSON data line, but accepting the full grammar makes captured
    # evidence robust to proxies.
    blocks = re.split(r"\r?\n\r?\n", text)
    events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    done = False
    for block_index, block in enumerate(blocks):
        if not block:
            continue
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line == "data":
                data_lines.append("")
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            done = True
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"event {block_index}: {exc}: {payload[:200]!r}")
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            parse_errors.append(f"event {block_index} was not an object")
    return {"events": events, "done": done, "parse_errors": parse_errors}


def normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(value):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        normalized.append({
            "index": int(call.get("index", index)) if isinstance(call.get("index", index), int) else index,
            "type": str(call.get("type", "function")),
            # Tool IDs may be synthesized by a serializer and are deliberately
            # excluded from transport-equivalence comparisons.
            "function": {
                "name": str(function.get("name", "")),
                "arguments": str(function.get("arguments", "")),
            },
        })
    normalized.sort(key=lambda item: item["index"])
    return normalized


def canonical_message(value: Any) -> dict[str, Any]:
    message = value if isinstance(value, dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    return {
        "role": str(message.get("role", "assistant")),
        "reasoning_content": reasoning if isinstance(reasoning, str) else "",
        "content": content if isinstance(content, str) else "",
        "tool_calls": normalize_tool_calls(message.get("tool_calls")),
    }


def validate_response_contract(
    message: dict[str, Any], finish_reason: Any, rules: dict[str, Any],
) -> dict[str, Any]:
    content = str(message.get("content", ""))
    reasoning = str(message.get("reasoning_content", ""))
    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    failures: list[str] = []
    min_content = rules.get("min_content_chars")
    if isinstance(min_content, int) and len(content) < min_content:
        failures.append(f"content has {len(content)} characters; minimum is {min_content}")
    min_reasoning = rules.get("min_reasoning_chars")
    if isinstance(min_reasoning, int) and len(reasoning) < min_reasoning:
        failures.append(f"reasoning has {len(reasoning)} characters; minimum is {min_reasoning}")
    required = rules.get("required_content_substrings", [])
    if isinstance(required, list):
        folded = content.casefold()
        for item in required:
            if isinstance(item, str) and item.casefold() not in folded:
                failures.append(f"content is missing required substring {item!r}")
    allowed_finish = rules.get("allowed_finish_reasons")
    if isinstance(allowed_finish, list) and finish_reason not in allowed_finish:
        failures.append(f"finish reason {finish_reason!r} is not one of {allowed_finish!r}")
    max_tool_calls = rules.get("max_tool_calls")
    if isinstance(max_tool_calls, int) and len(tool_calls) > max_tool_calls:
        failures.append(f"response made {len(tool_calls)} tool calls; maximum is {max_tool_calls}")
    return {
        "passed": not failures,
        "failures": failures,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "tool_call_count": len(tool_calls),
    }


def reconstruct_sse(parsed: dict[str, Any]) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reasons: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    token_trace: list[dict[str, Any]] = []
    for event in parsed["events"]:
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if isinstance(event.get("timings"), dict):
            timings = event["timings"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                finish_reasons.append(finish)
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            raw_reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if isinstance(raw_reasoning, str):
                reasoning.append(raw_reasoning)
            for call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", len(tool_calls))
                if not isinstance(index, int):
                    index = len(tool_calls)
                current = tool_calls.setdefault(index, {
                    "index": index, "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if isinstance(call.get("type"), str):
                    current["type"] = call["type"]
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                if isinstance(function.get("name"), str):
                    current["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    current["function"]["arguments"] += function["arguments"]
            logprobs = choice.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                token_trace.extend(normalize_token_trace(logprobs["content"]))
    message = canonical_message({
        "role": "assistant",
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
    })
    return {
        "message": message,
        "finish_reason": finish_reasons[-1] if finish_reasons else None,
        "finish_reason_count": len(finish_reasons),
        "usage": usage,
        "timings": timings,
        "token_trace": token_trace,
    }


def normalize_token_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    trace: list[dict[str, Any]] = []
    for position, token in enumerate(value):
        if not isinstance(token, dict):
            continue
        top = token.get("top_logprobs", token.get("top_probs", []))
        compact_top: list[dict[str, Any]] = []
        if isinstance(top, list):
            for candidate in top:
                if not isinstance(candidate, dict):
                    continue
                compact_top.append({
                    key: candidate[key]
                    for key in ("id", "token", "bytes", "logprob", "prob")
                    if key in candidate
                })
        trace.append({
            "position": position,
            **{
                key: token[key]
                for key in ("id", "token", "bytes", "logprob", "prob")
                if key in token
            },
            "top": compact_top,
        })
    return trace


def parse_nonstream(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError("chat completion response was not an object")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("chat completion response has no choice")
    choice = choices[0]
    logprobs = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
    return {
        "response": value,
        "message": canonical_message(choice.get("message")),
        "finish_reason": choice.get("finish_reason"),
        "finish_reason_count": 1 if choice.get("finish_reason") is not None else 0,
        "usage": value.get("usage") if isinstance(value.get("usage"), dict) else {},
        "timings": value.get("timings") if isinstance(value.get("timings"), dict) else {},
        "token_trace": normalize_token_trace(logprobs.get("content")),
    }


def message_hash(message: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(message))


def token_trace_hash(trace: list[dict[str, Any]]) -> str:
    tokens = [item.get("id") for item in trace]
    return sha256_bytes(canonical_json_bytes(tokens))


def first_token_divergence(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> dict[str, Any] | None:
    common = min(len(baseline), len(candidate))
    for index in range(common):
        left = baseline[index].get("id")
        right = candidate[index].get("id")
        if left != right:
            return {
                "kind": "token_mismatch",
                "position": index,
                "baseline": baseline[index],
                "candidate": candidate[index],
                "prefix_ids": [item.get("id") for item in baseline[max(0, index - 16):index]],
            }
    if len(baseline) != len(candidate):
        index = common
        return {
            "kind": "candidate_ended" if len(candidate) < len(baseline) else "baseline_ended",
            "position": index,
            "baseline": baseline[index] if index < len(baseline) else None,
            "candidate": candidate[index] if index < len(candidate) else None,
            "prefix_ids": [item.get("id") for item in baseline[max(0, index - 16):index]],
        }
    return None


def token_evidence_error(trace: list[dict[str, Any]]) -> str | None:
    """Return the first missing/malformed exact-token evidence field."""
    def valid_bytes(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
            for item in value
        )

    def valid_score(value: dict[str, Any]) -> bool:
        for key in ("logprob", "prob"):
            score = value.get(key)
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            ):
                return True
        return False

    for position, token in enumerate(trace):
        if not isinstance(token.get("id"), int) or isinstance(token.get("id"), bool):
            return f"token {position} has no integer id"
        if not valid_bytes(token.get("bytes")):
            return f"token {position} has no valid byte sequence"
        if not valid_score(token):
            return f"token {position} has no finite logprob/prob"
        top = token.get("top")
        if not isinstance(top, list) or not top:
            return f"token {position} has no top-candidate evidence"
        for candidate_index, candidate in enumerate(top):
            if not isinstance(candidate, dict):
                return f"token {position} top candidate {candidate_index} is not an object"
            if not isinstance(candidate.get("id"), int) or isinstance(candidate.get("id"), bool):
                return f"token {position} top candidate {candidate_index} has no integer id"
            if not valid_bytes(candidate.get("bytes")):
                return f"token {position} top candidate {candidate_index} has no valid byte sequence"
            if not valid_score(candidate):
                return f"token {position} top candidate {candidate_index} has no finite logprob/prob"
    return None


def timing_cache_n(timings: dict[str, Any]) -> int | None:
    value = timings.get("cache_n")
    return int(value) if isinstance(value, (int, float)) else None


def usage_cached_n(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def server_log_mark(path: pathlib.Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return 0


def server_log_since(path: pathlib.Path | None, offset: int | None) -> str:
    if path is None or offset is None:
        return ""
    try:
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(offset if size >= offset else 0)
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"# server log capture failed: {exc}\n"


def parse_draft_acceptance_events(log_text: str) -> list[dict[str, Any]]:
    """Parse canonical request-scoped LLAMA_TRACE draft-acceptance events."""
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        match = DRAFT_ACCEPTANCE_RE.search(line)
        if match is None:
            continue
        accepted = int(match.group("accepted"))
        draft_total = int(match.group("draft_total"))
        events.append({
            "line_number": line_number,
            "accepted": accepted,
            "draft_total": draft_total,
            "partial": 0 <= accepted < draft_total,
            "rollback_distance": draft_total - accepted,
            "restore_checkpoint": match.group("checkpoint") is not None,
            "valid": draft_total > 0 and 0 <= accepted <= draft_total,
        })
    return events


def n3_partial_acceptance_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether n_max=3 exercised every bounded rollback distance."""
    required_accepted = (0, 1, 2)
    counts = {accepted: 0 for accepted in required_accepted}
    cases = {accepted: set() for accepted in required_accepted}
    n3_case_ids: set[str] = set()
    n3_cases_with_events: set[str] = set()
    total_trace_events = 0

    for row in rows:
        if row.get("status") != "ok" or int(row.get("n_max", -1)) != 3:
            continue
        case_id = str(row.get("case_id", "unknown"))
        n3_case_ids.add(case_id)
        events = row.get("draft_acceptance_events", [])
        if not isinstance(events, list):
            continue
        valid_events = [event for event in events if isinstance(event, dict) and event.get("valid") is True]
        if valid_events:
            n3_cases_with_events.add(case_id)
        total_trace_events += len(valid_events)
        for event in valid_events:
            accepted = event.get("accepted")
            if event.get("draft_total") == 3 and accepted in counts and event.get("partial") is True:
                counts[accepted] += 1
                cases[accepted].add(case_id)

    observed_accepted = [accepted for accepted in required_accepted if counts[accepted] > 0]
    missing_accepted = [accepted for accepted in required_accepted if counts[accepted] == 0]
    return {
        "required_draft_total": 3,
        "required_partial_accepted_lengths": list(required_accepted),
        "required_rollback_distances": [3, 2, 1],
        "observed_partial_accepted_lengths": observed_accepted,
        "observed_rollback_distances": [3 - accepted for accepted in observed_accepted],
        "missing_partial_accepted_lengths": missing_accepted,
        "missing_rollback_distances": [3 - accepted for accepted in missing_accepted],
        "event_counts_by_accepted_length": {
            str(accepted): counts[accepted] for accepted in required_accepted
        },
        "case_ids_by_accepted_length": {
            str(accepted): sorted(cases[accepted]) for accepted in required_accepted
        },
        "successful_n_max_3_cases": len(n3_case_ids),
        "n_max_3_cases_with_trace_events": len(n3_cases_with_events),
        "valid_n_max_3_trace_events": total_trace_events,
        "passed": not missing_accepted,
    }


def condition_matrix(repeats: int) -> list[dict[str, Any]]:
    return [
        {
            "temperature": temperature,
            "n_max": n_max,
            "stream": stream,
            "repeat": repeat,
        }
        for repeat in range(1, repeats + 1)
        for temperature in (0.0, 1.0)
        for n_max in (0, 1, 2, 3)
        for stream in (False, True)
    ]


def build_payload(
    base: dict[str, Any], condition: dict[str, Any], seed: int, slot: int,
    max_tokens: int | None, top_logprobs: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload["temperature"] = condition["temperature"]
    payload["seed"] = seed
    payload["stream"] = condition["stream"]
    payload["cache_prompt"] = False
    payload["id_slot"] = slot
    payload["speculative.n_max"] = condition["n_max"]
    payload.pop("reasoning_format", None)
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs["enable_thinking"] = True
    payload["chat_template_kwargs"] = kwargs
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if condition["stream"]:
        options = payload.get("stream_options")
        if not isinstance(options, dict):
            options = {}
        options["include_usage"] = True
        payload["stream_options"] = options
        # llama-server explicitly rejects logprobs with tools + streaming.
        # Preserve the production tool path and obtain exact token IDs from the
        # paired non-stream request instead of mutating the tool schema.
        payload.pop("logprobs", None)
        payload.pop("top_logprobs", None)
    else:
        payload.pop("stream_options", None)
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    return payload


def case_id(server_label: str, condition: dict[str, Any]) -> str:
    temp = str(condition["temperature"]).replace(".", "p")
    mode = "stream" if condition["stream"] else "nonstream"
    return safe_name(
        f"{server_label}-r{condition['repeat']:02d}-t{temp}-n{condition['n_max']}-{mode}"
    )


def run_case(
    run_dir: pathlib.Path,
    base_url: str,
    base_request: dict[str, Any],
    validation: dict[str, Any],
    condition: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    server_log: pathlib.Path | None,
) -> dict[str, Any]:
    identifier = case_id(args.server_label, condition)
    row: dict[str, Any] = {
        "schema": 1,
        "case_id": identifier,
        "server_label": args.server_label,
        "temperature": condition["temperature"],
        "n_max": condition["n_max"],
        "stream": condition["stream"],
        "repeat": condition["repeat"],
        "seed": args.seed,
        "ts": utc_now(),
    }
    erase = erase_slot(base_url, args.slot, api_key, args.timeout)
    row["slot_erase"] = erase
    mark = server_log_mark(server_log)
    payload = build_payload(
        base_request, condition, args.seed, args.slot, args.max_tokens, args.top_logprobs,
    )
    payload = redact_secret_value(payload, api_key)
    request_bytes = canonical_json_bytes(payload)
    request_rel = pathlib.Path("requests") / f"{identifier}.json"
    request_path = run_dir / request_rel
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(request_bytes + b"\n")
    row.update({
        "request_file": str(request_rel),
        "request_sha256": sha256_bytes(request_bytes),
    })
    raw_rel = pathlib.Path("raw-sse" if condition["stream"] else "raw-http") / (
        f"{identifier}.sse" if condition["stream"] else f"{identifier}.json"
    )
    headers_rel = pathlib.Path("http-headers") / f"{identifier}.json"
    parsed_rel = pathlib.Path("parsed") / f"{identifier}.json"
    token_rel = pathlib.Path("tokens") / f"{identifier}.json"
    log_rel = pathlib.Path("request-logs") / f"{identifier}.log"
    try:
        response = http_request(
            "POST", join_url(base_url, "v1/chat/completions"), request_bytes, api_key,
            args.timeout,
        )
        safe_response_body = redact_secret_bytes(response["body"], api_key)
        safe_response_headers = redact_secret_value(response["headers"], api_key)
        (run_dir / raw_rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / raw_rel).write_bytes(safe_response_body)
        atomic_json(run_dir / headers_rel, {
            "status": response["status"],
            "reason": redact_secret_text(str(response["reason"]), api_key),
            "headers": safe_response_headers,
            "wall_ms": round(response["wall_ms"], 3),
        })
        row.update({
            "http_status": response["status"],
            "http_wall_ms": round(response["wall_ms"], 3),
            "raw_response_file": str(raw_rel),
            "http_headers_file": str(headers_rel),
            "raw_response_sha256": sha256_bytes(safe_response_body),
        })
        if response["status"] != 200:
            raise RuntimeError(
                f"chat completion returned HTTP {response['status']}: "
                f"{safe_response_body[:500].decode('utf-8', errors='replace')}"
            )
        if condition["stream"]:
            sse = parse_sse(response["body"])
            parsed = reconstruct_sse(sse)
            parsed["sse"] = {
                "done": sse["done"], "parse_errors": sse["parse_errors"],
                "event_count": len(sse["events"]),
            }
            sse_events_rel = pathlib.Path("sse-events") / f"{identifier}.json"
            atomic_json(run_dir / sse_events_rel, redact_secret_value(sse["events"], api_key))
            row["sse_events_file"] = str(sse_events_rel)
            row["sse_done"] = sse["done"]
            row["sse_parse_errors"] = sse["parse_errors"]
            if not sse["done"] or sse["parse_errors"]:
                raise RuntimeError(
                    f"invalid SSE termination: done={sse['done']} errors={sse['parse_errors']}"
                )
        else:
            parsed = parse_nonstream(response["body"])
        parsed = redact_secret_value(parsed, api_key)
        atomic_json(run_dir / parsed_rel, {
            key: value for key, value in parsed.items() if key not in {"response", "token_trace"}
        })
        atomic_json(run_dir / token_rel, parsed["token_trace"])
        row.update({
            "parsed_file": str(parsed_rel),
            "token_file": str(token_rel),
            "token_count": len(parsed["token_trace"]),
            "token_trace_sha256": token_trace_hash(parsed["token_trace"]),
            "canonical_message_sha256": message_hash(parsed["message"]),
            "reasoning_chars": len(parsed["message"]["reasoning_content"]),
            "content_chars": len(parsed["message"]["content"]),
            "tool_call_count": len(parsed["message"]["tool_calls"]),
            "finish_reason": parsed["finish_reason"],
            "finish_reason_count": parsed["finish_reason_count"],
            "usage": parsed["usage"],
            "timings": parsed["timings"],
        })
        row["response_contract"] = validate_response_contract(
            parsed["message"], parsed["finish_reason"], validation,
        )
        cache_n = timing_cache_n(parsed["timings"])
        cached_tokens = usage_cached_n(parsed["usage"])
        row["fresh_slot"] = {
            "cache_n": cache_n,
            "usage_cached_tokens": cached_tokens,
            "passed": cache_n == 0 and cached_tokens in {None, 0},
        }
        if cache_n != 0:
            raise RuntimeError(
                f"fresh-slot invariant failed: timings.cache_n={cache_n!r}, expected 0"
            )
        if cached_tokens not in {None, 0}:
            raise RuntimeError(
                f"fresh-slot invariant failed: usage cached_tokens={cached_tokens!r}"
            )
        if not condition["stream"] and not parsed["token_trace"]:
            raise RuntimeError("non-stream response omitted exact token logprobs")
        if not condition["stream"]:
            if evidence_error := token_evidence_error(parsed["token_trace"]):
                raise RuntimeError(f"non-stream token evidence is incomplete: {evidence_error}")
            completion_tokens = parsed["usage"].get("completion_tokens")
            if (
                isinstance(completion_tokens, (int, float))
                and int(completion_tokens) != len(parsed["token_trace"])
            ):
                raise RuntimeError(
                    "token trace length does not match usage.completion_tokens: "
                    f"{len(parsed['token_trace'])} != {int(completion_tokens)}"
                )
        if parsed["finish_reason_count"] != 1:
            raise RuntimeError(
                f"expected exactly one finish reason, got {parsed['finish_reason_count']}"
            )
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = redact_secret_text(repr(exc), api_key)
    finally:
        time.sleep(0.05)
        log_text = redact_secret_text(server_log_since(server_log, mark), api_key)
        (run_dir / log_rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / log_rel).write_text(log_text, encoding="utf-8")
        row["server_log_file"] = str(log_rel) if server_log is not None else None
        row["server_log_bytes"] = len(log_text.encode("utf-8")) if server_log is not None else 0
        row["server_log_sha256"] = sha256_bytes(log_text.encode("utf-8")) if server_log is not None else None
        row["draft_acceptance_events"] = parse_draft_acceptance_events(log_text)
        row["draft_acceptance_event_count"] = len(row["draft_acceptance_events"])
    return row


def load_trace(run_dir: pathlib.Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = row.get("token_file")
    if not isinstance(path_value, str):
        return []
    path = run_dir / path_value
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def row_map(rows: list[dict[str, Any]]) -> dict[tuple[float, int, bool, int], dict[str, Any]]:
    return {
        (float(row["temperature"]), int(row["n_max"]), bool(row["stream"]), int(row["repeat"])): row
        for row in rows
    }


def classify_run(
    run_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    reference_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    indexed = row_map(rows)
    comparisons: list[dict[str, Any]] = []
    request_errors = [row["case_id"] for row in rows if row.get("status") != "ok"]
    response_contract_failures = [
        {
            "case_id": row["case_id"],
            "failures": row.get("response_contract", {}).get("failures", []),
        }
        for row in rows
        if row.get("status") == "ok" and row.get("response_contract", {}).get("passed") is False
    ]
    greedy_divergences: list[dict[str, Any]] = []
    stochastic_cross_n: list[dict[str, Any]] = []
    transport_divergences: list[dict[str, Any]] = []
    repeat_divergences: list[dict[str, Any]] = []
    mtp_not_exercised: list[str] = []
    fresh_failures = [
        row["case_id"] for row in rows
        if row.get("status") == "ok" and not row.get("fresh_slot", {}).get("passed")
    ]
    server_log_capture = bool(rows) and all(
        isinstance(row.get("server_log_file"), str)
        and isinstance(row.get("server_log_bytes"), int)
        and row["server_log_bytes"] > 0
        for row in rows
    )
    rollback_coverage = n3_partial_acceptance_coverage(rows)
    repeats = sorted({int(row["repeat"]) for row in rows})

    for temperature in (0.0, 1.0):
        for repeat in repeats:
            baseline = indexed.get((temperature, 0, False, repeat))
            if not baseline or baseline.get("status") != "ok":
                continue
            baseline_trace = load_trace(run_dir, baseline)
            for n_max in (1, 2, 3):
                candidate = indexed.get((temperature, n_max, False, repeat))
                if not candidate or candidate.get("status") != "ok":
                    continue
                divergence = first_token_divergence(
                    baseline_trace, load_trace(run_dir, candidate),
                )
                comparison = {
                    "kind": "nmax_vs_n0",
                    "temperature": temperature,
                    "repeat": repeat,
                    "n_max": n_max,
                    "baseline_case": baseline["case_id"],
                    "candidate_case": candidate["case_id"],
                    "exact_match": divergence is None,
                    "first_divergence": divergence,
                }
                comparisons.append(comparison)
                if divergence is not None:
                    (greedy_divergences if temperature == 0.0 else stochastic_cross_n).append(comparison)
                draft_n = candidate.get("timings", {}).get("draft_n")
                if not isinstance(draft_n, (int, float)) or draft_n <= 0:
                    mtp_not_exercised.append(candidate["case_id"])

            for n_max in (0, 1, 2, 3):
                nonstream = indexed.get((temperature, n_max, False, repeat))
                streamed = indexed.get((temperature, n_max, True, repeat))
                if not nonstream or not streamed:
                    continue
                if nonstream.get("status") != "ok" or streamed.get("status") != "ok":
                    continue
                exact = (
                    nonstream.get("canonical_message_sha256") == streamed.get("canonical_message_sha256")
                    and nonstream.get("finish_reason") == streamed.get("finish_reason")
                )
                comparison = {
                    "kind": "stream_vs_nonstream",
                    "temperature": temperature,
                    "repeat": repeat,
                    "n_max": n_max,
                    "nonstream_case": nonstream["case_id"],
                    "stream_case": streamed["case_id"],
                    "exact_canonical_match": exact,
                    "nonstream_message_sha256": nonstream.get("canonical_message_sha256"),
                    "stream_message_sha256": streamed.get("canonical_message_sha256"),
                    "nonstream_finish_reason": nonstream.get("finish_reason"),
                    "stream_finish_reason": streamed.get("finish_reason"),
                }
                comparisons.append(comparison)
                if not exact:
                    transport_divergences.append(comparison)

    if len(repeats) > 1:
        first_repeat = repeats[0]
        for temperature in (0.0, 1.0):
            for n_max in (0, 1, 2, 3):
                baseline = indexed.get((temperature, n_max, False, first_repeat))
                if not baseline or baseline.get("status") != "ok":
                    continue
                baseline_trace = load_trace(run_dir, baseline)
                for repeat in repeats[1:]:
                    candidate = indexed.get((temperature, n_max, False, repeat))
                    if not candidate or candidate.get("status") != "ok":
                        continue
                    divergence = first_token_divergence(
                        baseline_trace, load_trace(run_dir, candidate),
                    )
                    comparison = {
                        "kind": "fixed_seed_repeatability",
                        "temperature": temperature,
                        "n_max": n_max,
                        "baseline_repeat": first_repeat,
                        "candidate_repeat": repeat,
                        "exact_match": divergence is None,
                        "first_divergence": divergence,
                    }
                    comparisons.append(comparison)
                    if divergence is not None:
                        repeat_divergences.append(comparison)

    reference_comparisons: list[dict[str, Any]] = []
    reference_failures: list[dict[str, Any]] = []
    reference_expected = 0
    if reference_dir is not None:
        reference_rows = read_jsonl(reference_dir / "results.jsonl")
        reference = row_map(reference_rows)
        for temperature in (0.0, 1.0):
            for repeat in repeats:
                reference_expected += 1
                left = reference.get((temperature, 0, False, repeat))
                right = indexed.get((temperature, 0, False, repeat))
                cell = {"temperature": temperature, "repeat": repeat}
                if left is None:
                    reference_failures.append({**cell, "reason": "reference n=0 non-stream cell is missing"})
                    continue
                if left.get("status") != "ok":
                    reference_failures.append({
                        **cell,
                        "reference_case": left.get("case_id"),
                        "reason": f"reference cell status is {left.get('status')!r}",
                    })
                    continue
                if right is None:
                    reference_failures.append({**cell, "reason": "candidate n=0 non-stream cell is missing"})
                    continue
                if right.get("status") != "ok":
                    reference_failures.append({
                        **cell,
                        "candidate_case": right.get("case_id"),
                        "reason": f"candidate cell status is {right.get('status')!r}",
                    })
                    continue
                left_request_sha = left.get("request_sha256")
                right_request_sha = right.get("request_sha256")
                if (
                    not isinstance(left_request_sha, str)
                    or not isinstance(right_request_sha, str)
                    or len(left_request_sha) != 64
                    or len(right_request_sha) != 64
                ):
                    reference_failures.append({
                        **cell,
                        "reference_case": left.get("case_id"),
                        "candidate_case": right.get("case_id"),
                        "reason": "reference or candidate request SHA-256 is missing",
                    })
                    continue
                if left_request_sha != right_request_sha:
                    reference_failures.append({
                        **cell,
                        "reference_case": left.get("case_id"),
                        "candidate_case": right.get("case_id"),
                        "reference_request_sha256": left_request_sha,
                        "candidate_request_sha256": right_request_sha,
                        "reason": "reference and candidate request bodies differ",
                    })
                    continue
                left_trace = load_trace(reference_dir, left)
                right_trace = load_trace(run_dir, right)
                if not left_trace:
                    reference_failures.append({
                        **cell,
                        "reference_case": left.get("case_id"),
                        "reason": "reference token trace is missing or empty",
                    })
                    continue
                if evidence_error := token_evidence_error(left_trace):
                    reference_failures.append({
                        **cell,
                        "reference_case": left.get("case_id"),
                        "reason": f"reference token evidence is incomplete: {evidence_error}",
                    })
                    continue
                if not right_trace:
                    reference_failures.append({
                        **cell,
                        "candidate_case": right.get("case_id"),
                        "reason": "candidate token trace is missing or empty",
                    })
                    continue
                divergence = first_token_divergence(left_trace, right_trace)
                reference_comparisons.append({
                    **cell,
                    "reference_case": left["case_id"],
                    "candidate_case": right["case_id"],
                    "exact_match": divergence is None,
                    "first_divergence": divergence,
                })

    causes: list[str] = []
    greedy_n1 = [item for item in greedy_divergences if item["n_max"] == 1]
    greedy_multi = [item for item in greedy_divergences if item["n_max"] in {2, 3}]
    if greedy_n1:
        causes.append(
            "n=1 diverged under greedy decoding: investigate single-row target/draft state, hidden-state handoff, or sampler acceptance before multi-row rollback"
        )
    elif greedy_multi:
        causes.append(
            "n=1 matched but n=2/3 diverged under greedy decoding: investigate multi-row verification, recurrent rollback, and 256-cell attention boundaries"
        )
    if transport_divergences and not greedy_divergences:
        causes.append(
            "streaming differed while non-stream greedy traces matched: investigate chat parser, reasoning/tool-call deltas, or SSE serialization"
        )
    if reference_failures:
        causes.append(
            "target-only n=0 reference coverage is incomplete or invalid; every "
            "temperature/repeat cell must use the identical request body, succeed, "
            "and retain a complete nonempty exact token trace"
        )
    if reference_comparisons and any(not item["exact_match"] for item in reference_comparisons):
        causes.append(
            "sidecar-loaded n=0 differed from the no-sidecar reference: investigate per-request bypass and target hidden-state export"
        )
    if stochastic_cross_n and not greedy_divergences:
        causes.append(
            "only temperature=1 differed across n_max; this localizes the stochastic/RNG path but is not by itself proof of distributional error"
        )
    if mtp_not_exercised:
        causes.append("one or more n>0 arms generated no draft tokens, so they cannot qualify an MTP fix")
    if response_contract_failures:
        causes.append(
            "one or more responses violated the fixture's completion contract; inspect whether failures are MTP-only or also affect n=0"
        )
    if not server_log_capture:
        causes.append("request-scoped server logs were not captured; rerun with --server-log for a complete diagnostic")
    elif not rollback_coverage["passed"]:
        missing = ", ".join(
            f"{accepted}/3" for accepted in rollback_coverage["missing_partial_accepted_lengths"]
        )
        causes.append(
            "n_max=3 did not exercise every bounded rollback path; missing partial "
            f"acceptance events: {missing}"
        )
    if not causes and not request_errors:
        causes.append("no external divergence was detected in this matrix")

    return {
        "schema": 1,
        "created_at": utc_now(),
        "request_errors": request_errors,
        "response_contract_failures": response_contract_failures,
        "fresh_slot_failures": fresh_failures,
        "mtp_not_exercised": sorted(set(mtp_not_exercised)),
        "greedy_divergences": greedy_divergences,
        "stochastic_cross_n_differences": stochastic_cross_n,
        "transport_divergences": transport_divergences,
        "repeatability_divergences": repeat_divergences,
        "reference_n0_comparisons": reference_comparisons,
        "reference_n0_failures": reference_failures,
        "reference_n0_expected_comparisons": reference_expected,
        "n_max_3_partial_acceptance_coverage": rollback_coverage,
        "comparisons": comparisons,
        "classification": causes,
        "acceptance": {
            "matrix_complete": len(rows) == 16 * len(repeats) and not request_errors,
            "fresh_slot_pass": not fresh_failures,
            "mtp_exercised": not mtp_not_exercised,
            "greedy_equivalence_pass": not greedy_divergences,
            "stream_transport_pass": not transport_divergences,
            "fixed_seed_repeatability_pass": not repeat_divergences,
            "response_contract_pass": not response_contract_failures,
            "server_log_capture": server_log_capture,
            "n_max_3_partial_acceptance_coverage_pass": rollback_coverage["passed"],
            "reference_n0_pass": (
                not reference_failures
                and len(reference_comparisons) == reference_expected
                and all(item["exact_match"] for item in reference_comparisons)
                if reference_dir is not None else None
            ),
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary(run_dir: pathlib.Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    comparisons = {
        (item["temperature"], item["n_max"], item["repeat"]): item
        for item in report["comparisons"] if item["kind"] == "nmax_vs_n0"
    }
    transport = {
        (item["temperature"], item["n_max"], item["repeat"]): item
        for item in report["comparisons"] if item["kind"] == "stream_vs_nonstream"
    }
    indexed = row_map(rows)
    lines = [
        f"# MTP diagnostic: {run_dir.name}\n\n",
        "## Acceptance\n\n",
    ]
    for key, value in report["acceptance"].items():
        lines.append(f"- `{key}`: `{value}`\n")
    coverage = report["n_max_3_partial_acceptance_coverage"]
    lines.extend([
        "\n## n_max=3 partial-acceptance coverage\n\n",
        "`--require-pass` requires at least one canonical LLAMA_TRACE event for each partial "
        "acceptance length 0/3, 1/3, and 2/3. These exercise rollback distances 3, 2, and 1.\n\n",
        "| Accepted/drafted | Rollback distance | Events | Cases | Coverage |\n",
        "| ----------------: | ----------------: | -----: | :---- | :------- |\n",
    ])
    for accepted in coverage["required_partial_accepted_lengths"]:
        key = str(accepted)
        count = coverage["event_counts_by_accepted_length"][key]
        case_ids = coverage["case_ids_by_accepted_length"][key]
        lines.append(
            f"| {accepted}/3 | {3 - accepted} | {count} | "
            f"{', '.join(f'`{case_id}`' for case_id in case_ids) or 'none'} | "
            f"{'PASS' if count else 'MISSING'} |\n"
        )
    lines.extend([
        f"\nOverall rollback coverage: **{'PASS' if coverage['passed'] else 'FAIL'}**. "
        f"Parsed {coverage['valid_n_max_3_trace_events']} valid trace events across "
        f"{coverage['n_max_3_cases_with_trace_events']}/"
        f"{coverage['successful_n_max_3_cases']} successful n_max=3 cases.\n",
        "\n## Classification\n\n",
    ])
    for item in report["classification"]:
        lines.append(f"- {item}\n")
    lines.extend([
        "\n## Condition matrix\n\n",
        "| Temp | n_max | Repeat | Non-stream | Contract | Tokens | Draft accepted/generated | First difference vs n=0 | Stream parity |\n",
        "| ---: | ----: | -----: | :--------- | :------- | -----: | :----------------------- | :---------------------- | :------------ |\n",
    ])
    repeats = sorted({int(row["repeat"]) for row in rows})
    for repeat in repeats:
        for temperature in (0.0, 1.0):
            for n_max in (0, 1, 2, 3):
                row = indexed.get((temperature, n_max, False, repeat), {})
                cmp = comparisons.get((temperature, n_max, repeat))
                stream_cmp = transport.get((temperature, n_max, repeat))
                timing = row.get("timings", {})
                difference = "baseline"
                if cmp:
                    div = cmp.get("first_divergence")
                    difference = "none" if div is None else f"{div['kind']} @ {div['position']}"
                parity = "" if stream_cmp is None else ("PASS" if stream_cmp["exact_canonical_match"] else "FAIL")
                contract = row.get("response_contract", {}).get("passed")
                contract_text = "" if contract is None else ("PASS" if contract else "FAIL")
                lines.append(
                    f"| {temperature:g} | {n_max} | {repeat} | {row.get('status', 'missing')} | "
                    f"{contract_text} | {row.get('token_count', '')} | {fmt(timing.get('draft_n_accepted'))}/{fmt(timing.get('draft_n'))} | "
                    f"{difference} | {parity} |\n"
                )
    lines.extend([
        "\nNon-stream token traces come directly from llama-server's OpenAI logprobs and contain generated token IDs and bytes. "
        "With tools present, llama-server rejects logprobs plus streaming; streamed arms therefore gate the canonical reconstruction of reasoning, content, tool calls, and finish reason against their paired non-stream response.\n",
    ])
    (run_dir / "summary.md").write_text("".join(lines), encoding="utf-8")


def command_run(args: argparse.Namespace) -> int:
    fixture_path = pathlib.Path(args.fixture).resolve()
    fixture_name, fixture_description, base_request, validation, fixture_sha = load_fixture(fixture_path)
    api_key, auth_mode = resolve_api_key(args)
    args.server_label = redact_secret_text(args.server_label, api_key)
    server_log = pathlib.Path(args.server_log).resolve() if args.server_log else None
    if server_log is not None and not server_log.is_file():
        raise ValueError(f"--server-log is not a readable file: {server_log}")
    reference_dir = pathlib.Path(args.reference_run).resolve() if args.reference_run else None
    if reference_dir is not None and not (reference_dir / "results.jsonl").exists():
        raise ValueError(f"reference run has no results.jsonl: {reference_dir}")
    if reference_dir is not None:
        reference_manifest_path = reference_dir / "manifest.json"
        if not reference_manifest_path.exists():
            raise ValueError(f"reference run has no manifest.json: {reference_dir}")
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        expected = {
            "harness_version": VERSION,
            "fixture_sha256": fixture_sha,
            "seed": args.seed,
            "max_tokens_override": args.max_tokens,
            "top_logprobs": args.top_logprobs,
            "repeats": args.repeats,
        }
        mismatched = {
            key: {"reference": reference_manifest.get(key), "candidate": value}
            for key, value in expected.items()
            if reference_manifest.get(key) != value
        }
        if mismatched:
            raise ValueError(
                "reference run is not request-identical to this run: "
                + json.dumps(mismatched, sort_keys=True)
            )
    output_root = pathlib.Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{stamp()}-mtp-diagnostic-{safe_name(args.server_label)}"
    run_dir.mkdir()
    conditions = condition_matrix(args.repeats)
    startup_log = (
        redact_secret_bytes(server_log.read_bytes(), api_key)
        if server_log is not None else b""
    )
    manifest = {
        "schema": 1,
        "harness_version": VERSION,
        "created_at": utc_now(),
        "server_label": args.server_label,
        "base_url": args.url,
        "fixture": str(fixture_path),
        "fixture_name": fixture_name,
        "fixture_description": fixture_description,
        "fixture_sha256": fixture_sha,
        "response_validation": validation,
        "auth_mode": auth_mode,
        "api_key_recorded": False,
        "server_log": str(server_log) if server_log else None,
        "server_log_capture": server_log is not None,
        "server_startup_log_bytes": len(startup_log),
        "server_startup_log_sha256": sha256_bytes(startup_log) if server_log else None,
        "reference_run": str(reference_dir) if reference_dir else None,
        "seed": args.seed,
        "slot": args.slot,
        "max_tokens_override": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "repeats": args.repeats,
        "conditions": conditions,
        "controlled_fields": [
            "temperature", "seed", "stream", "cache_prompt", "id_slot",
            "speculative.n_max", "stream_options.include_usage", "logprobs", "top_logprobs",
        ],
    }
    manifest = redact_secret_value(manifest, api_key)
    atomic_json(run_dir / "manifest.json", manifest)
    if server_log is not None:
        (run_dir / "server-startup.log").write_bytes(startup_log)
    if server_log is None:
        (run_dir / "request-logs").mkdir()
        (run_dir / "request-logs" / "UNAVAILABLE.txt").write_text(
            "No --server-log path was supplied. HTTP evidence is complete, but request-scoped server logs were not captured.\n",
            encoding="utf-8",
        )
    rows: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index:02d}/{len(conditions):02d}] temp={condition['temperature']:g} "
            f"n_max={condition['n_max']} stream={str(condition['stream']).lower()} "
            f"repeat={condition['repeat']}",
            flush=True,
        )
        row = run_case(
            run_dir, args.url, base_request, validation, condition, args, api_key, server_log,
        )
        row = redact_secret_value(row, api_key)
        append_jsonl(run_dir / "results.jsonl", row)
        rows.append(row)
        if row.get("status") != "ok":
            print(f"  ERROR: {row.get('error')}", flush=True)
    report = redact_secret_value(classify_run(run_dir, rows, reference_dir), api_key)
    atomic_json(run_dir / "comparisons.json", report)
    write_summary(run_dir, rows, report)
    print(f"Results: {run_dir}")
    print(f"Greedy equivalence: {'PASS' if report['acceptance']['greedy_equivalence_pass'] else 'FAIL'}")
    print(f"Stream transport: {'PASS' if report['acceptance']['stream_transport_pass'] else 'FAIL'}")
    coverage = report["n_max_3_partial_acceptance_coverage"]
    observed = ", ".join(
        f"{accepted}/3" for accepted in coverage["observed_partial_accepted_lengths"]
    ) or "none"
    missing = ", ".join(
        f"{accepted}/3" for accepted in coverage["missing_partial_accepted_lengths"]
    ) or "none"
    print(
        f"n_max=3 rollback coverage: {'PASS' if coverage['passed'] else 'FAIL'} "
        f"(observed: {observed}; missing: {missing})"
    )
    for item in report["classification"]:
        print(f"- {item}")
    passed = all(
        report["acceptance"][key]
        for key in (
            "matrix_complete", "fresh_slot_pass", "mtp_exercised",
            "greedy_equivalence_pass", "stream_transport_pass", "fixed_seed_repeatability_pass",
            "response_contract_pass", "server_log_capture",
            "n_max_3_partial_acceptance_coverage_pass",
        )
    )
    if reference_dir is not None:
        passed = passed and report["acceptance"]["reference_n0_pass"] is True
    return 0 if passed or not args.require_pass else 1


def self_test() -> None:
    test_secret = "sk-test-never-persist"
    assert redact_secret_text(f"Bearer {test_secret}", test_secret) == f"Bearer {API_KEY_REDACTION}"
    assert test_secret.encode() not in redact_secret_bytes(
        f"argv --api-key {test_secret}".encode(), test_secret,
    )
    redacted_value = redact_secret_value({
        "header": f"Bearer {test_secret}",
        "nested": [test_secret, {test_secret: f"prefix-{test_secret}-suffix"}],
    }, test_secret)
    assert test_secret.encode() not in canonical_json_bytes(redacted_value)
    assert canonical_json_bytes(redacted_value).count(API_KEY_REDACTION.encode()) == 4

    fixture_name, _description, fixture_request, fixture_validation, fixture_sha = load_fixture(DEFAULT_FIXTURE)
    assert fixture_name == "openwebui-agentic-ideation-synthetic"
    assert fixture_request["chat_template_kwargs"]["enable_thinking"] is True
    assert fixture_validation["required_content_substrings"] == ["benchmark"]
    assert len(fixture_sha) == 64

    matrix = condition_matrix(1)
    assert len(matrix) == 16
    assert {
        (item["temperature"], item["n_max"], item["stream"])
        for item in matrix
    } == {
        (temperature, n_max, stream)
        for temperature in (0.0, 1.0)
        for n_max in (0, 1, 2, 3)
        for stream in (False, True)
    }

    trace_log = (
        "1.00 I slot accepted  0/ 3 draft tokens\n"
        "1.01 I slot accepted 1/3 draft tokens (restore checkpoint)\n"
        "1.02 I slot accepted 2/3 draft tokens\n"
        "1.03 D slot accepted 2/3 draft tokens, new n_tokens = 42\n"
        "1.04 I slot accepted 3/3 draft tokens\n"
    )
    trace_events = parse_draft_acceptance_events(trace_log)
    assert [(event["accepted"], event["draft_total"]) for event in trace_events] == [
        (0, 3), (1, 3), (2, 3), (3, 3),
    ]
    assert trace_events[1]["restore_checkpoint"] is True
    assert trace_events[3]["partial"] is False
    coverage_row = {
        "case_id": "coverage", "status": "ok", "n_max": 3,
        "draft_acceptance_events": trace_events,
    }
    coverage = n3_partial_acceptance_coverage([coverage_row])
    assert coverage["passed"] is True
    assert coverage["observed_partial_accepted_lengths"] == [0, 1, 2]
    assert coverage["observed_rollback_distances"] == [3, 2, 1]
    missing_one = copy.deepcopy(coverage_row)
    missing_one["draft_acceptance_events"] = [
        event for event in trace_events if event["accepted"] != 1
    ]
    missing_coverage = n3_partial_acceptance_coverage([missing_one])
    assert missing_coverage["passed"] is False
    assert missing_coverage["missing_partial_accepted_lengths"] == [1]
    assert missing_coverage["missing_rollback_distances"] == [2]

    sse_raw = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning_content":"think "},"finish_reason":null}]}\r\n\r\n'
        b'data: {"choices":[{"delta":{"reasoning_content":"carefully"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"look","arguments":"{\\\"q\\\":"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":0}},"timings":{"cache_n":0}}\n\n'
        b'data: [DONE]\n\n'
    )
    parsed_sse = parse_sse(sse_raw)
    assert parsed_sse["done"] and not parsed_sse["parse_errors"]
    reconstructed = reconstruct_sse(parsed_sse)
    assert reconstructed["message"]["reasoning_content"] == "think carefully"
    assert reconstructed["message"]["content"] == "answer"
    assert reconstructed["message"]["tool_calls"][0]["function"] == {
        "name": "look", "arguments": '{"q":1}',
    }
    assert reconstructed["finish_reason"] == "tool_calls"
    assert reconstructed["finish_reason_count"] == 1
    assert timing_cache_n(reconstructed["timings"]) == 0
    contract = validate_response_contract(
        reconstructed["message"], reconstructed["finish_reason"], {
            "min_content_chars": 10,
            "required_content_substrings": ["benchmark"],
            "allowed_finish_reasons": ["stop"],
            "max_tool_calls": 0,
        },
    )
    assert contract["passed"] is False and len(contract["failures"]) == 4

    nonstream_value = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "reasoning_content": "r", "content": "a"},
            "logprobs": {"content": [
                {"id": 10, "token": "r", "bytes": [114], "logprob": -0.1,
                 "top_logprobs": [{"id": 10, "token": "r", "bytes": [114], "logprob": -0.1}]},
                {"id": 11, "token": "a", "bytes": [97], "logprob": -0.2,
                 "top_logprobs": [{"id": 11, "token": "a", "bytes": [97], "logprob": -0.2}]},
            ]},
        }],
        "usage": {"completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}},
        "timings": {"cache_n": 0, "draft_n": 1, "draft_n_accepted": 1},
    }
    parsed_nonstream = parse_nonstream(canonical_json_bytes(nonstream_value))
    assert [item["id"] for item in parsed_nonstream["token_trace"]] == [10, 11]
    assert token_evidence_error(parsed_nonstream["token_trace"]) is None
    incomplete_evidence = copy.deepcopy(parsed_nonstream["token_trace"])
    del incomplete_evidence[1]["bytes"]
    assert "byte sequence" in str(token_evidence_error(incomplete_evidence))
    assert parsed_nonstream["message"]["reasoning_content"] == "r"
    assert parsed_nonstream["message"]["content"] == "a"
    assert first_token_divergence(
        parsed_nonstream["token_trace"], copy.deepcopy(parsed_nonstream["token_trace"]),
    ) is None
    changed = copy.deepcopy(parsed_nonstream["token_trace"])
    changed[1]["id"] = 12
    divergence = first_token_divergence(parsed_nonstream["token_trace"], changed)
    assert divergence and divergence["kind"] == "token_mismatch" and divergence["position"] == 1
    shorter = first_token_divergence(parsed_nonstream["token_trace"], changed[:1])
    assert shorter and shorter["kind"] == "candidate_ended" and shorter["position"] == 1

    fixture_base = {
        "model": "test", "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_format": "deepseek",
    }
    nonstream_payload = build_payload(fixture_base, matrix[0], 1234, 0, 32, 5)
    assert "reasoning_format" not in nonstream_payload
    assert nonstream_payload["logprobs"] is True and nonstream_payload["top_logprobs"] == 5
    stream_condition = next(item for item in matrix if item["stream"])
    stream_payload = build_payload(fixture_base, stream_condition, 1234, 0, 32, 5)
    assert "logprobs" not in stream_payload and "top_logprobs" not in stream_payload
    assert stream_payload["stream_options"]["include_usage"] is True

    with tempfile.TemporaryDirectory() as raw_tmp:
        root = pathlib.Path(raw_tmp)
        fake_rows: list[dict[str, Any]] = []
        for condition in matrix:
            identifier = case_id("fake", condition)
            trace = copy.deepcopy(parsed_nonstream["token_trace"])
            if condition["temperature"] == 0.0 and condition["n_max"] == 1:
                trace[1]["id"] = 12
            token_file = pathlib.Path("tokens") / f"{identifier}.json"
            atomic_json(root / token_file, trace)
            fake_rows.append({
                "case_id": identifier,
                "status": "ok",
                "temperature": condition["temperature"],
                "n_max": condition["n_max"],
                "stream": condition["stream"],
                "repeat": condition["repeat"],
                "request_sha256": sha256_bytes(canonical_json_bytes({
                    "temperature": condition["temperature"],
                    "n_max": condition["n_max"],
                    "stream": condition["stream"],
                    "repeat": condition["repeat"],
                })),
                "token_file": str(token_file),
                "canonical_message_sha256": "same",
                "finish_reason": "stop",
                "fresh_slot": {"passed": True},
                "draft_acceptance_events": trace_events if condition["n_max"] == 3 else [],
                "timings": {
                    "draft_n": 0 if condition["n_max"] == 0 else condition["n_max"],
                    "draft_n_accepted": 0 if condition["n_max"] == 0 else condition["n_max"],
                },
            })
        fake_report = classify_run(root, fake_rows)
        assert fake_report["acceptance"]["matrix_complete"] is True
        assert fake_report["acceptance"]["n_max_3_partial_acceptance_coverage_pass"] is True
        assert fake_report["acceptance"]["greedy_equivalence_pass"] is False
        assert fake_report["greedy_divergences"][0]["n_max"] == 1
        assert any("single-row" in item for item in fake_report["classification"])

        reference_root = root / "reference"
        reference_root.mkdir()
        reference_rows = [
            copy.deepcopy(row) for row in fake_rows
            if row["n_max"] == 0 and row["stream"] is False
        ]
        for row in reference_rows:
            row["case_id"] = "reference-" + row["case_id"]
            trace = load_trace(root, row)
            atomic_json(reference_root / row["token_file"], trace)
            append_jsonl(reference_root / "results.jsonl", row)
        reference_report = classify_run(root, fake_rows, reference_root)
        assert reference_report["reference_n0_expected_comparisons"] == 2
        assert len(reference_report["reference_n0_comparisons"]) == 2
        assert not reference_report["reference_n0_failures"]
        assert reference_report["acceptance"]["reference_n0_pass"] is True

        incomplete_reference_root = root / "incomplete-reference"
        incomplete_reference_root.mkdir()
        row = reference_rows[0]
        atomic_json(incomplete_reference_root / row["token_file"], load_trace(reference_root, row))
        append_jsonl(incomplete_reference_root / "results.jsonl", row)
        incomplete_reference_report = classify_run(root, fake_rows, incomplete_reference_root)
        assert incomplete_reference_report["reference_n0_expected_comparisons"] == 2
        assert len(incomplete_reference_report["reference_n0_failures"]) == 1
        assert incomplete_reference_report["acceptance"]["reference_n0_pass"] is False

        mismatched_reference_root = root / "mismatched-reference"
        mismatched_reference_root.mkdir()
        for index, source_row in enumerate(reference_rows):
            row = copy.deepcopy(source_row)
            if index == 0:
                row["request_sha256"] = "0" * 64
            atomic_json(mismatched_reference_root / row["token_file"], load_trace(reference_root, source_row))
            append_jsonl(mismatched_reference_root / "results.jsonl", row)
        mismatched_reference_report = classify_run(root, fake_rows, mismatched_reference_root)
        assert len(mismatched_reference_report["reference_n0_failures"]) == 1
        assert "request bodies differ" in mismatched_reference_report["reference_n0_failures"][0]["reason"]
        assert mismatched_reference_report["acceptance"]["reference_n0_pass"] is False

        fixture_path = root / "fixture.json"
        atomic_json(fixture_path, {"name": "test", "request": fixture_base})
        name, _description, loaded, validation, digest = load_fixture(fixture_path)
        assert name == "test" and loaded["model"] == "test" and len(digest) == 64
        assert validation == {}

    print(f"qwen_mtp_diag.py {VERSION}: self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the 16-condition MTP chat diagnostic")
    run.add_argument("--url", default="http://127.0.0.1:8080")
    run.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    run.add_argument("--output-root", default="results")
    run.add_argument("--server-label", default="current")
    run.add_argument("--server-log", help="llama-server stdout/stderr log to slice per request")
    run.add_argument("--reference-run", help="optional no-sidecar diagnostic run for n=0 comparison")
    run.add_argument("--api-key-env", default="QWEN_TEST_API_KEY")
    run.add_argument("--api-key-file")
    run.add_argument("--slot", type=int, default=0)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--top-logprobs", type=int, default=5)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--require-pass", action="store_true")
    sub.add_parser("self-test", help="run offline parser and comparison tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.require_pass and not args.reference_run:
            raise ValueError(
                "--require-pass requires --reference-run from a request-identical target-only server"
            )
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        if args.slot < 0:
            raise ValueError("--slot must be non-negative")
        if args.max_tokens is not None and args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        if not 1 <= args.top_logprobs <= 20:
            raise ValueError("--top-logprobs must be from 1 through 20")
        return command_run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
