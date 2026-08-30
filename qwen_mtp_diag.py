#!/usr/bin/env python3
"""Token-level OpenAI chat diagnostic for Qwen MTP equivalence.

This runner is intentionally separate from the throughput harness.  It sends a
fixed OpenAI chat request through either the full 16-cell matrix or a focused
isolation profile while preserving raw HTTP evidence.  Non-streamed
requests ask llama-server for token logprobs, whose entries include the exact
generated token IDs and bytes needed to locate the first greedy divergence.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import http.client
import io
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


VERSION = "1.7.1"
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
    if rules.get("require_nonempty_content_on_stop") is True and finish_reason == "stop":
        if not content.strip():
            failures.append("stop response has no non-whitespace content")
    if rules.get("require_valid_tool_calls_on_tool_finish") is True and finish_reason == "tool_calls":
        if not tool_calls:
            failures.append("tool_calls finish has no tool call")
        for index, call in enumerate(tool_calls):
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name.strip():
                failures.append(f"tool call {index} has no function name")
            if not isinstance(arguments, str):
                failures.append(f"tool call {index} arguments are not a JSON string")
                continue
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                failures.append(f"tool call {index} arguments are not valid JSON")
                continue
            if not isinstance(parsed_arguments, dict):
                failures.append(f"tool call {index} arguments are not a JSON object")
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
        left_id = baseline[index].get("id")
        right_id = candidate[index].get("id")
        left_bytes = baseline[index].get("bytes")
        right_bytes = candidate[index].get("bytes")
        if left_id != right_id or left_bytes != right_bytes:
            return {
                "kind": "token_mismatch",
                "position": index,
                "identity_fields_differ": [
                    field for field, differs in (
                        ("id", left_id != right_id),
                        ("bytes", left_bytes != right_bytes),
                    ) if differs
                ],
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


def first_scored_distribution_divergence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float = 1e-6,
) -> dict[str, Any] | None:
    """Find score drift where both traces retained comparable evidence."""
    def selected_top_score(item: dict[str, Any]) -> tuple[str, float] | None:
        top = item.get("top")
        if not isinstance(top, list) or not top:
            return None
        for top_candidate in top:
            if (
                isinstance(top_candidate, dict)
                and top_candidate.get("id") == item.get("id")
                and top_candidate.get("bytes") == item.get("bytes")
            ):
                for field in ("logprob", "prob"):
                    score = top_candidate.get(field)
                    if (
                        isinstance(score, (int, float))
                        and not isinstance(score, bool)
                        and math.isfinite(float(score))
                    ):
                        return field, float(score)
        return None

    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if left.get("id") != right.get("id") or left.get("bytes") != right.get("bytes"):
            break
        left_evidence = selected_top_score(left)
        right_evidence = selected_top_score(right)
        if left_evidence is None or right_evidence is None or left_evidence[0] != right_evidence[0]:
            continue
        score_field = left_evidence[0]
        left_score = left_evidence[1]
        right_score = right_evidence[1]
        delta = abs(left_score - right_score)
        if delta > tolerance:
            return {
                "position": index,
                "token_id": left.get("id"),
                "token": left.get("token"),
                "baseline_logprob": left_score,
                "candidate_logprob": right_score,
                "score_field": score_field,
                "absolute_delta": delta,
                "tolerance": tolerance,
            }
    return None


def token_identity_error(trace: list[dict[str, Any]]) -> str | None:
    """Return the first field error that prevents exact token comparison."""
    def valid_bytes(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
            for item in value
        )

    for position, token in enumerate(trace):
        if not isinstance(token.get("id"), int) or isinstance(token.get("id"), bool):
            return f"token {position} has no integer id"
        if not valid_bytes(token.get("bytes")):
            return f"token {position} has no valid byte sequence"
    return None


def token_distribution_error(trace: list[dict[str, Any]]) -> str | None:
    """Return the first missing score/top-candidate field.

    llama-server currently emits exact IDs and bytes for accepted MTP draft
    tokens while leaving their top-logprob arrays empty.  That is incomplete
    distribution evidence, but it must never hide an observable token-sequence
    divergence by turning the whole response into an incomparable request.
    """
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


def token_evidence_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    identity_error = token_identity_error(trace)
    distribution_error = token_distribution_error(trace)
    return {
        "identity_complete": identity_error is None,
        "identity_error": identity_error,
        "distribution_complete": distribution_error is None,
        "distribution_error": distribution_error,
        "tokens_missing_top_candidates": sum(
            1 for token in trace if not isinstance(token.get("top"), list) or not token["top"]
        ),
    }


def legacy_distribution_only_error(row: dict[str, Any]) -> bool:
    """Recognize v1.2 rows rejected only for empty accepted-draft top lists."""
    error = row.get("error")
    token_count = row.get("token_count")
    completion_tokens = (
        row.get("usage", {}).get("completion_tokens")
        if isinstance(row.get("usage"), dict) else None
    )
    canonical_hash = row.get("canonical_message_sha256")
    return (
        row.get("status") == "error"
        and isinstance(error, str)
        and "non-stream token evidence is incomplete" in error
        and "has no top-candidate evidence" in error
        and row.get("http_status") == 200
        and isinstance(token_count, int)
        and token_count > 0
        and isinstance(completion_tokens, (int, float))
        and int(completion_tokens) == token_count
        and row.get("finish_reason_count") == 1
        and isinstance(canonical_hash, str)
        and len(canonical_hash) == 64
        and row.get("fresh_slot", {}).get("passed") is True
    )


def row_transport_complete(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok" or legacy_distribution_only_error(row)


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


def outputs_before_first_rejection(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservatively bound generated output before any rejected draft.

    A fully accepted speculative step emits every accepted draft token plus one
    additional target token.  Counting only those complete steps deliberately
    excludes the initial ordinary token and therefore remains a lower bound.
    If a zero-based divergence position is below this bound, that divergence
    necessarily predates both bounded rollback and checkpoint restore.
    """
    full_events = 0
    minimum_outputs = 0
    first_rejection_event: int | None = None
    first_rejection_restores_checkpoint = False
    evidence_gap = False

    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("valid") is not True:
            evidence_gap = True
            break
        if event.get("partial") is True:
            first_rejection_event = event_index
            first_rejection_restores_checkpoint = event.get("restore_checkpoint") is True
            break
        accepted = event.get("accepted")
        draft_total = event.get("draft_total")
        if (
            isinstance(accepted, int)
            and isinstance(draft_total, int)
            and draft_total > 0
            and accepted == draft_total
        ):
            full_events += 1
            minimum_outputs += draft_total + 1
        else:
            evidence_gap = True
            break

    all_captured_events_fully_accepted = bool(events) and (
        not evidence_gap
        and first_rejection_event is None
        and full_events == len(events)
    )

    return {
        "first_rejection_event_index": first_rejection_event,
        "first_rejection_restores_checkpoint": first_rejection_restores_checkpoint,
        "full_acceptance_events_before_first_rejection": full_events,
        "minimum_outputs_before_first_rejection": minimum_outputs,
        "first_rejection_observed": first_rejection_event is not None,
        "all_captured_events_fully_accepted": all_captured_events_fully_accepted,
        "evidence_gap_before_first_rejection": evidence_gap,
    }


def debug_cap_contract_truncation(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Identify response-contract failures caused solely by a debug token cap."""
    max_tokens = manifest.get("max_tokens_override")
    expected_failure = "finish reason 'length' is not one of ['stop', 'tool_calls']"
    affected = {item.get("case_id") for item in failures}
    complete_rows = [row for row in rows if row_transport_complete(row)]
    solely_length = bool(failures) and all(
        item.get("failures") == [expected_failure] for item in failures
    )
    all_complete_rows_affected = bool(complete_rows) and affected == {
        row.get("case_id") for row in complete_rows
    }

    def generated_count(row: dict[str, Any]) -> Any:
        token_count = row.get("token_count")
        if isinstance(token_count, int) and token_count > 0:
            return token_count
        usage = row.get("usage")
        return usage.get("completion_tokens") if isinstance(usage, dict) else None

    all_hit_cap = (
        isinstance(max_tokens, int)
        and max_tokens > 0
        and all(generated_count(row) == max_tokens for row in complete_rows)
    )
    detected = solely_length and all_complete_rows_affected and all_hit_cap
    return {
        "detected": detected,
        "max_tokens_override": max_tokens,
        "affected_cases": len(failures) if detected else 0,
        "note": (
            "Every complete arm reached the explicit diagnostic token cap; "
            "response completeness was not assessed."
            if detected else None
        ),
    }


def n3_partial_acceptance_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether n_max=3 exercised every bounded rollback distance."""
    required_accepted = (0, 1, 2)
    counts = {accepted: 0 for accepted in required_accepted}
    cases = {accepted: set() for accepted in required_accepted}
    n3_case_ids: set[str] = set()
    n3_cases_with_events: set[str] = set()
    total_trace_events = 0

    for row in rows:
        if not row_transport_complete(row) or int(row.get("n_max", -1)) != 3:
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


def condition_matrix(repeats: int, profile: str = "full") -> list[dict[str, Any]]:
    if profile == "full":
        temperatures = (0.0, 1.0)
        n_max_values = (0, 1, 2, 3)
        stream_values = (False, True)
    elif profile == "greedy-n01":
        temperatures = (0.0,)
        n_max_values = (0, 1)
        stream_values = (False,)
    elif profile == "production-n03":
        temperatures = (1.0,)
        n_max_values = (0, 3)
        stream_values = (False, True)
    else:
        raise ValueError(f"unknown diagnostic matrix profile: {profile}")

    return [
        {
            "temperature": temperature,
            "n_max": n_max,
            "stream": stream,
            "repeat": repeat,
        }
        for repeat in range(1, repeats + 1)
        for temperature in temperatures
        for n_max in n_max_values
        for stream in stream_values
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
            row["token_evidence"] = token_evidence_summary(parsed["token_trace"])
            if identity_error := row["token_evidence"]["identity_error"]:
                raise RuntimeError(
                    f"non-stream token identity evidence is incomplete: {identity_error}"
                )
            completion_tokens = parsed["usage"].get("completion_tokens")
            if (
                isinstance(completion_tokens, (int, float))
                and int(completion_tokens) != len(parsed["token_trace"])
            ):
                raise RuntimeError(
                    "token trace length does not match usage.completion_tokens: "
                    f"{len(parsed['token_trace'])} != {int(completion_tokens)}"
                )
        else:
            row["token_evidence"] = {
                "identity_complete": None,
                "identity_error": None,
                "distribution_complete": None,
                "distribution_error": None,
                "tokens_missing_top_candidates": None,
            }
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
    observed_repeats = sorted({int(row["repeat"]) for row in rows})
    manifest_conditions: list[dict[str, Any]] = []
    manifest_value: dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_value = loaded_manifest if isinstance(loaded_manifest, dict) else {}
        raw_conditions = manifest_value.get("conditions") if isinstance(manifest_value, dict) else None
        if isinstance(raw_conditions, list):
            manifest_conditions = [item for item in raw_conditions if isinstance(item, dict)]
    if manifest_conditions:
        expected_keys = {
            (
                float(item["temperature"]), int(item["n_max"]),
                bool(item["stream"]), int(item["repeat"]),
            )
            for item in manifest_conditions
        }
        repeats = sorted({item[3] for item in expected_keys})
    else:
        repeats = observed_repeats
        expected_keys = {
            (temperature, n_max, stream, repeat)
            for temperature in (0.0, 1.0)
            for n_max in (0, 1, 2, 3)
            for stream in (False, True)
            for repeat in repeats
        }
    actual_keys = [
        (float(row["temperature"]), int(row["n_max"]), bool(row["stream"]), int(row["repeat"]))
        for row in rows
    ]
    actual_key_set = set(actual_keys)
    duplicate_cells = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
    missing_cells = sorted(expected_keys - actual_key_set)
    unexpected_cells = sorted(actual_key_set - expected_keys)
    legacy_distribution_rows = [
        row["case_id"] for row in rows if legacy_distribution_only_error(row)
    ]
    request_errors = [
        row["case_id"] for row in rows if not row_transport_complete(row)
    ]
    token_identity_failures: list[dict[str, Any]] = []
    token_distribution_failures: list[dict[str, Any]] = []
    comparison_ready: set[str] = set()
    for row in rows:
        if not row_transport_complete(row):
            continue
        if not bool(row.get("stream")):
            trace = load_trace(run_dir, row)
            if not trace:
                token_identity_failures.append({
                    "case_id": row["case_id"],
                    "error": "non-stream token trace is missing or empty",
                })
                continue
            evidence = token_evidence_summary(trace)
            if evidence["identity_error"] is not None:
                token_identity_failures.append({
                    "case_id": row["case_id"],
                    "error": evidence["identity_error"],
                })
                continue
            if evidence["distribution_error"] is not None:
                token_distribution_failures.append({
                    "case_id": row["case_id"],
                    "error": evidence["distribution_error"],
                    "tokens_missing_top_candidates": evidence["tokens_missing_top_candidates"],
                    "token_count": len(trace),
                })
        comparison_ready.add(row["case_id"])

    def ready(row: dict[str, Any] | None) -> bool:
        return row is not None and row.get("case_id") in comparison_ready

    response_contract_failures = [
        {
            "case_id": row["case_id"],
            "failures": row.get("response_contract", {}).get("failures", []),
        }
        for row in rows
        if ready(row) and row.get("response_contract", {}).get("passed") is False
    ]
    debug_cap_truncation = debug_cap_contract_truncation(
        rows, response_contract_failures, manifest_value,
    )
    greedy_divergences: list[dict[str, Any]] = []
    stochastic_cross_n: list[dict[str, Any]] = []
    transport_divergences: list[dict[str, Any]] = []
    repeat_divergences: list[dict[str, Any]] = []
    mtp_not_exercised: list[str] = []
    fresh_failures = [
        row["case_id"] for row in rows
        if ready(row) and not row.get("fresh_slot", {}).get("passed")
    ]
    server_log_capture = bool(rows) and all(
        isinstance(row.get("server_log_file"), str)
        and isinstance(row.get("server_log_bytes"), int)
        and row["server_log_bytes"] > 0
        for row in rows
    )
    rollback_coverage = n3_partial_acceptance_coverage(rows)

    for row in rows:
        if not ready(row) or int(row.get("n_max", 0)) <= 0:
            continue
        draft_n = row.get("timings", {}).get("draft_n")
        if not isinstance(draft_n, (int, float)) or draft_n <= 0:
            mtp_not_exercised.append(row["case_id"])

    for temperature in (0.0, 1.0):
        for repeat in repeats:
            baseline = indexed.get((temperature, 0, False, repeat))
            if not ready(baseline):
                continue
            baseline_trace = load_trace(run_dir, baseline)
            for n_max in (1, 2, 3):
                candidate = indexed.get((temperature, n_max, False, repeat))
                if not ready(candidate):
                    continue
                divergence = first_token_divergence(
                    baseline_trace, load_trace(run_dir, candidate),
                )
                distribution_divergence = first_scored_distribution_divergence(
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
                    "first_scored_distribution_divergence": distribution_divergence,
                }
                if divergence is not None:
                    pre_rejection = outputs_before_first_rejection(
                        candidate.get("draft_acceptance_events", []),
                    )
                    position = divergence.get("position")
                    evidence_covers_divergence = bool(
                        isinstance(position, int)
                        and position < pre_rejection["minimum_outputs_before_first_rejection"]
                    )
                    pre_rejection["divergence_before_first_rejection"] = bool(
                        evidence_covers_divergence
                        and (
                            pre_rejection["first_rejection_observed"]
                            or pre_rejection["all_captured_events_fully_accepted"]
                        )
                    )
                    pre_rejection["proof_kind"] = (
                        "before_observed_rejection"
                        if pre_rejection["divergence_before_first_rejection"]
                        and pre_rejection["first_rejection_observed"]
                        else "captured_without_rejection"
                        if pre_rejection["divergence_before_first_rejection"]
                        else None
                    )
                    comparison["pre_rejection_analysis"] = pre_rejection
                comparisons.append(comparison)
                if divergence is not None:
                    (greedy_divergences if temperature == 0.0 else stochastic_cross_n).append(comparison)
            for n_max in (0, 1, 2, 3):
                nonstream = indexed.get((temperature, n_max, False, repeat))
                streamed = indexed.get((temperature, n_max, True, repeat))
                if not nonstream or not streamed:
                    continue
                if not ready(nonstream) or not ready(streamed):
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
                if not ready(baseline):
                    continue
                baseline_trace = load_trace(run_dir, baseline)
                for repeat in repeats[1:]:
                    candidate = indexed.get((temperature, n_max, False, repeat))
                    if not ready(candidate):
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
        reference_cells = sorted({
            (temperature, repeat)
            for temperature, n_max, stream, repeat in expected_keys
            if n_max == 0 and stream is False
        })
        for temperature, repeat in reference_cells:
            reference_expected += 1
            left = reference.get((temperature, 0, False, repeat))
            right = indexed.get((temperature, 0, False, repeat))
            cell = {"temperature": temperature, "repeat": repeat}
            if left is None:
                reference_failures.append({**cell, "reason": "reference n=0 non-stream cell is missing"})
                continue
            if not row_transport_complete(left):
                reference_failures.append({
                    **cell,
                    "reference_case": left.get("case_id"),
                    "reason": f"reference cell status is {left.get('status')!r}",
                })
                continue
            if right is None:
                reference_failures.append({**cell, "reason": "candidate n=0 non-stream cell is missing"})
                continue
            if not ready(right):
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
            if evidence_error := token_identity_error(left_trace):
                reference_failures.append({
                    **cell,
                    "reference_case": left.get("case_id"),
                    "reason": f"reference token identity evidence is incomplete: {evidence_error}",
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

    greedy_comparisons = [
        item for item in comparisons
        if item["kind"] == "nmax_vs_n0" and item["temperature"] == 0.0
    ]
    transport_comparisons = [
        item for item in comparisons if item["kind"] == "stream_vs_nonstream"
    ]
    repeatability_comparisons = [
        item for item in comparisons if item["kind"] == "fixed_seed_repeatability"
    ]
    baseline_repeatability_comparisons = [
        item for item in repeatability_comparisons
        if item["temperature"] == 0.0 and item["n_max"] == 0
    ]
    baseline_repeatability_divergences = [
        item for item in baseline_repeatability_comparisons if not item["exact_match"]
    ]
    greedy_expected = sum(
        (0.0, 0, False, repeat) in expected_keys
        and (0.0, n_max, False, repeat) in expected_keys
        for repeat in repeats
        for n_max in (1, 2, 3)
    )
    transport_expected = sum(
        (temperature, n_max, False, repeat) in expected_keys
        and (temperature, n_max, True, repeat) in expected_keys
        for temperature in (0.0, 1.0)
        for n_max in (0, 1, 2, 3)
        for repeat in repeats
    )
    repeatability_expected = 0
    if repeats:
        first_repeat = repeats[0]
        repeatability_expected = sum(
            (temperature, n_max, False, first_repeat) in expected_keys
            and (temperature, n_max, False, repeat) in expected_keys
            for temperature in (0.0, 1.0)
            for n_max in (0, 1, 2, 3)
            for repeat in repeats[1:]
        )
    baseline_repeatability_expected = 0
    if repeats:
        first_repeat = repeats[0]
        baseline_repeatability_expected = sum(
            (0.0, 0, False, first_repeat) in expected_keys
            and (0.0, 0, False, repeat) in expected_keys
            for repeat in repeats[1:]
        )
    baseline_repeatability_pass = (
        len(baseline_repeatability_comparisons) == baseline_repeatability_expected
        and not baseline_repeatability_divergences
        if baseline_repeatability_expected > 0 else None
    )
    n3_coverage_required = any(n_max == 3 for _, n_max, _, _ in expected_keys)

    causes: list[str] = []
    greedy_n1 = [item for item in greedy_divergences if item["n_max"] == 1]
    greedy_multi = [item for item in greedy_divergences if item["n_max"] in {2, 3}]
    pre_rejection_greedy = [
        item for item in greedy_divergences
        if item.get("pre_rejection_analysis", {}).get("divergence_before_first_rejection") is True
    ]
    prompt_boundary_score_drift = [
        item for item in greedy_comparisons
        if (item.get("first_scored_distribution_divergence") or {}).get("position") == 0
    ]
    prime_matches_first_baseline = False
    priming_path = run_dir / "priming.json"
    if repeats and priming_path.is_file():
        loaded_priming = json.loads(priming_path.read_text(encoding="utf-8"))
        prime_rows = loaded_priming if isinstance(loaded_priming, list) else [loaded_priming]
        prime_rows = [item for item in prime_rows if isinstance(item, dict)]
        first_baseline = indexed.get((0.0, 0, False, repeats[0]))
        n0_prime_rows = [item for item in prime_rows if item.get("n_max") == 0]
        if n0_prime_rows and ready(first_baseline):
            prime_trace = load_trace(run_dir, n0_prime_rows[-1])
            first_baseline_trace = load_trace(run_dir, first_baseline)
            prime_matches_first_baseline = bool(
                prime_trace and first_baseline_trace
                and first_token_divergence(prime_trace, first_baseline_trace) is None
            )
    if baseline_repeatability_divergences:
        later_cross_n_matches = [
            item for item in greedy_comparisons
            if item.get("repeat") != repeats[0] and item.get("exact_match") is True
        ] if repeats else []
        if prime_matches_first_baseline:
            causes.append(
                "the explicit n=0 prime and first measured n=0 request match exactly, "
                "but n=0 changes only after the first interleaved MTP request; this is a "
                "persistent MTP-induced process-state transition, not a cold-start outlier" + (
                    ", and later n=0/n>0 pairs match in the altered regime"
                    if later_cross_n_matches else ""
                ) + "; inspect MTP prompt output masking, graph reserve/allocation, and "
                "target hidden-state export before rollback"
            )
        else:
            causes.append(
                "the greedy n=0 baseline is not fixed-seed repeatable, so cross-n MTP "
                "equivalence is inconclusive; the first measured request is a cold-start "
                "outlier" + (
                    " while a later n=0/n>0 pair matches exactly"
                    if later_cross_n_matches else ""
                ) + "; prime the request-identical target graph before measuring and "
                "investigate startup warm-up/recurrent-state initialization"
            )
    if prompt_boundary_score_drift and not baseline_repeatability_divergences:
        details = ", ".join(
            f"n={item['n_max']} {item['first_scored_distribution_divergence']['baseline_logprob']:.6f}"
            f"->{item['first_scored_distribution_divergence']['candidate_logprob']:.6f}"
            for item in prompt_boundary_score_drift
        )
        causes.append(
            "target score drift is already visible on output token 0, before draft verification or "
            f"checkpoint restore ({details}); prioritize target hidden-state export/graph numerics and "
            "request-state initialization over rollback"
        )
    if pre_rejection_greedy and not baseline_repeatability_divergences:
        first = pre_rejection_greedy[0]
        pre_rejection = first["pre_rejection_analysis"]
        divergence = first.get("first_divergence") or {}
        timing_clause = (
            "occurred before the first rejected draft"
            if pre_rejection.get("proof_kind") == "before_observed_rejection"
            else "occurred while every captured speculative step was fully accepted; "
                 "no rejected draft occurred in the captured request"
        )
        causes.append(
            f"n={first['n_max']} greedy divergence at token {divergence.get('position')} {timing_clause}: "
            f"{pre_rejection['full_acceptance_events_before_first_rejection']} "
            "full-acceptance events emitted at least "
            f"{pre_rejection['minimum_outputs_before_first_rejection']} tokens first; rejection rollback, "
            "checkpoint restore, PLE rewind, speculative-state restore, and sampler restore cannot be "
            "the primary cause, so inspect accepted-path target state, logical multi-row decode/memory "
            "initialization, verifier logits indexing, and hidden-state handoff"
        )
    elif greedy_n1 and not baseline_repeatability_divergences:
        causes.append(
            "n=1 diverged under greedy decoding: this rules out a fault that requires two or more draft tokens; "
            "the target still verifies a two-row batch, so inspect one-token rejection rollback, target recurrent/PLE state, "
            "hidden-state handoff, batched verification, and verifier acceptance"
        )
    elif greedy_multi and not baseline_repeatability_divergences:
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
    if len(greedy_comparisons) != greedy_expected:
        causes.append(
            "greedy comparison coverage is incomplete: "
            f"observed {len(greedy_comparisons)}/{greedy_expected} required n>0 versus n=0 pairs"
        )
    if transport_expected > 0 and len(transport_comparisons) != transport_expected:
        causes.append(
            "stream/non-stream comparison coverage is incomplete: "
            f"observed {len(transport_comparisons)}/{transport_expected} required pairs"
        )
    if repeatability_expected > 0 and len(repeatability_comparisons) != repeatability_expected:
        causes.append(
            "fixed-seed repeatability coverage is incomplete: "
            f"observed {len(repeatability_comparisons)}/{repeatability_expected} required pairs"
        )
    if repeatability_expected == 0 and len(repeats) < 2:
        causes.append("fixed-seed repeatability was not tested; run at least two repeats")
    if token_distribution_failures:
        causes.append(
            "llama-server omitted full score/top-candidate evidence for accepted MTP tokens; "
            "exact token IDs remain comparable and this warning must not suppress divergences"
        )
    if debug_cap_truncation["detected"]:
        causes.append(
            f"all response-contract failures are expected diagnostic truncations at the explicit "
            f"{debug_cap_truncation['max_tokens_override']}-token cap; token divergence remains valid, "
            "but this short run cannot qualify response completeness"
        )
    elif response_contract_failures:
        causes.append(
            "one or more responses violated the fixture's completion contract; inspect whether failures are MTP-only or also affect n=0"
        )
    if not server_log_capture:
        causes.append("request-scoped server logs were not captured; rerun with --server-log for a complete diagnostic")
    elif n3_coverage_required and not rollback_coverage["passed"]:
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
        "legacy_distribution_rows_recovered": legacy_distribution_rows,
        "token_identity_failures": token_identity_failures,
        "token_distribution_failures": token_distribution_failures,
        "missing_cells": [list(item) for item in missing_cells],
        "unexpected_cells": [list(item) for item in unexpected_cells],
        "duplicate_cells": [list(item) for item in duplicate_cells],
        "response_contract_failures": response_contract_failures,
        "debug_cap_contract_truncation": debug_cap_truncation,
        "fresh_slot_failures": fresh_failures,
        "mtp_not_exercised": sorted(set(mtp_not_exercised)),
        "greedy_divergences": greedy_divergences,
        "pre_rejection_greedy_divergences": pre_rejection_greedy,
        "prompt_boundary_score_drift": prompt_boundary_score_drift,
        "stochastic_cross_n_differences": stochastic_cross_n,
        "transport_divergences": transport_divergences,
        "repeatability_divergences": repeat_divergences,
        "reference_n0_comparisons": reference_comparisons,
        "reference_n0_failures": reference_failures,
        "reference_n0_expected_comparisons": reference_expected,
        "n_max_3_partial_acceptance_coverage": rollback_coverage,
        "comparisons": comparisons,
        "comparison_coverage": {
            "greedy_observed": len(greedy_comparisons),
            "greedy_expected": greedy_expected,
            "stream_transport_observed": len(transport_comparisons),
            "stream_transport_expected": transport_expected,
            "fixed_seed_repeatability_observed": len(repeatability_comparisons),
            "fixed_seed_repeatability_expected": repeatability_expected,
            "baseline_repeatability_observed": len(baseline_repeatability_comparisons),
            "baseline_repeatability_expected": baseline_repeatability_expected,
            "n_max_3_coverage_required": n3_coverage_required,
        },
        "classification": causes,
        "acceptance": {
            "matrix_complete": (
                bool(expected_keys)
                and len(rows) == len(expected_keys)
                and not missing_cells
                and not unexpected_cells
                and not duplicate_cells
                and not request_errors
                and not token_identity_failures
            ),
            "fresh_slot_pass": not fresh_failures,
            "mtp_exercised": not mtp_not_exercised,
            "greedy_equivalence_pass": (
                False if not expected_keys else
                None if greedy_expected == 0 else
                None if baseline_repeatability_pass is False else
                len(greedy_comparisons) == greedy_expected
                and not greedy_divergences
            ),
            "stream_transport_pass": (
                transport_expected > 0
                and len(transport_comparisons) == transport_expected
                and not transport_divergences
            ) if transport_expected > 0 else None,
            "fixed_seed_repeatability_pass": (
                len(repeatability_comparisons) == repeatability_expected and not repeat_divergences
                if repeatability_expected > 0 else None
            ),
            "baseline_repeatability_pass": baseline_repeatability_pass,
            "token_distribution_evidence_pass": not token_distribution_failures,
            "response_contract_pass": not response_contract_failures,
            "server_log_capture": server_log_capture,
            "n_max_3_partial_acceptance_coverage_pass": (
                rollback_coverage["passed"] if n3_coverage_required else None
            ),
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


def write_summary(
    run_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    filename: str = "summary.md",
) -> None:
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
        rendered = "N/A" if value is None else str(value)
        lines.append(f"- `{key}`: `{rendered}`\n")
    coverage = report["n_max_3_partial_acceptance_coverage"]
    n3_required = report["comparison_coverage"].get("n_max_3_coverage_required") is True
    lines.append("\n## n_max=3 partial-acceptance coverage\n\n")
    if n3_required:
        lines.extend([
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
        lines.append(
            f"\nOverall rollback coverage: **{'PASS' if coverage['passed'] else 'FAIL'}**. "
            f"Parsed {coverage['valid_n_max_3_trace_events']} valid trace events across "
            f"{coverage['n_max_3_cases_with_trace_events']}/"
            f"{coverage['successful_n_max_3_cases']} successful n_max=3 cases.\n"
        )
    else:
        lines.append("Not applicable to this reduced matrix profile.\n")
    lines.append("\n## Divergence timing relative to rejection\n\n")
    pre_rejection = report.get("pre_rejection_greedy_divergences", [])
    if pre_rejection:
        for item in pre_rejection:
            divergence = item.get("first_divergence") or {}
            timing = item.get("pre_rejection_analysis") or {}
            timing_text = (
                "before the first rejected draft"
                if timing.get("proof_kind") == "before_observed_rejection"
                else "with no rejected draft in the captured request"
            )
            lines.append(
                f"- `n_max={item.get('n_max')}` diverged at zero-based output token "
                f"{divergence.get('position')}, {timing_text}. "
                f"{timing.get('full_acceptance_events_before_first_rejection')} full-acceptance "
                f"events emitted at least {timing.get('minimum_outputs_before_first_rejection')} "
                "tokens first.\n"
            )
    else:
        lines.append("- No greedy divergence was proven to predate the first rejected draft.\n")
    lines.extend([
        "\n## Classification\n\n",
    ])
    for item in report["classification"]:
        lines.append(f"- {item}\n")
    lines.extend([
        "\n## Condition matrix\n\n",
        "| Temp | n_max | Repeat | Non-stream | Contract | Tokens | Draft accepted/generated | First difference vs n=0 | Before rejection | Stream parity |\n",
        "| ---: | ----: | -----: | :--------- | :------- | -----: | :----------------------- | :---------------------- | :------------- | :------------ |\n",
    ])
    condition_cells = {
        (int(row["repeat"]), float(row["temperature"]), int(row["n_max"]))
        for row in rows if not bool(row.get("stream"))
    }
    condition_cells.update(
        (int(repeat), float(temperature), int(n_max))
        for temperature, n_max, stream, repeat in report.get("missing_cells", [])
        if not bool(stream)
    )
    for repeat, temperature, n_max in sorted(condition_cells):
        row = indexed.get((temperature, n_max, False, repeat), {})
        cmp = comparisons.get((temperature, n_max, repeat))
        stream_cmp = transport.get((temperature, n_max, repeat))
        timing = row.get("timings", {})
        difference = "baseline"
        before_rejection = ""
        if cmp:
            div = cmp.get("first_divergence")
            difference = "none" if div is None else f"{div['kind']} @ {div['position']}"
            if div is not None:
                before_rejection = (
                    "YES" if cmp.get("pre_rejection_analysis", {}).get("divergence_before_first_rejection")
                    else "not proven"
                )
        parity = "" if stream_cmp is None else ("PASS" if stream_cmp["exact_canonical_match"] else "FAIL")
        contract = row.get("response_contract", {}).get("passed")
        contract_text = "" if contract is None else ("PASS" if contract else "FAIL")
        row_status = row.get("status", "missing")
        if row and legacy_distribution_only_error(row):
            row_status = "recovered-v1.2"
        lines.append(
            f"| {temperature:g} | {n_max} | {repeat} | {row_status} | "
            f"{contract_text} | {row.get('token_count', '')} | {fmt(timing.get('draft_n_accepted'))}/{fmt(timing.get('draft_n'))} | "
            f"{difference} | {before_rejection} | {parity} |\n"
        )
    lines.extend([
        "\nNon-stream token traces come directly from llama-server's OpenAI logprobs and contain generated token IDs and bytes. "
        "With tools present, llama-server rejects logprobs plus streaming; streamed arms therefore gate the canonical reconstruction of reasoning, content, tool calls, and finish reason against their paired non-stream response.\n",
    ])
    (run_dir / filename).write_text("".join(lines), encoding="utf-8")


def print_verdict(report: dict[str, Any]) -> None:
    coverage = report["comparison_coverage"]
    greedy_matched = sum(
        item.get("exact_match") is True
        for item in report.get("comparisons", [])
        if item.get("kind") == "nmax_vs_n0" and item.get("temperature") == 0.0
    )
    transport_matched = sum(
        item.get("exact_canonical_match") is True
        for item in report.get("comparisons", [])
        if item.get("kind") == "stream_vs_nonstream"
    )
    greedy_gate = report["acceptance"]["greedy_equivalence_pass"]
    greedy_label = (
        "INCONCLUSIVE"
        if greedy_gate is None and coverage["greedy_expected"] > 0
        else "N/A" if greedy_gate is None
        else "PASS" if greedy_gate else "FAIL"
    )
    print(
        f"Greedy equivalence: {greedy_label} "
        f"({greedy_matched}/{coverage['greedy_expected']} matched; "
        f"{coverage['greedy_observed']}/{coverage['greedy_expected']} evaluated)"
    )
    baseline_gate = report["acceptance"].get("baseline_repeatability_pass")
    baseline_label = "N/A" if baseline_gate is None else "PASS" if baseline_gate else "FAIL"
    print(
        f"Greedy n=0 repeatability: {baseline_label} "
        f"({coverage.get('baseline_repeatability_observed', 0)}/"
        f"{coverage.get('baseline_repeatability_expected', 0)} evaluated)"
    )
    repeat_gate = report["acceptance"].get("fixed_seed_repeatability_pass")
    repeat_label = "N/A" if repeat_gate is None else "PASS" if repeat_gate else "FAIL"
    repeat_matched = sum(
        item.get("exact_match") is True
        for item in report.get("comparisons", [])
        if item.get("kind") == "fixed_seed_repeatability"
    )
    print(
        f"Fixed-seed repeatability: {repeat_label} "
        f"({repeat_matched}/{coverage.get('fixed_seed_repeatability_expected', 0)} matched; "
        f"{coverage.get('fixed_seed_repeatability_observed', 0)}/"
        f"{coverage.get('fixed_seed_repeatability_expected', 0)} evaluated)"
    )
    transport_gate = report["acceptance"]["stream_transport_pass"]
    print(
        f"Stream transport: {'N/A' if transport_gate is None else 'PASS' if transport_gate else 'FAIL'} "
        f"({transport_matched}/{coverage['stream_transport_expected']} matched; "
        f"{coverage['stream_transport_observed']}/{coverage['stream_transport_expected']} evaluated)"
    )
    rollback = report["n_max_3_partial_acceptance_coverage"]
    observed = ", ".join(
        f"{accepted}/3" for accepted in rollback["observed_partial_accepted_lengths"]
    ) or "none"
    missing = ", ".join(
        f"{accepted}/3" for accepted in rollback["missing_partial_accepted_lengths"]
    ) or "none"
    rollback_gate = report["acceptance"]["n_max_3_partial_acceptance_coverage_pass"]
    print(
        f"n_max=3 rollback coverage: "
        f"{'N/A' if rollback_gate is None else 'PASS' if rollback_gate else 'FAIL'} "
        f"(observed: {observed}; missing: {missing})"
    )
    contract_gate = report["acceptance"].get("response_contract_pass")
    contract_failures = len(report.get("response_contract_failures", []))
    print(
        f"Response completeness: "
        f"{'N/A' if contract_gate is None else 'PASS' if contract_gate else 'FAIL'} "
        f"({contract_failures} contract failures)"
    )
    for divergence in report["greedy_divergences"]:
        first = divergence.get("first_divergence") or {}
        baseline = first.get("baseline") if isinstance(first.get("baseline"), dict) else {}
        candidate = first.get("candidate") if isinstance(first.get("candidate"), dict) else {}
        print(
            f"- greedy n_max={divergence['n_max']} first divergence at token "
            f"{first.get('position')}: {baseline.get('id')} {baseline.get('token')!r} -> "
            f"{candidate.get('id')} {candidate.get('token')!r}"
        )
    for item in report["classification"]:
        print(f"- {item}")


def command_reclassify(args: argparse.Namespace) -> int:
    run_dir = pathlib.Path(args.run_dir).resolve()
    results_path = run_dir / "results.jsonl"
    if not results_path.is_file():
        raise ValueError(f"run has no results.jsonl: {run_dir}")
    rows = read_jsonl(results_path)
    reference_dir = pathlib.Path(args.reference_run).resolve() if args.reference_run else None
    report = classify_run(run_dir, rows, reference_dir)
    report["reclassified_at"] = utc_now()
    report["reclassified_by_harness_version"] = VERSION
    atomic_json(run_dir / "comparisons-reclassified.json", report)
    write_summary(run_dir, rows, report, "summary-reclassified.md")
    print(f"Reclassified: {run_dir}")
    print_verdict(report)
    return 0


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
    conditions = condition_matrix(args.repeats, args.matrix_profile)
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
        "matrix_profile": args.matrix_profile,
        "prime_requests": args.prime_requests,
        "prime_n_max": args.prime_n_max,
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
    prime_rows: list[dict[str, Any]] = []
    for prime_index in range(1, args.prime_requests + 1):
        print(
            f"[prime {prime_index:02d}/{args.prime_requests:02d}] "
            f"request-identical greedy n_max={args.prime_n_max}",
            flush=True,
        )
        prime_args = copy.copy(args)
        prime_args.server_label = f"{args.server_label}-prime{prime_index:02d}"
        prime_condition = {
            "temperature": 0.0,
            "n_max": args.prime_n_max,
            "stream": False,
            "repeat": 0,
        }
        prime_row = run_case(
            run_dir, args.url, base_request, validation, prime_condition,
            prime_args, api_key, server_log,
        )
        prime_row = redact_secret_value(prime_row, api_key)
        prime_rows.append(prime_row)
        if prime_row.get("status") != "ok":
            atomic_json(run_dir / "priming.json", prime_rows)
            raise RuntimeError(
                f"priming request failed: {prime_row.get('error')}"
            )
    if prime_rows:
        atomic_json(run_dir / "priming.json", prime_rows)
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
    print_verdict(report)
    required_gates = (
        "matrix_complete", "fresh_slot_pass", "mtp_exercised",
        "greedy_equivalence_pass", "response_contract_pass", "server_log_capture",
    )
    optional_gates = (
        "stream_transport_pass", "fixed_seed_repeatability_pass",
        "baseline_repeatability_pass",
        "n_max_3_partial_acceptance_coverage_pass",
    )
    passed = all(report["acceptance"][key] is True for key in required_gates)
    passed = passed and all(
        report["acceptance"][key] in (None, True) for key in optional_gates
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
    assert fixture_validation["required_content_substrings"] == []
    assert fixture_validation["allowed_finish_reasons"] == ["stop", "tool_calls"]
    assert fixture_validation["require_nonempty_content_on_stop"] is True
    assert fixture_validation["require_valid_tool_calls_on_tool_finish"] is True
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
    quick_matrix = condition_matrix(2, "greedy-n01")
    assert len(quick_matrix) == 4
    assert {
        (item["temperature"], item["n_max"], item["stream"], item["repeat"])
        for item in quick_matrix
    } == {
        (0.0, 0, False, 1), (0.0, 1, False, 1),
        (0.0, 0, False, 2), (0.0, 1, False, 2),
    }
    production_matrix = condition_matrix(2, "production-n03")
    assert len(production_matrix) == 8
    assert {
        (item["temperature"], item["n_max"], item["stream"], item["repeat"])
        for item in production_matrix
    } == {
        (1.0, n_max, stream, repeat)
        for repeat in (1, 2)
        for n_max in (0, 3)
        for stream in (False, True)
    }
    try:
        condition_matrix(1, "not-a-profile")
        raise AssertionError("unknown matrix profile should fail")
    except ValueError:
        pass

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
    pre_rejection = outputs_before_first_rejection(trace_events)
    assert pre_rejection["first_rejection_event_index"] == 0
    assert pre_rejection["full_acceptance_events_before_first_rejection"] == 0
    assert pre_rejection["minimum_outputs_before_first_rejection"] == 0
    assert pre_rejection["first_rejection_restores_checkpoint"] is False
    assert pre_rejection["all_captured_events_fully_accepted"] is False
    full_then_restore = parse_draft_acceptance_events(
        "1.00 I slot accepted 1/1 draft tokens\n"
        "1.01 I slot accepted 2/2 draft tokens\n"
        "1.02 I slot accepted 0/1 draft tokens (restore checkpoint)\n"
    )
    pre_rejection = outputs_before_first_rejection(full_then_restore)
    assert pre_rejection["full_acceptance_events_before_first_rejection"] == 2
    assert pre_rejection["minimum_outputs_before_first_rejection"] == 5
    assert pre_rejection["first_rejection_observed"] is True
    assert pre_rejection["first_rejection_restores_checkpoint"] is True
    bounded_rejection = parse_draft_acceptance_events(
        "1.00 I slot accepted 1/1 draft tokens\n"
        "1.01 I slot accepted 0/1 draft tokens\n"
        "1.02 I slot accepted 1/1 draft tokens (restore checkpoint)\n"
    )
    pre_rejection = outputs_before_first_rejection(bounded_rejection)
    assert pre_rejection["first_rejection_event_index"] == 1
    assert pre_rejection["minimum_outputs_before_first_rejection"] == 2
    assert pre_rejection["first_rejection_restores_checkpoint"] is False
    fully_accepted = outputs_before_first_rejection(parse_draft_acceptance_events(
        "1.00 I slot accepted 1/1 draft tokens\n"
        "1.01 I slot accepted 1/1 draft tokens\n"
    ))
    assert fully_accepted["first_rejection_observed"] is False
    assert fully_accepted["all_captured_events_fully_accepted"] is True
    assert fully_accepted["minimum_outputs_before_first_rejection"] == 4
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
    first_turn_rules = {
        "min_content_chars": 0,
        "min_reasoning_chars": 5,
        "allowed_finish_reasons": ["stop", "tool_calls"],
        "max_tool_calls": 2,
        "require_nonempty_content_on_stop": True,
        "require_valid_tool_calls_on_tool_finish": True,
    }
    valid_tool_turn = validate_response_contract(
        reconstructed["message"], "tool_calls", first_turn_rules,
    )
    assert valid_tool_turn["passed"] is True
    empty_stop = validate_response_contract(
        {"content": "", "reasoning_content": "enough", "tool_calls": []},
        "stop", first_turn_rules,
    )
    assert empty_stop["passed"] is False
    malformed_tool = copy.deepcopy(reconstructed["message"])
    malformed_tool["tool_calls"][0]["function"]["arguments"] = "not-json"
    invalid_tool_turn = validate_response_contract(
        malformed_tool, "tool_calls", first_turn_rules,
    )
    assert invalid_tool_turn["passed"] is False
    no_tool_turn = validate_response_contract(
        {"content": "", "reasoning_content": "enough", "tool_calls": []},
        "tool_calls", first_turn_rules,
    )
    assert no_tool_turn["passed"] is False
    unnamed_tool = copy.deepcopy(reconstructed["message"])
    unnamed_tool["tool_calls"][0]["function"]["name"] = ""
    assert validate_response_contract(unnamed_tool, "tool_calls", first_turn_rules)["passed"] is False
    non_object_tool = copy.deepcopy(reconstructed["message"])
    non_object_tool["tool_calls"][0]["function"]["arguments"] = "[]"
    assert validate_response_contract(non_object_tool, "tool_calls", first_turn_rules)["passed"] is False

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
    assert token_identity_error(parsed_nonstream["token_trace"]) is None
    assert token_distribution_error(parsed_nonstream["token_trace"]) is None
    incomplete_evidence = copy.deepcopy(parsed_nonstream["token_trace"])
    del incomplete_evidence[1]["bytes"]
    assert "byte sequence" in str(token_identity_error(incomplete_evidence))
    incomplete_distribution = copy.deepcopy(parsed_nonstream["token_trace"])
    incomplete_distribution[1]["top"] = []
    assert token_identity_error(incomplete_distribution) is None
    assert "top-candidate" in str(token_distribution_error(incomplete_distribution))
    assert parsed_nonstream["message"]["reasoning_content"] == "r"
    assert parsed_nonstream["message"]["content"] == "a"
    assert first_token_divergence(
        parsed_nonstream["token_trace"], copy.deepcopy(parsed_nonstream["token_trace"]),
    ) is None
    changed = copy.deepcopy(parsed_nonstream["token_trace"])
    changed[1]["id"] = 12
    divergence = first_token_divergence(parsed_nonstream["token_trace"], changed)
    assert divergence and divergence["kind"] == "token_mismatch" and divergence["position"] == 1
    changed_bytes = copy.deepcopy(parsed_nonstream["token_trace"])
    changed_bytes[1]["bytes"] = [98]
    byte_divergence = first_token_divergence(parsed_nonstream["token_trace"], changed_bytes)
    assert byte_divergence and byte_divergence["identity_fields_differ"] == ["bytes"]
    changed_score = copy.deepcopy(parsed_nonstream["token_trace"])
    changed_score[0]["logprob"] = -0.25
    changed_score[0]["top"][0]["logprob"] = -0.25
    score_divergence = first_scored_distribution_divergence(
        parsed_nonstream["token_trace"], changed_score,
    )
    assert score_divergence and score_divergence["position"] == 0
    assert first_scored_distribution_divergence(
        parsed_nonstream["token_trace"], copy.deepcopy(parsed_nonstream["token_trace"]),
    ) is None
    placeholder_score = copy.deepcopy(parsed_nonstream["token_trace"])
    placeholder_score[0]["logprob"] = 0.0
    placeholder_score[0]["top"] = []
    assert first_scored_distribution_divergence(
        parsed_nonstream["token_trace"], placeholder_score,
    ) is None
    prob_left = copy.deepcopy(parsed_nonstream["token_trace"])
    prob_right = copy.deepcopy(parsed_nonstream["token_trace"])
    for trace in (prob_left, prob_right):
        trace[0].pop("logprob", None)
        trace[0]["prob"] = 0.75
        trace[0]["top"][0].pop("logprob", None)
        trace[0]["top"][0]["prob"] = 0.75
    prob_right[0]["top"][0]["prob"] = 0.5
    prob_divergence = first_scored_distribution_divergence(prob_left, prob_right)
    assert prob_divergence and prob_divergence["score_field"] == "prob"
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
            legacy_distribution_error = (
                condition["temperature"] == 0.0
                and condition["n_max"] == 1
                and condition["stream"] is False
            )
            if condition["temperature"] == 0.0 and condition["n_max"] == 1:
                trace[1]["id"] = 12
            if (
                condition["temperature"] == 0.0
                and condition["n_max"] == 2
                and condition["stream"] is False
            ):
                trace[0]["logprob"] = -0.25
                trace[0]["top"][0]["logprob"] = -0.25
            if legacy_distribution_error:
                trace[1]["top"] = []
            token_file = pathlib.Path("tokens") / f"{identifier}.json"
            atomic_json(root / token_file, trace)
            fake_rows.append({
                "case_id": identifier,
                "status": "error" if legacy_distribution_error else "ok",
                **({
                    "error": "RuntimeError('non-stream token evidence is incomplete: token 1 has no top-candidate evidence')",
                } if legacy_distribution_error else {}),
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
                "token_count": len(trace),
                "http_status": 200,
                "canonical_message_sha256": ("d" if legacy_distribution_error else "a") * 64,
                "finish_reason": "stop",
                "finish_reason_count": 1,
                "fresh_slot": {"passed": True},
                "draft_acceptance_events": (
                    parse_draft_acceptance_events(
                        "1.00 I slot accepted 1/1 draft tokens\n"
                        "1.01 I slot accepted 0/1 draft tokens (restore checkpoint)\n"
                    )
                    if condition["temperature"] == 0.0
                    and condition["n_max"] == 1
                    and condition["stream"] is False
                    else trace_events if condition["n_max"] == 3 else []
                ),
                "timings": {
                    "draft_n": 0 if condition["n_max"] == 0 else condition["n_max"],
                    "draft_n_accepted": 0 if condition["n_max"] == 0 else condition["n_max"],
                },
                "usage": {"completion_tokens": len(trace)},
            })
        fake_report = classify_run(root, fake_rows)
        assert fake_report["acceptance"]["matrix_complete"] is True
        assert fake_report["acceptance"]["n_max_3_partial_acceptance_coverage_pass"] is True
        assert fake_report["acceptance"]["greedy_equivalence_pass"] is False
        assert fake_report["acceptance"]["stream_transport_pass"] is False
        assert fake_report["acceptance"]["fixed_seed_repeatability_pass"] is None
        assert fake_report["acceptance"]["token_distribution_evidence_pass"] is False
        assert len(fake_report["legacy_distribution_rows_recovered"]) == 1
        assert len(fake_report["token_distribution_failures"]) == 1
        assert fake_report["greedy_divergences"][0]["n_max"] == 1
        assert fake_report["pre_rejection_greedy_divergences"][0]["n_max"] == 1
        assert (
            fake_report["pre_rejection_greedy_divergences"][0]
            ["pre_rejection_analysis"]["minimum_outputs_before_first_rejection"]
        ) == 2
        assert any(
            item["n_max"] == 2
            and item["first_scored_distribution_divergence"]["position"] == 0
            for item in fake_report["prompt_boundary_score_drift"]
        )
        assert any(
            item.startswith("target score drift is already visible on output token 0")
            for item in fake_report["classification"]
        )
        assert any(
            item.startswith("n=1 diverged under greedy decoding")
            for item in fake_report["classification"]
        ) is False
        assert any(
            "occurred before the first rejected draft" in item
            for item in fake_report["classification"]
        )

        verdict_report = copy.deepcopy(fake_report)
        stream_items = [
            item for item in verdict_report["comparisons"]
            if item["kind"] == "stream_vs_nonstream"
        ]
        assert len(stream_items) == 8
        for index, item in enumerate(stream_items):
            item["exact_canonical_match"] = index < 6
        verdict_report["acceptance"]["stream_transport_pass"] = False
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            print_verdict(verdict_report)
        assert "Stream transport: FAIL (6/8 matched; 8/8 evaluated)" in captured.getvalue()

        debug_rows = copy.deepcopy(fake_rows)
        for row in debug_rows:
            row["token_count"] = 2
            row["response_contract"] = {
                "passed": False,
                "failures": ["finish reason 'length' is not one of ['stop', 'tool_calls']"],
            }
        atomic_json(root / "manifest.json", {
            "conditions": matrix,
            "max_tokens_override": 2,
        })
        debug_report = classify_run(root, debug_rows)
        assert debug_report["debug_cap_contract_truncation"]["detected"] is True
        assert any(
            item.startswith("all response-contract failures are expected diagnostic truncations")
            for item in debug_report["classification"]
        )

        empty_root = root / "empty-run"
        empty_root.mkdir()
        empty_report = classify_run(empty_root, [])
        assert empty_report["acceptance"]["matrix_complete"] is False
        assert empty_report["acceptance"]["greedy_equivalence_pass"] is False
        assert empty_report["acceptance"]["stream_transport_pass"] is None

        quick_root = root / "quick-run"
        quick_root.mkdir()
        quick_conditions = condition_matrix(1, "greedy-n01")
        atomic_json(quick_root / "manifest.json", {"conditions": quick_conditions})
        quick_keys = {
            (item["temperature"], item["n_max"], item["stream"], item["repeat"])
            for item in quick_conditions
        }
        quick_rows = [
            row for row in fake_rows
            if (row["temperature"], row["n_max"], row["stream"], row["repeat"]) in quick_keys
        ]
        for row in quick_rows:
            source_token = root / row["token_file"]
            atomic_json(quick_root / row["token_file"], json.loads(source_token.read_text()))
        quick_report = classify_run(quick_root, quick_rows)
        assert quick_report["acceptance"]["matrix_complete"] is True
        assert quick_report["comparison_coverage"]["greedy_expected"] == 1
        assert quick_report["comparison_coverage"]["stream_transport_expected"] == 0
        assert quick_report["acceptance"]["stream_transport_pass"] is None
        assert quick_report["acceptance"]["n_max_3_partial_acceptance_coverage_pass"] is None
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            print_verdict(quick_report)
        assert "Stream transport: N/A (0/0 matched; 0/0 evaluated)" in captured.getvalue()
        assert "n_max=3 rollback coverage: N/A" in captured.getvalue()

        production_root = root / "production-run"
        production_root.mkdir()
        atomic_json(production_root / "manifest.json", {"conditions": production_matrix})
        production_rows: list[dict[str, Any]] = []
        for condition in production_matrix:
            source = next(
                row for row in fake_rows
                if row["temperature"] == condition["temperature"]
                and row["n_max"] == condition["n_max"]
                and row["stream"] == condition["stream"]
            )
            row = copy.deepcopy(source)
            row.update(condition)
            row["case_id"] = case_id("production", condition)
            token_file = pathlib.Path("tokens") / f"{row['case_id']}.json"
            atomic_json(production_root / token_file, load_trace(root, source))
            row["token_file"] = str(token_file)
            production_rows.append(row)
        production_report = classify_run(production_root, production_rows)
        assert production_report["comparison_coverage"]["greedy_expected"] == 0
        assert production_report["acceptance"]["greedy_equivalence_pass"] is None
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            print_verdict(production_report)
        assert "Greedy equivalence: N/A (0/0 matched; 0/0 evaluated)" in captured.getvalue()

        cold_root = root / "cold-start-run"
        cold_root.mkdir()
        cold_conditions = condition_matrix(2, "greedy-n01")
        atomic_json(cold_root / "manifest.json", {"conditions": cold_conditions})
        cold_rows: list[dict[str, Any]] = []
        stable_trace = copy.deepcopy(parsed_nonstream["token_trace"])
        cold_trace = copy.deepcopy(stable_trace)
        cold_trace[1]["id"] = 12
        cold_trace[1]["bytes"] = [99]
        cold_trace[1]["token"] = "c"
        for condition in cold_conditions:
            identifier = case_id("cold", condition)
            trace = (
                cold_trace
                if condition["repeat"] == 1 and condition["n_max"] == 0
                else stable_trace
            )
            token_file = pathlib.Path("tokens") / f"{identifier}.json"
            atomic_json(cold_root / token_file, trace)
            cold_rows.append({
                "case_id": identifier,
                "status": "ok",
                "temperature": condition["temperature"],
                "n_max": condition["n_max"],
                "stream": condition["stream"],
                "repeat": condition["repeat"],
                "token_file": str(token_file),
                "token_count": len(trace),
                "http_status": 200,
                "canonical_message_sha256": token_trace_hash(trace),
                "finish_reason": "stop",
                "finish_reason_count": 1,
                "fresh_slot": {"passed": True},
                "draft_acceptance_events": (
                    parse_draft_acceptance_events("1.00 I slot accepted 1/1 draft tokens\n")
                    if condition["n_max"] == 1 else []
                ),
                "timings": {"draft_n": condition["n_max"]},
                "usage": {"completion_tokens": len(trace)},
                "server_log_file": "synthetic.log",
                "server_log_bytes": 1,
            })
        cold_report = classify_run(cold_root, cold_rows)
        assert cold_report["acceptance"]["baseline_repeatability_pass"] is False
        assert cold_report["acceptance"]["greedy_equivalence_pass"] is None
        assert any(
            "first measured request is a cold-start outlier" in item
            for item in cold_report["classification"]
        )
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            print_verdict(cold_report)
        assert "Greedy equivalence: INCONCLUSIVE" in captured.getvalue()

        # The same measured order has a different diagnosis when an explicit
        # request-identical prime proves that repeat 1 is already stable. In
        # that case the intervening n=1 request, not startup, causes the n=0
        # transition observed at repeat 2.
        first_n0 = next(
            row for row in cold_rows
            if row["repeat"] == 1 and row["n_max"] == 0 and not row["stream"]
        )
        atomic_json(cold_root / "priming.json", [{
            **copy.deepcopy(first_n0),
            "case_id": "cold-prime",
            "repeat": 0,
        }])
        transition_report = classify_run(cold_root, cold_rows)
        assert any(
            "persistent MTP-induced process-state transition" in item
            for item in transition_report["classification"]
        )
        assert not any(
            "first measured request is a cold-start outlier" in item
            for item in transition_report["classification"]
        )

        # An MTP prime intentionally changes the scheduler/allocation regime;
        # it must not be treated as evidence that the ordinary n=0 baseline was
        # already stable before the measured matrix.
        mtp_prime = copy.deepcopy(first_n0)
        mtp_prime.update({"case_id": "mtp-prime", "repeat": 0, "n_max": 1})
        atomic_json(cold_root / "priming.json", [mtp_prime])
        mtp_primed_report = classify_run(cold_root, cold_rows)
        assert any(
            "first measured request is a cold-start outlier" in item
            for item in mtp_primed_report["classification"]
        )

        boundary_root = root / "pre-rejection-boundary"
        boundary_root.mkdir()
        atomic_json(boundary_root / "manifest.json", {"conditions": quick_conditions})
        boundary_rows = copy.deepcopy(quick_rows)
        for row in boundary_rows:
            row["status"] = "ok"
            row.pop("error", None)
            trace = copy.deepcopy(parsed_nonstream["token_trace"])
            third = copy.deepcopy(trace[-1])
            third.update({"position": 2, "id": 13, "token": "c", "bytes": [99]})
            third["top"][0].update({"id": 13, "token": "c", "bytes": [99]})
            trace.append(third)
            if row["n_max"] == 1:
                trace[2].update({"id": 14, "token": "d", "bytes": [100]})
                trace[2]["top"][0].update({"id": 14, "token": "d", "bytes": [100]})
                row["draft_acceptance_events"] = parse_draft_acceptance_events(
                    "1.00 I slot accepted 1/1 draft tokens\n"
                    "1.01 I slot accepted 0/1 draft tokens (restore checkpoint)\n"
                )
            token_file = pathlib.Path("tokens") / f"boundary-{row['n_max']}.json"
            atomic_json(boundary_root / token_file, trace)
            row["token_file"] = str(token_file)
            row["token_count"] = len(trace)
        boundary_report = classify_run(boundary_root, boundary_rows)
        assert boundary_report["greedy_divergences"][0]["first_divergence"]["position"] == 2
        assert boundary_report["greedy_divergences"][0]["pre_rejection_analysis"][
            "minimum_outputs_before_first_rejection"
        ] == 2
        assert boundary_report["pre_rejection_greedy_divergences"] == []

        no_rejection_rows = copy.deepcopy(boundary_rows)
        for row in no_rejection_rows:
            if row["n_max"] == 1:
                row["draft_acceptance_events"] = parse_draft_acceptance_events(
                    "1.00 I slot accepted 1/1 draft tokens\n"
                    "1.01 I slot accepted 1/1 draft tokens\n"
                )
        no_rejection_report = classify_run(boundary_root, no_rejection_rows)
        assert no_rejection_report["pre_rejection_greedy_divergences"][0][
            "pre_rejection_analysis"
        ]["proof_kind"] == "captured_without_rejection"
        assert any(
            "no rejected draft occurred in the captured request" in item
            for item in no_rejection_report["classification"]
        )

        truncated_root = root / "truncated-run"
        truncated_root.mkdir()
        two_repeat_conditions = condition_matrix(2)
        atomic_json(truncated_root / "manifest.json", {"conditions": two_repeat_conditions})
        for row in fake_rows:
            source_token = root / row["token_file"]
            atomic_json(truncated_root / row["token_file"], json.loads(source_token.read_text()))
        truncated_report = classify_run(truncated_root, fake_rows)
        assert truncated_report["acceptance"]["matrix_complete"] is False
        assert truncated_report["comparison_coverage"]["greedy_expected"] == 6
        assert truncated_report["comparison_coverage"]["greedy_observed"] == 3
        assert truncated_report["acceptance"]["greedy_equivalence_pass"] is None
        assert truncated_report["acceptance"]["fixed_seed_repeatability_pass"] is False

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

    parsed_prime = build_parser().parse_args([
        "run", "--prime-requests", "1", "--prime-n-max", "1",
    ])
    assert parsed_prime.prime_requests == 1
    assert parsed_prime.prime_n_max == 1

    print(f"qwen_mtp_diag.py {VERSION}: self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the MTP chat diagnostic matrix")
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
    run.add_argument(
        "--prime-requests", type=int, default=0,
        help=(
            "run and archive this many unmeasured request-identical greedy "
            "requests before the matrix; each request is followed by the normal "
            "fresh-slot erase before measurement"
        ),
    )
    run.add_argument(
        "--prime-n-max", type=int, choices=(0, 1, 2, 3), default=0,
        help="speculative.n_max used by unmeasured priming requests (default: 0)",
    )
    run.add_argument(
        "--matrix-profile",
        choices=("full", "greedy-n01", "production-n03"),
        default="full",
        help=(
            "full 16-cell matrix, short greedy n=0/n=1 isolation matrix, or "
            "focused temperature=1 production n=0/n=3 stream matrix"
        ),
    )
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--require-pass", action="store_true")
    reclassify = sub.add_parser(
        "reclassify", help="rebuild verdicts for an existing diagnostic without model requests",
    )
    reclassify.add_argument("run_dir")
    reclassify.add_argument("--reference-run")
    sub.add_parser("self-test", help="run offline parser and comparison tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "reclassify":
            return command_reclassify(args)
        if args.require_pass and not args.reference_run:
            raise ValueError(
                "--require-pass requires --reference-run from a request-identical target-only server"
            )
        if args.require_pass and args.matrix_profile != "full":
            raise ValueError("--require-pass is only valid with --matrix-profile full")
        if args.require_pass and args.repeats < 2:
            raise ValueError("--require-pass requires at least two fixed-seed repeats")
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        if args.prime_requests < 0:
            raise ValueError("--prime-requests must be non-negative")
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
