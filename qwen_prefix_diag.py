#!/usr/bin/env python3
"""Two-slot A/B/A prefix-cache and state-isolation diagnostic for llama-server."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

import qwen_mtp_diag as mtp


VERSION = "1.1.0"


def response_slot_id(response: dict[str, Any]) -> int | None:
    for container in (response, response.get("__verbose")):
        if not isinstance(container, dict):
            continue
        for field in ("id_slot", "slot_id"):
            value = container.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def content_text(message: dict[str, Any]) -> str:
    value = message.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def make_prefix(family: str, secret: str, blocks: int) -> str:
    lines = [
        f"Persistent conversation {family}.",
        f"The private ledger key for this conversation is {secret}.",
        "Retain the ledger exactly and do not substitute facts from another conversation.",
    ]
    for index in range(blocks):
        checksum = (index * 7919 + (17 if family == "A" else 43)) % 104729
        lines.append(
            f"Ledger {family} row {index:04d}: checksum {checksum:06d}; "
            f"owner {family}; immutable archival observation."
        )
    return "\n".join(lines)


def make_payload(
    model: str,
    prefix: str,
    requested_marker: str,
    slot: int | None,
    n_max: int,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                prefix
                + "\n\nCurrent task: Return exactly the following marker and nothing else: "
                + requested_marker
            ),
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "cache_prompt": True,
        "speculative.n_max": n_max,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True,
        "top_logprobs": 1,
    }
    if slot is not None:
        payload["id_slot"] = slot
    return payload


def semantic_request_hash(payload: dict[str, Any]) -> str:
    value = copy.deepcopy(payload)
    value.pop("id_slot", None)
    return mtp.sha256_bytes(mtp.canonical_json_bytes(value))


def cache_count(row: dict[str, Any]) -> int | None:
    timing = row.get("cache_n")
    usage = row.get("usage_cached_n")
    if isinstance(timing, int):
        return timing
    return usage if isinstance(usage, int) else None


def run_request(
    run_dir: pathlib.Path,
    base_url: str,
    case: dict[str, Any],
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    server_log: pathlib.Path | None,
) -> dict[str, Any]:
    identifier = case["case_id"]
    safe_payload = mtp.redact_secret_value(payload, api_key)
    request_bytes = mtp.canonical_json_bytes(safe_payload)
    request_rel = pathlib.Path("requests") / f"{identifier}.json"
    request_path = run_dir / request_rel
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(request_bytes + b"\n")

    mark = mtp.server_log_mark(server_log)
    response = mtp.http_request(
        "POST",
        mtp.join_url(base_url, "v1/chat/completions"),
        request_bytes,
        api_key,
        timeout,
    )
    safe_body = mtp.redact_secret_bytes(response["body"], api_key)
    raw_rel = pathlib.Path("raw-http") / f"{identifier}.json"
    raw_path = run_dir / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(safe_body)
    log_text = mtp.redact_secret_text(mtp.server_log_since(server_log, mark), api_key)
    log_rel = pathlib.Path("request-logs") / f"{identifier}.log"
    log_path = run_dir / log_rel
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    row: dict[str, Any] = {
        **case,
        "ts": mtp.utc_now(),
        "request_file": str(request_rel),
        "request_sha256": mtp.sha256_bytes(request_bytes),
        "semantic_request_sha256": semantic_request_hash(safe_payload),
        "raw_response_file": str(raw_rel),
        "raw_response_sha256": mtp.sha256_bytes(safe_body),
        "request_log_file": str(log_rel),
        "http_status": response["status"],
        "http_wall_ms": round(response["wall_ms"], 3),
    }
    if response["status"] != 200:
        row.update({
            "status": "error",
            "error": (
                f"chat completion returned HTTP {response['status']}: "
                + safe_body[:500].decode("utf-8", errors="replace")
            ),
        })
        return row

    try:
        parsed = mtp.parse_nonstream(response["body"])
        safe_parsed = mtp.redact_secret_value(parsed, api_key)
        response_object = safe_parsed["response"]
        trace = safe_parsed["token_trace"]
        parsed_rel = pathlib.Path("parsed") / f"{identifier}.json"
        token_rel = pathlib.Path("tokens") / f"{identifier}.json"
        mtp.atomic_json(run_dir / parsed_rel, safe_parsed)
        mtp.atomic_json(run_dir / token_rel, trace)
        timings = safe_parsed["timings"]
        usage = safe_parsed["usage"]
        message = safe_parsed["message"]
        row.update({
            "status": "ok",
            "parsed_file": str(parsed_rel),
            "token_file": str(token_rel),
            "token_count": len(trace),
            "token_trace_sha256": mtp.token_trace_hash(trace),
            "canonical_message_sha256": mtp.message_hash(message),
            "content": content_text(message),
            "finish_reason": safe_parsed["finish_reason"],
            "reported_slot": response_slot_id(response_object),
            "prompt_n": int(timings["prompt_n"]) if isinstance(timings.get("prompt_n"), (int, float)) else None,
            "usage_prompt_n": int(usage["prompt_tokens"]) if isinstance(usage.get("prompt_tokens"), (int, float)) else None,
            "cache_n": mtp.timing_cache_n(timings),
            "usage_cached_n": mtp.usage_cached_n(usage),
            "draft_n": int(timings["draft_n"]) if isinstance(timings.get("draft_n"), (int, float)) else None,
            "draft_n_accepted": int(timings["draft_n_accepted"]) if isinstance(timings.get("draft_n_accepted"), (int, float)) else None,
            "token_evidence": mtp.token_evidence_summary(trace),
            "draft_acceptance_events": mtp.parse_draft_acceptance_events(log_text),
        })
    except Exception as exc:
        row.update({"status": "error", "error": f"response parse failed: {exc!r}"})
    return row


def exact_output_comparison(
    run_dir: pathlib.Path, cached: dict[str, Any], cold: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cached_case": cached.get("case_id"),
        "cold_case": cold.get("case_id"),
        "semantic_requests_match": (
            cached.get("semantic_request_sha256") == cold.get("semantic_request_sha256")
        ),
        "messages_match": (
            cached.get("canonical_message_sha256") == cold.get("canonical_message_sha256")
        ),
        "token_traces_match": False,
        "first_token_divergence": None,
    }
    try:
        cached_trace = json.loads((run_dir / cached["token_file"]).read_text(encoding="utf-8"))
        cold_trace = json.loads((run_dir / cold["token_file"]).read_text(encoding="utf-8"))
        divergence = mtp.first_token_divergence(cold_trace, cached_trace)
        result["first_token_divergence"] = divergence
        result["token_traces_match"] = divergence is None
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        result["comparison_error"] = repr(exc)
    result["passed"] = bool(
        result["semantic_requests_match"]
        and result["messages_match"]
        and result["token_traces_match"]
    )
    return result


def classify(
    run_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    min_cache_ratio: float,
    max_cold_cache_tokens: int,
    n_max: int,
) -> dict[str, Any]:
    failures: list[str] = []
    row_checks: list[dict[str, Any]] = []
    by_id = {row["case_id"]: row for row in rows}
    for row in rows:
        count = cache_count(row)
        prompt_n = row.get("prompt_n")
        usage_prompt_n = row.get("usage_prompt_n")
        prompt_total = usage_prompt_n if isinstance(usage_prompt_n, int) else prompt_n
        ratio = (
            count / prompt_total
            if isinstance(count, int) and isinstance(prompt_total, int) and prompt_total > 0
            else None
        )
        checks = {
            "case_id": row["case_id"],
            "requested_slot": row.get("requested_slot"),
            "expected_slot": row.get("expected_slot"),
            "status_ok": row.get("status") == "ok",
            "slot_reported": row.get("reported_slot") is not None,
            "slot_match": (
                row.get("reported_slot") is not None
                and (
                    row.get("expected_slot") is None
                    or row.get("reported_slot") == row.get("expected_slot")
                )
            ),
            "marker_present": row.get("expected_marker", "") in row.get("content", ""),
            "cross_slot_marker_absent": row.get("forbidden_marker", "") not in row.get("content", ""),
            "finish_reason_stop": row.get("finish_reason") == "stop",
            "token_identity_complete": bool(
                row.get("token_count", 0) > 0
                and row.get("token_evidence", {}).get("identity_complete") is True
            ),
            "cache_n": count,
            "prompt_n": prompt_total,
            "cache_ratio": ratio,
            "cache_expectation": row.get("cache_expectation"),
        }
        if row.get("cache_expectation") == "warm":
            checks["cache_pass"] = ratio is not None and ratio >= min_cache_ratio
        else:
            checks["cache_pass"] = count is not None and count <= max_cold_cache_tokens
        checks["passed"] = all(
            checks[name]
            for name in (
                "status_ok", "slot_reported", "slot_match", "marker_present",
                "cross_slot_marker_absent", "finish_reason_stop",
                "token_identity_complete", "cache_pass",
            )
        )
        if not checks["passed"]:
            failed = [
                name for name in (
                    "status_ok", "slot_reported", "slot_match", "marker_present",
                    "cross_slot_marker_absent", "finish_reason_stop",
                    "token_identity_complete", "cache_pass",
                ) if not checks[name]
            ]
            failures.append(f"{row['case_id']}: failed {', '.join(failed)}")
        row_checks.append(checks)

    comparison = exact_output_comparison(
        run_dir, by_id["a-final-cached"], by_id["a-final-cold"],
    )
    if not comparison["passed"]:
        failures.append("cached A final output differs from the request-identical cold A reference")

    automatic_comparison: dict[str, Any] | None = None
    automatic_rows = [row for row in rows if row.get("routing") == "automatic"]
    automatic_slot_map: dict[str, int | None] | None = None
    if automatic_rows:
        auto_a = by_id["auto-a-establish"].get("reported_slot")
        auto_b = by_id["auto-b-establish"].get("reported_slot")
        automatic_slot_map = {"A": auto_a, "B": auto_b}
        if auto_a is None or auto_b is None or auto_a == auto_b:
            failures.append("automatic LCP routing did not establish A and B in distinct slots")
        automatic_comparison = exact_output_comparison(
            run_dir, by_id["auto-a-final"], by_id["a-final-cold"],
        )
        if not automatic_comparison["passed"]:
            failures.append("automatically routed cached A output differs from the cold A reference")

    draft_total = sum(
        row.get("draft_n", 0)
        for row in rows
        if isinstance(row.get("draft_n"), int)
    )
    if n_max > 0 and draft_total <= 0:
        failures.append("speculative decoding was requested but no draft tokens were reported")

    return {
        "schema": 1,
        "ts": mtp.utc_now(),
        "passed": not failures,
        "failures": failures,
        "row_checks": row_checks,
        "exact_cached_vs_cold": comparison,
        "automatic_exact_cached_vs_cold": automatic_comparison,
        "automatic_slot_map": automatic_slot_map,
        "draft_n_total": draft_total,
        "criteria": {
            "min_warm_cache_ratio": min_cache_ratio,
            "max_cold_cache_tokens": max_cold_cache_tokens,
            "requested_n_max": n_max,
        },
    }


def write_summary(run_dir: pathlib.Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Two-slot prefix diagnostic: {run_dir.name}",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Case | Slot | Cache | Prompt | Ratio | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["row_checks"]:
        ratio = item["cache_ratio"]
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else "n/a"
        lines.append(
            f"| {item['case_id']} | "
            f"{item['requested_slot'] if item['requested_slot'] is not None else 'n/a'} | "
            f"{item['cache_n'] if item['cache_n'] is not None else 'n/a'} | "
            f"{item['prompt_n'] if item['prompt_n'] is not None else 'n/a'} | "
            f"{ratio_text} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    exact = report["exact_cached_vs_cold"]
    lines.extend([
        "",
        "## Exact cached-versus-cold A-final comparison",
        "",
        f"- Semantic requests match: {exact['semantic_requests_match']}",
        f"- Canonical messages match: {exact['messages_match']}",
        f"- Exact token traces match: {exact['token_traces_match']}",
    ])
    automatic = report.get("automatic_exact_cached_vs_cold")
    if isinstance(automatic, dict):
        lines.extend([
            "",
            "## Automatic LCP routing",
            "",
            f"- Established slot map: {report.get('automatic_slot_map')}",
            f"- Semantic requests match cold reference: {automatic['semantic_requests_match']}",
            f"- Canonical messages match cold reference: {automatic['messages_match']}",
            f"- Exact token traces match cold reference: {automatic['token_traces_match']}",
        ])
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in report["failures"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_run(args: argparse.Namespace) -> int:
    api_key, api_key_source = mtp.resolve_api_key(args)
    base_url = args.url.rstrip("/")
    server_log = pathlib.Path(args.server_log).resolve() if args.server_log else None
    if server_log is not None and not server_log.is_file():
        raise ValueError(f"--server-log is not a readable file: {server_log}")
    if args.slot_a == args.slot_b:
        raise ValueError("--slot-a and --slot-b must be different")

    run_dir = pathlib.Path(args.output_root).resolve() / (
        f"{mtp.stamp()}-prefix-slots-{mtp.safe_name(args.server_label)}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    initial_log_mark = mtp.server_log_mark(server_log)
    secret_a = "A_LEDGER_314159"
    secret_b = "B_LEDGER_271828"
    prefix_a = make_prefix("A", secret_a, args.prefix_blocks)
    prefix_b = make_prefix("B", secret_b, args.prefix_blocks)

    sequence = [
        ("a-establish", "A", args.slot_a, prefix_a, "A_ESTABLISHED_1001", "B_", "cold"),
        ("b-establish", "B", args.slot_b, prefix_b, "B_ESTABLISHED_2001", "A_", "cold"),
        ("a-resume", "A", args.slot_a, prefix_a, "A_RESUMED_1002", "B_", "warm"),
        ("b-resume", "B", args.slot_b, prefix_b, "B_RESUMED_2002", "A_", "warm"),
        ("a-final-cached", "A", args.slot_a, prefix_a, "A_FINAL_1003", "B_", "warm"),
    ]
    rows: list[dict[str, Any]] = []
    erasures = [
        {"slot": args.slot_a, **mtp.erase_slot(base_url, args.slot_a, api_key, args.timeout)},
        {"slot": args.slot_b, **mtp.erase_slot(base_url, args.slot_b, api_key, args.timeout)},
    ]
    for index, (identifier, family, slot, prefix, marker, forbidden, expectation) in enumerate(sequence, 1):
        print(f"[{index:02d}/11] {identifier} slot={slot} cache={expectation}")
        payload = make_payload(
            args.model, prefix, marker, slot, args.n_max, args.max_tokens, args.seed,
        )
        row = run_request(
            run_dir, base_url,
            {
                "case_id": identifier,
                "family": family,
                "requested_slot": slot,
                "expected_slot": slot,
                "routing": "explicit",
                "expected_marker": marker,
                "forbidden_marker": forbidden,
                "cache_expectation": expectation,
            },
            payload, api_key, args.timeout, server_log,
        )
        rows.append(row)

    erasures.append({
        "slot": args.slot_b,
        "purpose": "cold A-final reference",
        **mtp.erase_slot(base_url, args.slot_b, api_key, args.timeout),
    })
    print(f"[06/11] a-final-cold slot={args.slot_b} cache=cold")
    cold_payload = make_payload(
        args.model, prefix_a, "A_FINAL_1003", args.slot_b, args.n_max,
        args.max_tokens, args.seed,
    )
    rows.append(run_request(
        run_dir, base_url,
        {
            "case_id": "a-final-cold",
            "family": "A",
            "requested_slot": args.slot_b,
            "expected_slot": args.slot_b,
            "routing": "explicit-cold-reference",
            "expected_marker": "A_FINAL_1003",
            "forbidden_marker": "B_",
            "cache_expectation": "cold",
        },
        cold_payload, api_key, args.timeout, server_log,
    ))

    erasures.extend([
        {
            "slot": args.slot_a,
            "purpose": "automatic-routing phase reset",
            **mtp.erase_slot(base_url, args.slot_a, api_key, args.timeout),
        },
        {
            "slot": args.slot_b,
            "purpose": "automatic-routing phase reset",
            **mtp.erase_slot(base_url, args.slot_b, api_key, args.timeout),
        },
    ])
    automatic_sequence = [
        ("auto-a-establish", "A", prefix_a, "A_ESTABLISHED_1001", "B_", "cold"),
        ("auto-b-establish", "B", prefix_b, "B_ESTABLISHED_2001", "A_", "cold"),
        ("auto-a-resume", "A", prefix_a, "A_RESUMED_1002", "B_", "warm"),
        ("auto-b-resume", "B", prefix_b, "B_RESUMED_2002", "A_", "warm"),
        ("auto-a-final", "A", prefix_a, "A_FINAL_1003", "B_", "warm"),
    ]
    automatic_slots: dict[str, int | None] = {"A": None, "B": None}
    for index, (identifier, family, prefix, marker, forbidden, expectation) in enumerate(
        automatic_sequence, 7,
    ):
        expected_slot = automatic_slots[family]
        print(
            f"[{index:02d}/11] {identifier} slot=automatic "
            f"expected={expected_slot} cache={expectation}"
        )
        payload = make_payload(
            args.model, prefix, marker, None, args.n_max, args.max_tokens, args.seed,
        )
        row = run_request(
            run_dir, base_url,
            {
                "case_id": identifier,
                "family": family,
                "requested_slot": None,
                "expected_slot": expected_slot,
                "routing": "automatic",
                "expected_marker": marker,
                "forbidden_marker": forbidden,
                "cache_expectation": expectation,
            },
            payload, api_key, args.timeout, server_log,
        )
        rows.append(row)
        if automatic_slots[family] is None and isinstance(row.get("reported_slot"), int):
            automatic_slots[family] = row["reported_slot"]

    mtp.atomic_json(run_dir / "results.json", rows)
    mtp.atomic_json(run_dir / "slot-erasures.json", erasures)
    report = classify(
        run_dir, rows, args.min_cache_ratio, args.max_cold_cache_tokens, args.n_max,
    )
    mtp.atomic_json(run_dir / "report.json", report)
    manifest = {
        "schema": 1,
        "ts": mtp.utc_now(),
        "version": VERSION,
        "server_label": args.server_label,
        "url": base_url,
        "api_key_source": api_key_source,
        "model": args.model,
        "slots": {"A": args.slot_a, "B": args.slot_b},
        "prefix_blocks": args.prefix_blocks,
        "n_max": args.n_max,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "server_log": str(server_log) if server_log else None,
        "sequence": [row["case_id"] for row in rows],
    }
    mtp.atomic_json(run_dir / "manifest.json", manifest)
    if server_log is not None:
        complete_log = mtp.redact_secret_text(
            mtp.server_log_since(server_log, initial_log_mark), api_key,
        )
        (run_dir / "server.log").write_text(complete_log, encoding="utf-8")
    write_summary(run_dir, report)
    print(f"Results: {run_dir}")
    print(f"Two-slot A/B/A prefix state: {'PASS' if report['passed'] else 'FAIL'}")
    for failure in report["failures"]:
        print(f"- {failure}")
    return 0 if report["passed"] or not args.require_pass else 2


def self_test() -> None:
    prefix_a = make_prefix("A", "A_LEDGER_314159", 3)
    prefix_b = make_prefix("B", "B_LEDGER_271828", 3)
    assert prefix_a != prefix_b
    assert "A_LEDGER_314159" in prefix_a and "B_LEDGER_271828" not in prefix_a
    payload_a = make_payload("model", prefix_a, "A_FINAL_1003", 0, 0, 16, 1234)
    payload_b = make_payload("model", prefix_a, "A_FINAL_1003", 1, 0, 16, 1234)
    assert payload_a["cache_prompt"] is True
    assert semantic_request_hash(payload_a) == semantic_request_hash(payload_b)
    assert response_slot_id({"__verbose": {"id_slot": 1}}) == 1
    assert response_slot_id({}) is None

    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw)
        trace = [{"position": 0, "id": 1, "bytes": [65], "token": "A", "top": []}]
        rows: list[dict[str, Any]] = []
        case_specs = [
            ("a-establish", 0, "A_ESTABLISHED_1001", "cold", 0),
            ("b-establish", 1, "B_ESTABLISHED_2001", "cold", 0),
            ("a-resume", 0, "A_RESUMED_1002", "warm", 900),
            ("b-resume", 1, "B_RESUMED_2002", "warm", 900),
            ("a-final-cached", 0, "A_FINAL_1003", "warm", 900),
            ("a-final-cold", 1, "A_FINAL_1003", "cold", 0),
        ]
        final_semantic = semantic_request_hash(payload_a)
        for identifier, slot, marker, expectation, cached in case_specs:
            token_file = pathlib.Path("tokens") / f"{identifier}.json"
            mtp.atomic_json(root / token_file, trace)
            rows.append({
                "case_id": identifier,
                "status": "ok",
                "requested_slot": slot,
                "reported_slot": slot,
                "expected_marker": marker,
                "forbidden_marker": "NEVER_PRESENT",
                "content": marker,
                "cache_expectation": expectation,
                "cache_n": cached,
                "usage_cached_n": cached,
                "prompt_n": 1000,
                "usage_prompt_n": 1000,
                "draft_n": 0,
                "token_file": str(token_file),
                "token_count": len(trace),
                "token_evidence": {"identity_complete": True},
                "finish_reason": "stop",
                "canonical_message_sha256": "same" if identifier.startswith("a-final") else identifier,
                "semantic_request_sha256": final_semantic if identifier.startswith("a-final") else identifier,
            })
        automatic_specs = [
            ("auto-a-establish", "A", 0, None, "A_ESTABLISHED_1001", "cold", 0),
            ("auto-b-establish", "B", 1, None, "B_ESTABLISHED_2001", "cold", 0),
            ("auto-a-resume", "A", 0, 0, "A_RESUMED_1002", "warm", 900),
            ("auto-b-resume", "B", 1, 1, "B_RESUMED_2002", "warm", 900),
            ("auto-a-final", "A", 0, 0, "A_FINAL_1003", "warm", 900),
        ]
        for identifier, family, reported, expected, marker, expectation, cached in automatic_specs:
            token_file = pathlib.Path("tokens") / f"{identifier}.json"
            mtp.atomic_json(root / token_file, trace)
            rows.append({
                "case_id": identifier,
                "family": family,
                "routing": "automatic",
                "status": "ok",
                "requested_slot": None,
                "expected_slot": expected,
                "reported_slot": reported,
                "expected_marker": marker,
                "forbidden_marker": "NEVER_PRESENT",
                "content": marker,
                "cache_expectation": expectation,
                "cache_n": cached,
                "usage_cached_n": cached,
                "prompt_n": 1000,
                "usage_prompt_n": 1000,
                "draft_n": 0,
                "token_file": str(token_file),
                "token_count": len(trace),
                "token_evidence": {"identity_complete": True},
                "finish_reason": "stop",
                "canonical_message_sha256": "same" if identifier == "auto-a-final" else identifier,
                "semantic_request_sha256": final_semantic if identifier == "auto-a-final" else identifier,
            })
        report = classify(root, rows, 0.75, 0, 0)
        assert report["passed"] is True
        assert report["automatic_slot_map"] == {"A": 0, "B": 1}
        assert report["automatic_exact_cached_vs_cold"]["passed"] is True
        rows[2]["cache_n"] = 10
        report = classify(root, rows, 0.75, 0, 0)
        assert report["passed"] is False
        assert any("a-resume" in item for item in report["failures"])
        rows[2]["cache_n"] = 900
        next(row for row in rows if row["case_id"] == "auto-b-establish")["reported_slot"] = 0
        report = classify(root, rows, 0.75, 0, 0)
        assert report["passed"] is False
        assert any("distinct slots" in item for item in report["failures"])
    print(f"qwen_prefix_diag.py {VERSION}: self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the two-slot A/B/A prefix diagnostic")
    run.add_argument("--url", default="http://127.0.0.1:8080")
    run.add_argument("--output-root", default="results")
    run.add_argument("--server-label", default="target-two-slot")
    run.add_argument("--server-log")
    run.add_argument("--api-key-env", default="QWEN_TEST_API_KEY")
    run.add_argument("--api-key-file")
    run.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next-think")
    run.add_argument("--slot-a", type=int, default=0)
    run.add_argument("--slot-b", type=int, default=1)
    run.add_argument("--prefix-blocks", type=int, default=192)
    run.add_argument("--n-max", type=int, choices=(0, 1, 2, 3), default=0)
    run.add_argument("--max-tokens", type=int, default=32)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--min-cache-ratio", type=float, default=0.70)
    run.add_argument("--max-cold-cache-tokens", type=int, default=0)
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--require-pass", action="store_true")
    sub.add_parser("self-test", help="run offline construction and classification tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.slot_a < 0 or args.slot_b < 0:
            raise ValueError("slot IDs must be non-negative")
        if args.prefix_blocks < 1:
            raise ValueError("--prefix-blocks must be positive")
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        if not 0 < args.min_cache_ratio <= 1:
            raise ValueError("--min-cache-ratio must be in (0, 1]")
        if args.max_cold_cache_tokens < 0:
            raise ValueError("--max-cold-cache-tokens must be non-negative")
        return command_run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
