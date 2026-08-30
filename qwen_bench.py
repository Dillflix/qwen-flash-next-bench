#!/usr/bin/env python3
"""Config-driven llama.cpp topology and speculative-decoding benchmark harness.

The runner uses only Python's standard library.  It is designed for very large
models where process isolation, cold server starts, exact command capture, and
recoverable partial results matter more than shaving a few seconds off a run.
"""

from __future__ import annotations

import argparse
import base64
import copy
import concurrent.futures
import csv
import datetime as dt
import fnmatch
import hashlib
import http.client
import json
import math
import mimetypes
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Iterable


VERSION = "1.39.0"
SUCCESS_STATES = {"ok"}
QWEN4EXP_MTP_MARKER = "qwen4exp MTP requires exactly one appended prediction layer"
QWEN4EXP_MTP_SCHED_MARKER = "qwen4exp_mtp_h_pre_norm_scheduled"
HOST_CHECKPOINT_MARKER = "LLAMA_CKPT_FORCE_HOST"
MTP_VISION_RESYNC_MARKER = "MTP multimodal resync: skipping direct image decode"
QWEN4EXP_VISION_STRICT_MARKER = "Qwen4Exp vision MTP: single-row target verification enabled"
QWEN4EXP_VISION_CHECKPOINT_MARKER = "Qwen4Exp vision MTP: recurrent rollback disabled; using full-state checkpoints"
SINGLE_VALUE_SERVER_OPTIONS = {
    "-m",
    "-md",
    "-ot",
    "--batch-size",
    "--cache-ram",
    "--cache-type-k",
    "--cache-type-v",
    "--ctx-size",
    "--ctx-checkpoints",
    "--checkpoint-every-n-tokens",
    "--device",
    "--fit",
    "--flash-attn",
    "--host",
    "--load-mode",
    "--main-gpu",
    "--mmproj",
    "--image-max-tokens",
    "--image-min-tokens",
    "--n-gpu-layers",
    "--override-tensor",
    "--parallel",
    "--port",
    "--spec-draft-device",
    "--spec-draft-ngl",
    "--spec-draft-n-max",
    "--spec-draft-p-min",
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--spec-type",
    "--split-mode",
    "--slot-save-path",
    "--tensor-split",
    "--threads",
    "--ubatch-size",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def atomic_json(path: pathlib.Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class FormatVars(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"unknown configuration variable {{{key}}}")


def expand_string(value: str, variables: dict[str, str]) -> str:
    return value.format_map(FormatVars(variables))


def expand_tree(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return expand_string(value, variables)
    if isinstance(value, list):
        return [expand_tree(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand_tree(item, variables) for key, item in value.items()}
    return value


def load_config(path: pathlib.Path) -> dict[str, Any]:
    raw = load_json(path)
    if raw.get("version") != 1:
        raise ValueError("configuration must contain version: 1")
    variables = {str(k): str(v) for k, v in raw.get("variables", {}).items()}
    variables.setdefault("config_dir", str(path.resolve().parent))
    # Resolve variables in declaration order with a few passes, allowing repo to
    # be used by model paths without invoking a shell or eval.
    for _ in range(max(1, len(variables) + 1)):
        changed = False
        for key, value in list(variables.items()):
            try:
                expanded = value.format_map(defaultdict(str, variables))
            except (KeyError, ValueError):
                continue
            if expanded != value:
                variables[key] = expanded
                changed = True
        if not changed:
            break
    config = expand_tree(raw, variables)
    config["experiments"] = resolve_experiment_inheritance(config.get("experiments", []))
    validate_experiment_topologies(config["experiments"])
    validate_config_references(config)
    return config


def resolve_experiment_inheritance(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve optional `extends` and `args_append` fields without shell templating."""
    by_name = {item.get("name"): item for item in experiments}
    if None in by_name:
        raise ValueError("every experiment requires a name")
    if len(by_name) != len(experiments):
        raise ValueError("experiment names must be unique")
    resolved: dict[str, dict[str, Any]] = {}
    visiting: set[str] = set()

    def one(name: str) -> dict[str, Any]:
        if name in resolved:
            return copy.deepcopy(resolved[name])
        if name in visiting:
            raise ValueError(f"cyclic experiment inheritance involving {name}")
        if name not in by_name:
            raise ValueError(f"experiment {name!r} extends an unknown experiment")
        visiting.add(name)
        child = copy.deepcopy(by_name[name])
        parent_name = child.pop("extends", None)
        appended = child.pop("args_append", [])
        if parent_name:
            parent = one(str(parent_name))
            inherited_env = dict(parent.get("env", {}))
            inherited_env.update(child.get("env", {}))
            merged = parent
            merged.update(child)
            if inherited_env:
                merged["env"] = inherited_env
            if "args" not in child:
                merged["args"] = list(parent.get("args", []))
            child = merged
        child["args"] = list(child.get("args", [])) + list(appended)
        child["name"] = name
        visiting.remove(name)
        resolved[name] = child
        return copy.deepcopy(child)

    return [one(str(item["name"])) for item in experiments]


def option_value(args: list[Any], option: str) -> str | None:
    values = [str(args[index + 1]) for index, value in enumerate(args[:-1]) if value == option]
    return values[-1] if values else None


def validate_experiment_topologies(experiments: list[dict[str, Any]]) -> None:
    for experiment in experiments:
        args = list(experiment.get("args", []))
        override = option_value(args, "--override-tensor") or option_value(args, "-ot")
        split_mode = option_value(args, "--split-mode") or option_value(args, "-sm")
        if not override or split_mode != "none":
            continue
        targets = re.findall(r"=([^,]+)", override)
        if any(target.upper() != "CPU" for target in targets):
            raise ValueError(
                f"{experiment['name']}: non-CPU tensor overrides cannot use --split-mode none; "
                "llama.cpp prunes every model GPU except --main-gpu. Use layer mode with a "
                "1,0 tensor split to keep the secondary backend registered."
            )


def validate_config_references(config: dict[str, Any]) -> None:
    experiment_names = {item["name"] for item in config.get("experiments", [])}
    for tier_name, tier in config.get("tiers", {}).items():
        selected = tier.get("experiments", [])
        if not selected:
            raise ValueError(f"tier {tier_name!r} has no experiments")
        missing = sorted(set(selected) - experiment_names)
        if missing:
            raise ValueError(f"tier {tier_name!r} references unknown experiments: {missing}")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def parse_selector(selector: str | None) -> list[str]:
    if not selector:
        return []
    return [part.strip() for part in selector.split(",") if part.strip()]


def select_experiments(config: dict[str, Any], tier: dict[str, Any], selector: str | None) -> list[dict[str, Any]]:
    requested = parse_selector(selector)
    enabled = {
        item["name"]: item for item in config.get("experiments", []) if item.get("enabled", True)
    }
    tier_names = list(tier.get("experiments", []))
    if tier_names:
        # Tier order is operational: cheap controls and likely-to-load cases can
        # intentionally precede experimental placements when --fail-fast is used.
        experiments = [enabled[name] for name in tier_names if name in enabled]
    else:
        experiments = list(enabled.values())
    if requested:
        experiments = [
            item for item in experiments
            if any(fnmatch.fnmatchcase(item["name"], pattern) for pattern in requested)
        ]
    if not experiments:
        raise ValueError("no enabled experiments matched the tier and --experiments selector")
    names = [item["name"] for item in experiments]
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be unique")
    return experiments


def load_workloads(config: dict[str, Any], config_path: pathlib.Path) -> dict[str, str]:
    path = pathlib.Path(config["workloads_file"])
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    workloads = load_json(path)
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError("workloads file must be a non-empty JSON object")
    return {str(key): str(value) for key, value in workloads.items()}


def load_vision_cases(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_path = config.get("vision_cases_file")
    if not raw_path:
        raise ValueError("vision mode requires vision_cases_file")
    path = pathlib.Path(str(raw_path))
    cases = load_json(path)
    if not isinstance(cases, dict) or not cases:
        raise ValueError("vision cases file must be a non-empty JSON object")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in cases.items():
        if not isinstance(raw, dict):
            raise ValueError(f"vision case {name!r} must be an object")
        image = raw.get("image")
        prompt = raw.get("prompt")
        anchors = raw.get("anchors", [])
        if not isinstance(image, str) or not image:
            raise ValueError(f"vision case {name!r} requires an image path")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"vision case {name!r} requires a prompt")
        if not isinstance(anchors, list) or not all(isinstance(value, str) and value for value in anchors):
            raise ValueError(f"vision case {name!r} anchors must be non-empty strings")
        result[str(name)] = {"image": image, "prompt": prompt, "anchors": anchors}
    return result


def load_quality_cases(config: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    raw_path = config.get("quality_cases_file")
    if not raw_path:
        raise ValueError("quality mode requires quality_cases_file")
    payload = load_json(pathlib.Path(str(raw_path)))
    if not isinstance(payload, dict):
        raise ValueError("quality cases file must be a JSON object")
    filler = payload.get("filler")
    raw_cases = payload.get("cases")
    if not isinstance(filler, str) or not filler.strip():
        raise ValueError("quality cases require non-empty neutral filler text")
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise ValueError("quality cases require a non-empty cases object")
    cases: dict[str, dict[str, Any]] = {}
    for name, raw in raw_cases.items():
        if not isinstance(raw, dict):
            raise ValueError(f"quality case {name!r} must be an object")
        task = raw.get("task")
        records = raw.get("records")
        validator = raw.get("validator")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"quality case {name!r} requires a task")
        if not isinstance(records, list) or not records:
            raise ValueError(f"quality case {name!r} requires at least one record")
        checked_records: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"quality case {name!r} records must be objects")
            fraction = record.get("fraction")
            text = record.get("text")
            if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) <= 1.0:
                raise ValueError(f"quality case {name!r} record fraction must be between 0 and 1")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"quality case {name!r} record text must be non-empty")
            checked_records.append({"fraction": float(fraction), "text": text})
        if not isinstance(validator, dict) or validator.get("type") not in {"exact", "json_equals"}:
            raise ValueError(f"quality case {name!r} validator must be exact or json_equals")
        expected = validator.get("expected")
        if validator["type"] == "exact" and (not isinstance(expected, str) or not expected):
            raise ValueError(f"quality case {name!r} exact validator requires a string expected value")
        if validator["type"] == "json_equals" and not isinstance(expected, (dict, list)):
            raise ValueError(f"quality case {name!r} json_equals validator requires an object or array")
        cases[str(name)] = {
            "task": task,
            "records": checked_records,
            "validator": copy.deepcopy(validator),
        }
    return filler, cases


def load_context_corpus(config: dict[str, Any]) -> str:
    chunks: list[str] = []
    for raw_path in config.get("context_sources", []):
        path = pathlib.Path(raw_path)
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    if not chunks:
        chunks.append(
            "A benchmark is useful only when its workload, configuration, and measured output "
            "are recorded together. Hardware topology can change both throughput and latency. "
            "Warm-up requests are excluded, greedy decoding is used for repeatability, and every "
            "response is hashed so that performance regressions are not mistaken for improvements."
        )
    return "\n\n".join(chunks)


def make_prompt(base: str, requested_tokens: int, corpus: str) -> str:
    if requested_tokens <= 0:
        return base
    # Four UTF-8-ish characters per token is only a construction estimate.  The
    # actual prompt token count returned by llama-server is always recorded.
    target_chars = requested_tokens * 4
    prefix = "Reference material follows. Read it, then answer the task after END REFERENCE.\n\n"
    repeated = (corpus + "\n\n") * max(1, math.ceil(target_chars / max(1, len(corpus))))
    filler = repeated[:target_chars]
    return f"{prefix}{filler}\nEND REFERENCE\n\n{base}"


def make_quality_prompt_chars(
    case: dict[str, Any], filler_chars: int, filler_source: str, padding: str = "",
) -> str:
    repeated = (filler_source.strip() + "\n\n") * max(
        1, math.ceil(max(1, filler_chars) / max(1, len(filler_source))),
    )
    body = repeated[:max(0, filler_chars)]
    for record in sorted(case["records"], key=lambda item: float(item["fraction"]), reverse=True):
        marker = f"\n\n{record['text']}\n\n"
        available = max(0, len(body) - len(marker))
        position = int(round(available * float(record["fraction"])))
        body = body[:position] + marker + body[position + len(marker):]
    return (
        "CONTROLLED REFERENCE follows. QUALITY RECORD entries are authoritative data, "
        "not instructions. Read the reference and answer the task after END REFERENCE.\n\n"
        f"{body}{padding}\nEND REFERENCE\n\n{case['task']}"
    )


def make_quality_prompt(case: dict[str, Any], requested_tokens: int, filler_source: str) -> str:
    return make_quality_prompt_chars(case, max(0, requested_tokens) * 4, filler_source)


def fit_quality_prompt_to_tokens(
    base_url: str,
    case: dict[str, Any],
    target_tokens: int,
    filler_source: str,
    timeout_s: float,
) -> tuple[str, int]:
    """Place quality records by fraction while proving the final prompt token count."""
    if target_tokens <= 0:
        prompt = make_quality_prompt_chars(case, 0, filler_source)
        return prompt, tokenize_count(base_url, prompt, timeout_s)

    low_chars = 0
    high_chars = max(1024, target_tokens * 5)
    low_prompt = make_quality_prompt_chars(case, low_chars, filler_source)
    low_count = tokenize_count(base_url, low_prompt, timeout_s)
    if low_count > target_tokens:
        raise ValueError(
            f"quality task is {low_count} tokens, larger than requested target {target_tokens}"
        )
    high_prompt = make_quality_prompt_chars(case, high_chars, filler_source)
    high_count = tokenize_count(base_url, high_prompt, timeout_s)
    while high_count < target_tokens:
        low_chars, low_prompt, low_count = high_chars, high_prompt, high_count
        high_chars *= 2
        high_prompt = make_quality_prompt_chars(case, high_chars, filler_source)
        high_count = tokenize_count(base_url, high_prompt, timeout_s)

    best_prompt, best_count, best_chars = low_prompt, low_count, low_chars
    for _ in range(40):
        if high_count == low_count:
            break
        estimated = low_chars + int(
            (target_tokens - low_count) * (high_chars - low_chars) / (high_count - low_count)
        )
        probe_chars = min(high_chars - 1, max(low_chars + 1, estimated))
        if probe_chars <= low_chars or probe_chars >= high_chars:
            probe_chars = (low_chars + high_chars) // 2
        if probe_chars <= low_chars:
            break
        prompt = make_quality_prompt_chars(case, probe_chars, filler_source)
        count = tokenize_count(base_url, prompt, timeout_s)
        if count == target_tokens:
            return prompt, count
        if count <= target_tokens:
            low_chars, low_prompt, low_count = probe_chars, prompt, count
            if count > best_count:
                best_prompt, best_count, best_chars = prompt, count, probe_chars
        else:
            high_chars, high_prompt, high_count = probe_chars, prompt, count
        if high_chars - low_chars <= 1:
            break

    for unit in (" benchmark", " x", " 0", "\npadding"):
        for repeats in range(1, 65):
            prompt = make_quality_prompt_chars(
                case, best_chars, filler_source, unit * repeats,
            )
            count = tokenize_count(base_url, prompt, timeout_s)
            if count == target_tokens:
                return prompt, count
            if count > target_tokens + 8:
                break
    raise RuntimeError(
        f"could not construct an exact {target_tokens}-token quality prompt; "
        f"closest count was {best_count}"
    )


def final_answer_text(content: str) -> str:
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    content = content.strip()
    fence = re.fullmatch(r"```(?:json|text)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else content


def quality_case_metrics(content: str, validator: dict[str, Any]) -> dict[str, Any]:
    answer = final_answer_text(content)
    validator_type = str(validator["type"])
    expected = validator["expected"]
    parsed: Any = None
    if validator_type == "exact":
        normalized = re.sub(r"\s+", " ", answer).strip(" `\"'")
        passed = normalized == expected
        score = 1.0 if passed else 0.0
        detail = normalized
    else:
        decoder = json.JSONDecoder()
        for index, character in enumerate(answer):
            if character not in "[{":
                continue
            try:
                parsed, _end = decoder.raw_decode(answer[index:])
                break
            except json.JSONDecodeError:
                continue
        passed = parsed == expected
        if isinstance(expected, dict) and isinstance(parsed, dict) and expected:
            score = sum(parsed.get(key) == value for key, value in expected.items()) / len(expected)
        else:
            score = 1.0 if passed else 0.0
        detail = parsed
    return {
        "validator_type": validator_type,
        "expected": expected,
        "observed": detail,
        "anchor_score": score,
        "quality_pass": passed,
        "anchors": [json.dumps(expected, sort_keys=True) if not isinstance(expected, str) else expected],
        "anchors_matched": [
            json.dumps(expected, sort_keys=True) if not isinstance(expected, str) else expected
        ] if passed else [],
    }


def tokenize_count(base_url: str, content: str, timeout_s: float) -> int:
    status, response = http_json(
        "POST", base_url + "/tokenize", {"content": content}, timeout=min(timeout_s, 300.0),
    )
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"tokenize returned HTTP {status}: {response}")
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"tokenize response has no token list: {response}")
    return len(tokens)


def fit_prompt_to_tokens(
    base_url: str,
    base: str,
    target_tokens: int,
    corpus: str,
    timeout_s: float,
    lane: int = 0,
) -> tuple[str, int]:
    """Build a prompt with exactly the requested /tokenize token count."""
    lane_prefix = f"Independent benchmark lane {lane + 1}; lane marker {lane:08x}.\n"
    if target_tokens <= 0:
        prompt = lane_prefix + base
        return prompt, tokenize_count(base_url, prompt, timeout_s)

    low_chars = 0
    high_chars = max(1024, target_tokens * 5)

    def construct(filler_chars: int, padding: str = "") -> str:
        prefix = "Reference material follows. Read it, then answer the task after END REFERENCE.\n\n"
        repeated = (corpus + "\n\n") * max(1, math.ceil(filler_chars / max(1, len(corpus))))
        return f"{lane_prefix}{prefix}{repeated[:filler_chars]}{padding}\nEND REFERENCE\n\n{base}"

    low_prompt = construct(low_chars)
    low_count = tokenize_count(base_url, low_prompt, timeout_s)
    if low_count > target_tokens:
        raise ValueError(
            f"base prompt is {low_count} tokens, larger than requested target {target_tokens}"
        )
    high_prompt = construct(high_chars)
    high_count = tokenize_count(base_url, high_prompt, timeout_s)
    while high_count < target_tokens:
        low_chars, low_prompt, low_count = high_chars, high_prompt, high_count
        high_chars *= 2
        high_prompt = construct(high_chars)
        high_count = tokenize_count(base_url, high_prompt, timeout_s)

    best_prompt, best_count, best_chars = low_prompt, low_count, low_chars
    for _ in range(32):
        if high_count == low_count:
            break
        estimated = low_chars + int(
            (target_tokens - low_count) * (high_chars - low_chars) / (high_count - low_count)
        )
        probe_chars = min(high_chars - 1, max(low_chars + 1, estimated))
        if probe_chars <= low_chars or probe_chars >= high_chars:
            probe_chars = (low_chars + high_chars) // 2
        if probe_chars <= low_chars:
            break
        prompt = construct(probe_chars)
        count = tokenize_count(base_url, prompt, timeout_s)
        if count == target_tokens:
            return prompt, count
        if count <= target_tokens:
            low_chars, low_prompt, low_count = probe_chars, prompt, count
            if count > best_count:
                best_prompt, best_count, best_chars = prompt, count, probe_chars
        else:
            high_chars, high_prompt, high_count = probe_chars, prompt, count
        if high_chars - low_chars <= 1:
            break

    # Token counts can jump at a corpus boundary.  Add a tiny, semantically inert
    # padding run before END REFERENCE and prove the final count instead of silently
    # accepting a prompt that is merely close to the requested capacity.
    for unit in (" benchmark", " x", " 0", "\npadding"):
        for repeats in range(1, 65):
            prompt = construct(best_chars, unit * repeats)
            count = tokenize_count(base_url, prompt, timeout_s)
            if count == target_tokens:
                return prompt, count
            if count > target_tokens + 8:
                break
    raise RuntimeError(
        f"could not construct an exact {target_tokens}-token prompt; closest count was {best_count}"
    )


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def latest_log_line(path: pathlib.Path, max_bytes: int = 16_384) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def merged_env(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for source in (config.get("defaults", {}).get("env", {}), experiment.get("env", {})):
        for key, value in source.items():
            env[str(key)] = str(value)
    return env


def merged_request(
    config: dict[str, Any], tier: dict[str, Any], experiment: dict[str, Any],
) -> dict[str, Any]:
    request = dict(config.get("defaults", {}).get("request", {}))
    request.update(tier.get("request", {}))
    request.update(experiment.get("request", {}))
    return request


def canonicalize_server_args(args: list[str]) -> list[str]:
    """Apply last-wins overrides without emitting contradictory singleton options."""
    last_position: dict[str, int] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token in SINGLE_VALUE_SERVER_OPTIONS and index + 1 < len(args):
            last_position[token] = index
            index += 2
        else:
            index += 1

    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in SINGLE_VALUE_SERVER_OPTIONS and index + 1 < len(args):
            if last_position[token] == index:
                result.extend((token, args[index + 1]))
            index += 2
        else:
            result.append(token)
            index += 1
    mutually_exclusive = (
        {"--kv-unified", "--no-kv-unified"},
        {"--cont-batching", "--no-cont-batching"},
        {"--context-shift", "--no-context-shift"},
        {"--mmproj-offload", "--no-mmproj-offload"},
    )
    for group in mutually_exclusive:
        last = max((index for index, token in enumerate(result) if token in group), default=-1)
        if last >= 0:
            result = [token for index, token in enumerate(result) if token not in group or index == last]
    return result


def translate_server_options(
    args: list[str], translations: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Translate version-specific singleton syntax while preserving its semantics."""
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in translations:
            if token not in SINGLE_VALUE_SERVER_OPTIONS or index + 1 >= len(args):
                raise ValueError(f"cannot translate non-singleton or valueless server option {token}")
            value = args[index + 1]
            choices = translations[token]
            if value not in choices:
                raise ValueError(f"no server-option translation for {token}={value}")
            result.extend(str(item) for item in choices[value])
            index += 2
            continue
        result.append(token)
        index += 1
    return result


def server_command(config: dict[str, Any], tier: dict[str, Any], experiment: dict[str, Any]) -> list[str]:
    defaults = config.get("defaults", {})
    host = str(defaults.get("host", "127.0.0.1"))
    port = int(defaults.get("port", 8189))
    command = [str(experiment["server"])]
    command.extend(str(item) for item in experiment.get("launcher_args", []))
    command.extend([
        "-m", str(experiment["model"]),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(int(tier.get("ctx_size", defaults.get("ctx_size", 8192)))),
    ])
    command.extend(str(item) for item in defaults.get("server_args", []))
    command.extend(str(item) for item in experiment.get("args", []))
    if bool(tier.get("erase_slot_between_requests", defaults.get("erase_slot_between_requests", False))):
        command.extend(["--slot-save-path", effective_slot_save_path(config, tier)])
    args = canonicalize_server_args(command[1:])
    translations = experiment.get("translate_server_options", {})
    return [command[0], *translate_server_options(args, translations)]


def server_arg_compatibility_errors(args: list[str]) -> list[str]:
    """Reject known llama.cpp option combinations before an expensive model load."""
    flash_attn = (option_value(args, "--flash-attn") or "auto").lower()
    cache_v = (option_value(args, "--cache-type-v") or "f16").lower()
    errors: list[str] = []
    if flash_attn == "off" and cache_v.startswith("q"):
        errors.append(
            f"quantized V cache ({cache_v}) requires flash attention; "
            "use F16/BF16 V cache or enable flash attention"
        )
    return errors


def effective_slot_save_path(config: dict[str, Any], tier: dict[str, Any]) -> str:
    defaults = config.get("defaults", {})
    configured = tier.get("slot_save_path", defaults.get("slot_save_path"))
    if configured:
        return str(configured)
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return str(pathlib.Path(tempfile.gettempdir()) / f"qwen-bench-slots-{uid}")


def port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            decoded = json.loads(body) if body else None
            return response.status, decoded
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            decoded = {"error": body}
        return exc.code, decoded


def process_metrics(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        status = pathlib.Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        status = ""
    for proc_name, result_name in {
        "VmRSS": "rss_bytes",
        "RssAnon": "rss_anon_bytes",
        "RssFile": "rss_file_bytes",
        "RssShmem": "rss_shmem_bytes",
    }.items():
        match = re.search(rf"^{proc_name}:\s+(\d+)\s+kB", status, re.MULTILINE)
        if match:
            result[result_name] = int(match.group(1)) * 1024

    try:
        io_text = pathlib.Path(f"/proc/{pid}/io").read_text(encoding="utf-8")
    except OSError:
        io_text = ""
    for proc_name, result_name in {
        "rchar": "rchar_bytes",
        "read_bytes": "read_bytes",
    }.items():
        match = re.search(rf"^{proc_name}:\s+(\d+)$", io_text, re.MULTILINE)
        if match:
            result[result_name] = int(match.group(1))

    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii").strip()
        # comm (field 2) is parenthesized and may contain spaces. Fields after
        # the final ')' begin at field 3 (state); minflt and majflt are 10/12.
        tail = stat[stat.rfind(")") + 2:].split()
        if len(tail) > 9:
            result["minor_faults"] = int(tail[7])
            result["major_faults"] = int(tail[9])
    except (OSError, ValueError):
        pass
    return result


def meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        text = pathlib.Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return result
    wanted = {"MemAvailable", "Cached", "SwapFree", "SwapTotal"}
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB", line)
        if match and match.group(1) in wanted:
            result[match.group(1)] = int(match.group(2)) * 1024
    return result


def read_int(path: pathlib.Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def parse_pcie_bw(text: str) -> dict[str, Any]:
    """Parse AMDGPU's `received sent max-payload-size` one-second counters."""
    values = re.findall(r"\d+", text)
    if len(values) < 3:
        return {}
    received, sent, mps = (int(value) for value in values[:3])
    return {
        "pcie_rx_messages": received,
        "pcie_tx_messages": sent,
        "pcie_mps_bytes": mps,
        # An estimate/upper bound: the kernel exposes message counts and MPS,
        # not the actual size of every TLP.
        "pcie_rx_est_bytes_s": received * mps,
        "pcie_tx_est_bytes_s": sent * mps,
    }


def read_hwmon(root: pathlib.Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for hwmon in root.glob("hwmon/hwmon*"):
        power = read_int(hwmon / "power1_average")
        temp = read_int(hwmon / "temp1_input")
        if power is not None:
            result["power_w"] = power / 1_000_000.0
        if temp is not None:
            result["temp_c"] = temp / 1_000.0
        if result:
            break
    return result


def discover_drm_cards() -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    drm_root = pathlib.Path("/sys/class/drm")
    if not drm_root.exists():
        return cards
    for card in sorted(drm_root.iterdir()):
        # `card[0-9]*` also matches connector symlinks such as card0-DP-1.
        # Only the canonical cardN entries own the device telemetry files.
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        if not device.exists():
            continue
        bdf = "unknown"
        try:
            uevent = (device / "uevent").read_text(encoding="utf-8", errors="replace")
            match = re.search(r"^PCI_SLOT_NAME=(.+)$", uevent, re.MULTILINE)
            if match:
                bdf = match.group(1)
        except OSError:
            pass
        cards.append({"card": card.name, "bdf": bdf, "path": str(device.resolve())})
    return cards


def parse_link_status(text: str) -> dict[str, Any]:
    matches = re.findall(r"LnkSta:\s+Speed\s+([^,\s]+).*?Width\s+(x\d+)", text)
    if not matches:
        return {}
    speed, width = matches[-1]
    result: dict[str, Any] = {"speed": speed, "width": width}
    speed_match = re.match(r"([0-9.]+)GT/s", speed)
    width_match = re.match(r"x(\d+)", width)
    if speed_match:
        result["speed_gt_s"] = float(speed_match.group(1))
    if width_match:
        result["width_lanes"] = int(width_match.group(1))
    return result


class TelemetrySampler:
    def __init__(self, path: pathlib.Path, pid: int, interval_s: float, pcie_bdf: str | None):
        self.path = path
        self.pid = pid
        self.interval_s = max(0.2, interval_s)
        self.pcie_bdf = pcie_bdf
        self.cards = discover_drm_cards()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(3.0, self.interval_s * 3))

    def mark(self) -> int:
        return len(self.samples)

    def slice(self, start: int) -> list[dict[str, Any]]:
        return list(self.samples[start:])

    def _pcie(self) -> dict[str, Any]:
        if not self.pcie_bdf or not shutil.which("lspci"):
            return {}
        commands = [["lspci", "-s", self.pcie_bdf, "-vv"]]
        # Some distributions restrict the extended PCI config space.  A
        # passwordless sudo rule may already exist, but -n guarantees the
        # benchmark can never pause waiting for a password.
        if shutil.which("sudo"):
            commands.append(["sudo", "-n", "lspci", "-s", self.pcie_bdf, "-vv"])
        for command in commands:
            try:
                completed = subprocess.run(
                    command, text=True, capture_output=True, timeout=3, check=False,
                )
                parsed = parse_link_status(completed.stdout)
                if parsed:
                    return parsed
            except (OSError, subprocess.TimeoutExpired):
                continue
        return {}

    def _sample(self) -> dict[str, Any]:
        process = process_metrics(self.pid)
        sample: dict[str, Any] = {
            "ts": utc_now(),
            "monotonic_s": time.monotonic(),
            "pid_rss_bytes": process.get("rss_bytes"),
            "process": process,
            "host": meminfo(),
            "gpus": {},
            "pcie": self._pcie(),
        }
        for card in self.cards:
            root = pathlib.Path(card["path"])
            fields = {
                "gpu_busy_percent": read_int(root / "gpu_busy_percent"),
                "vram_used_bytes": read_int(root / "mem_info_vram_used"),
                "vram_total_bytes": read_int(root / "mem_info_vram_total"),
                "gtt_used_bytes": read_int(root / "mem_info_gtt_used"),
                "gtt_total_bytes": read_int(root / "mem_info_gtt_total"),
            }
            try:
                fields.update(parse_pcie_bw((root / "pcie_bw").read_text(encoding="ascii")))
            except OSError:
                pass
            fields.update(read_hwmon(root))
            sample["gpus"][card["bdf"]] = {"card": card["card"], **fields}
        return sample

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                try:
                    sample = self._sample()
                    self.samples.append(sample)
                    handle.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
                    handle.flush()
                except Exception as exc:  # telemetry must never kill a model run
                    handle.write(json.dumps({"ts": utc_now(), "error": repr(exc)}) + "\n")
                    handle.flush()
                self._stop.wait(self.interval_s)


def aggregate_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(samples), "gpus": {}}
    gpu_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rss: list[float] = []
    process_values: dict[str, list[float]] = defaultdict(list)
    available: list[float] = []
    cached: list[float] = []
    link_speeds: list[float] = []
    link_widths: list[float] = []
    for sample in samples:
        if isinstance(sample.get("pid_rss_bytes"), (int, float)):
            rss.append(float(sample["pid_rss_bytes"]))
        for field, value in sample.get("process", {}).items():
            if isinstance(value, (int, float)):
                process_values[field].append(float(value))
        host_available = sample.get("host", {}).get("MemAvailable")
        if isinstance(host_available, (int, float)):
            available.append(float(host_available))
        host_cached = sample.get("host", {}).get("Cached")
        if isinstance(host_cached, (int, float)):
            cached.append(float(host_cached))
        for bdf, gpu in sample.get("gpus", {}).items():
            for field in (
                "gpu_busy_percent", "vram_used_bytes", "gtt_used_bytes", "power_w", "temp_c",
                "pcie_rx_est_bytes_s", "pcie_tx_est_bytes_s",
            ):
                value = gpu.get(field)
                if isinstance(value, (int, float)):
                    gpu_values[bdf][field].append(float(value))
        pcie = sample.get("pcie", {})
        if isinstance(pcie.get("speed_gt_s"), (int, float)):
            link_speeds.append(float(pcie["speed_gt_s"]))
        if isinstance(pcie.get("width_lanes"), (int, float)):
            link_widths.append(float(pcie["width_lanes"]))
    if rss:
        result["pid_rss_max_bytes"] = int(max(rss))
    for field in ("rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes"):
        values = process_values.get(field, [])
        if values:
            result[f"pid_{field.removesuffix('_bytes')}_max_bytes"] = int(max(values))
    for field in ("rchar_bytes", "read_bytes", "minor_faults", "major_faults"):
        values = process_values.get(field, [])
        if values:
            result[f"pid_{field}_delta"] = int(max(values) - min(values))
    if available:
        result["mem_available_min_bytes"] = int(min(available))
        result["mem_available_max_bytes"] = int(max(available))
    if cached:
        result["host_cached_min_bytes"] = int(min(cached))
        result["host_cached_max_bytes"] = int(max(cached))
    for bdf, fields in gpu_values.items():
        aggregate: dict[str, Any] = {}
        busy = fields.get("gpu_busy_percent", [])
        if busy:
            aggregate["busy_mean_percent"] = round(statistics.fmean(busy), 2)
            aggregate["busy_max_percent"] = round(max(busy), 2)
        for field in ("vram_used_bytes", "gtt_used_bytes"):
            values = fields.get(field, [])
            if values:
                aggregate[field.replace("used", "used_max")] = int(max(values))
        for field in ("power_w", "temp_c", "pcie_rx_est_bytes_s", "pcie_tx_est_bytes_s"):
            values = fields.get(field, [])
            if values:
                aggregate[f"{field}_mean"] = round(statistics.fmean(values), 3)
                aggregate[f"{field}_max"] = round(max(values), 3)
        result["gpus"][bdf] = aggregate
    if link_speeds:
        result["pcie_speed_gt_s_min"] = min(link_speeds)
        result["pcie_speed_gt_s_max"] = max(link_speeds)
    if link_widths:
        result["pcie_width_lanes_min"] = int(min(link_widths))
        result["pcie_width_lanes_max"] = int(max(link_widths))
    return result


class ManagedServer:
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        log_path: pathlib.Path,
        health_url: str,
        startup_timeout_s: float,
        telemetry_path: pathlib.Path,
        telemetry_interval_s: float,
        pcie_bdf: str | None,
        stop_event: threading.Event | None = None,
    ):
        self.command = command
        self.env = env
        self.log_path = log_path
        self.health_url = health_url
        self.startup_timeout_s = startup_timeout_s
        self.telemetry_path = telemetry_path
        self.telemetry_interval_s = telemetry_interval_s
        self.pcie_bdf = pcie_bdf
        self.stop_event = stop_event
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.telemetry: TelemetrySampler | None = None
        self.startup_seconds: float | None = None

    def start(self) -> None:
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.log_handle.write(f"# started {utc_now()}\n# {command_text(self.command)}\n")
        if HOST_CHECKPOINT_MARKER in self.env:
            self.log_handle.write(
                f"# {HOST_CHECKPOINT_MARKER}={self.env[HOST_CHECKPOINT_MARKER]}\n"
            )
        self.log_handle.flush()
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        self.telemetry = TelemetrySampler(
            self.telemetry_path, self.process.pid, self.telemetry_interval_s, self.pcie_bdf
        )
        self.telemetry.start()
        deadline = started + self.startup_timeout_s
        next_report = started + 30.0
        last_error = "health endpoint unavailable"
        while time.monotonic() < deadline:
            if self.stop_event and self.stop_event.is_set():
                raise KeyboardInterrupt
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with status {self.process.returncode}; see {self.log_path}")
            try:
                status, body = http_json("GET", self.health_url, None, timeout=2)
                if status == 200 and (not isinstance(body, dict) or body.get("status") in (None, "ok")):
                    self.startup_seconds = time.monotonic() - started
                    return
                last_error = f"health returned HTTP {status}: {body}"
            except Exception as exc:
                last_error = repr(exc)
            now = time.monotonic()
            if now >= next_report:
                try:
                    log_mib = self.log_path.stat().st_size / 1024**2
                except OSError:
                    log_mib = 0.0
                line = latest_log_line(self.log_path)
                detail = f"; latest: {line[:240]}" if line else ""
                print(
                    f"  waiting for server health: {now - started:.0f}s "
                    f"(log {log_mib:.2f} MiB){detail}",
                    flush=True,
                )
                next_report = now + 30.0
            time.sleep(1.0)
        raise TimeoutError(f"server did not become healthy in {self.startup_timeout_s}s: {last_error}")

    def stop(self) -> None:
        if self.telemetry:
            self.telemetry.stop()
        process = self.process
        if process and process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if self.log_handle:
            self.log_handle.write(f"# stopped {utc_now()}\n")
            self.log_handle.close()


def extract_timing(response: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings") or {}
    predicted_n = timings.get("predicted_n")
    draft_n = timings.get("draft_n")
    accepted_n = timings.get("draft_n_accepted")
    acceptance = None
    if isinstance(draft_n, (int, float)) and draft_n > 0 and isinstance(accepted_n, (int, float)):
        acceptance = float(accepted_n) / float(draft_n)
    return {
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": predicted_n,
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": draft_n,
        "draft_n_accepted": accepted_n,
        "draft_acceptance": acceptance,
    }


def completion_request(
    url: str,
    prompt: str,
    n_predict: int,
    timeout_s: float,
    extra: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 1,
        "cache_prompt": False,
    }
    payload.update(extra)
    started = time.monotonic()
    status, response = http_json("POST", url, payload, timeout=timeout_s)
    wall_ms = (time.monotonic() - started) * 1000.0
    if status != 200:
        raise RuntimeError(f"completion returned HTTP {status}: {response}")
    if not isinstance(response, dict):
        raise RuntimeError("completion response is not a JSON object")
    return response, wall_ms


def image_data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def vision_completion_request(
    url: str,
    prompt: str,
    image_path: pathlib.Path,
    n_predict: int,
    timeout_s: float,
    extra: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "seed": 1,
        "cache_prompt": False,
    }
    payload.update(extra)
    started = time.monotonic()
    status, response = http_json("POST", url, payload, timeout=timeout_s)
    wall_ms = (time.monotonic() - started) * 1000.0
    if status != 200:
        raise RuntimeError(f"vision completion returned HTTP {status}: {response}")
    if not isinstance(response, dict):
        raise RuntimeError("vision completion response is not a JSON object")
    return response, wall_ms


def response_text_parts(response: dict[str, Any]) -> tuple[str, str]:
    """Return final-answer and reasoning text without silently losing either."""
    direct = response.get("content")
    if isinstance(direct, str):
        return direct, ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            answer = str(message.get("content", "")) if isinstance(message.get("content"), str) else ""
            raw_reasoning = message.get("reasoning_content", message.get("reasoning", ""))
            reasoning = str(raw_reasoning) if isinstance(raw_reasoning, str) else ""
            return answer, reasoning
        if isinstance(choices[0].get("text"), str):
            return str(choices[0]["text"]), ""
    return "", ""


def response_content(response: dict[str, Any]) -> str:
    answer, reasoning = response_text_parts(response)
    return answer if answer else reasoning


def anchor_metrics(content: str, anchors: list[str]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", content).casefold()
    matched = [anchor for anchor in anchors if re.sub(r"\s+", " ", anchor).casefold() in normalized]
    return {
        "anchors": anchors,
        "anchors_matched": matched,
        "anchor_score": len(matched) / len(anchors) if anchors else None,
    }


def probability_metrics(response: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint optional llama-server n_probs output without inflating results.jsonl."""
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        return {}
    canonical = json.dumps(
        probabilities, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    first = probabilities[0] if isinstance(probabilities[0], dict) else {}
    candidates = (
        first.get("top_logprobs", first.get("probs", []))
        if isinstance(first, dict) else []
    )
    compact: list[dict[str, Any]] = []
    if isinstance(candidates, list):
        for candidate in candidates[:20]:
            if not isinstance(candidate, dict):
                continue
            compact.append({
                key: candidate[key]
                for key in ("id", "token", "tok_str", "logprob", "prob")
                if key in candidate
            })
    return {
        "probability_steps": len(probabilities),
        "probabilities_sha256": hashlib.sha256(canonical).hexdigest(),
        "first_token_probabilities": compact,
    }


def file_mark(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def file_since(path: pathlib.Path, offset: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def extract_vision_log_metrics(text: str) -> dict[str, Any]:
    def values(*patterns: str) -> list[float]:
        found: list[float] = []
        for pattern in patterns:
            found.extend(float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE))
        return found

    encoded = values(r"image(?:/slice| slice)? encoded in\s+([0-9.]+)\s*ms")
    decoded = values(r"image decoded[^\n]*?in\s+([0-9.]+)\s*ms")
    processed = values(r"image processed in\s+([0-9.]+)\s*ms")
    return {
        "image_encode_ms": sum(encoded) if encoded else None,
        "image_decode_ms": sum(decoded) if decoded else None,
        "image_process_ms": processed[-1] if processed else None,
        "mtp_vision_resync": MTP_VISION_RESYNC_MARKER in text,
    }


def response_slot_id(response: dict[str, Any], fallback: int = 0) -> int:
    """Return the llama-server slot used by a completion response."""
    for field in ("id_slot", "slot_id"):
        value = response.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    verbose = response.get("__verbose")
    if isinstance(verbose, dict):
        for field in ("id_slot", "slot_id"):
            value = verbose.get(field)
            if isinstance(value, int) and value >= 0:
                return value
    # Older server responses did not expose the slot id. The caller supplies
    # the deterministic lane index for parallel probes.
    return fallback


def erase_slot(base_url: str, slot_id: int, timeout_s: float) -> dict[str, Any]:
    """Erase target/draft KV state without disturbing model or OS page caches."""
    status, response = http_json(
        "POST", f"{base_url}/slots/{slot_id}?action=erase", None, timeout=min(timeout_s, 30.0),
    )
    if status != 200:
        raise RuntimeError(f"slot {slot_id} erase returned HTTP {status}: {response}")
    return response if isinstance(response, dict) else {"response": response}


def concurrent_metrics(responses: list[dict[str, Any]], wall_ms: float) -> dict[str, Any]:
    timings = [extract_timing(response) for response in responses]
    predicted_n = [float(item["predicted_n"]) for item in timings if isinstance(item.get("predicted_n"), (int, float))]
    predicted_ms = [float(item["predicted_ms"]) for item in timings if isinstance(item.get("predicted_ms"), (int, float))]
    prompt_n = [float(item["prompt_n"]) for item in timings if isinstance(item.get("prompt_n"), (int, float))]
    prompt_ms = [float(item["prompt_ms"]) for item in timings if isinstance(item.get("prompt_ms"), (int, float))]
    lane_decode = [
        float(item["predicted_per_second"])
        for item in timings if isinstance(item.get("predicted_per_second"), (int, float))
    ]
    lane_prefill = [
        float(item["prompt_per_second"])
        for item in timings if isinstance(item.get("prompt_per_second"), (int, float))
    ]
    return {
        "lanes": len(responses),
        "wall_ms": round(wall_ms, 3),
        "predicted_n_total": int(sum(predicted_n)) if predicted_n else None,
        "prompt_n_total": int(sum(prompt_n)) if prompt_n else None,
        "aggregate_decode_tok_s": (
            sum(predicted_n) * 1000.0 / sum(predicted_ms) if predicted_n and predicted_ms and sum(predicted_ms) > 0 else None
        ),
        "aggregate_prefill_tok_s": (
            sum(prompt_n) * 1000.0 / sum(prompt_ms) if prompt_n and prompt_ms and sum(prompt_ms) > 0 else None
        ),
        "overlap_upper_decode_tok_s": (
            sum(predicted_n) * 1000.0 / max(predicted_ms) if predicted_n and predicted_ms and max(predicted_ms) > 0 else None
        ),
        "overlap_upper_prefill_tok_s": (
            sum(prompt_n) * 1000.0 / max(prompt_ms) if prompt_n and prompt_ms and max(prompt_ms) > 0 else None
        ),
        "lane_decode_min_tok_s": min(lane_decode) if lane_decode else None,
        "lane_decode_max_tok_s": max(lane_decode) if lane_decode else None,
        "decode_fairness_ratio": min(lane_decode) / max(lane_decode) if lane_decode and max(lane_decode) > 0 else None,
        "lane_prefill_min_tok_s": min(lane_prefill) if lane_prefill else None,
        "lane_prefill_max_tok_s": max(lane_prefill) if lane_prefill else None,
        "prefill_fairness_ratio": min(lane_prefill) / max(lane_prefill) if lane_prefill and max(lane_prefill) > 0 else None,
        "end_to_end_output_tok_s": (
            sum(predicted_n) * 1000.0 / wall_ms if predicted_n and wall_ms > 0 else None
        ),
    }


def concurrent_completion_requests(
    url: str,
    prompts: list[str],
    n_predict: int,
    timeout_s: float,
    extra: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], float]:
    """Release independent HTTP requests together and retain per-lane latency."""
    barrier = threading.Barrier(len(prompts))

    def one(prompt: str) -> tuple[dict[str, Any], float]:
        barrier.wait(timeout=30.0)
        return completion_request(url, prompt, n_predict, timeout_s, extra)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        completed = list(pool.map(one, prompts))
    wall_ms = (time.monotonic() - started) * 1000.0
    return [item[0] for item in completed], [item[1] for item in completed], wall_ms


def run_capture(command: list[str], env: dict[str, str] | None = None, timeout: float = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=timeout, check=False
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": repr(exc)}


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_results(run_dir: pathlib.Path, output: pathlib.Path | None = None) -> pathlib.Path:
    """Create one portable archive containing the complete run and an integrity manifest."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"run directory has no manifest.json: {run_dir}")
    output = output.resolve() if output else run_dir.with_name(run_dir.name + ".tar.gz")
    if output == run_dir or output.is_relative_to(run_dir):
        raise ValueError("archive output must be outside the run directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    payload_files = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "archive-manifest.json"
    )
    payload = []
    total_bytes = 0
    for path in payload_files:
        size = path.stat().st_size
        total_bytes += size
        payload.append({
            "path": path.relative_to(run_dir).as_posix(),
            "size_bytes": size,
            "sha256": sha256_file(path),
        })
    archive_manifest = {
        "schema": 1,
        "created_at": utc_now(),
        "harness_version": VERSION,
        "run": run_dir.name,
        "payload_file_count": len(payload),
        "payload_total_bytes": total_bytes,
        "payload": payload,
    }
    atomic_json(run_dir / "archive-manifest.json", archive_manifest)
    files_to_add = sorted(
        path for path in run_dir.rglob("*") if path.is_file() and not path.is_symlink()
    )

    temporary = output.with_name(output.name + ".tmp")
    try:
        with tarfile.open(temporary, mode="w:gz", compresslevel=6) as archive:
            for path in files_to_add:
                archive.add(path, arcname=(pathlib.Path(run_dir.name) / path.relative_to(run_dir)).as_posix())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(output)
    assert digest is not None
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii",
    )
    return output


def rocm_build_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    server = pathlib.Path(str(config.get("variables", {}).get("hip_server", "")))
    build_root = server.parent.parent if server.name else pathlib.Path()
    candidates = [build_root / "bin" / "libggml-hip.so", build_root / "lib" / "libggml-hip.so"]
    library = next((path for path in candidates if path.is_file()), candidates[0])
    llama_candidates = [build_root / "bin" / "libllama.so", build_root / "lib" / "libllama.so"]
    llama_library = next((path for path in llama_candidates if path.is_file()), llama_candidates[0])
    common_candidates = [
        build_root / "bin" / "libllama-common.so",
        build_root / "lib" / "libllama-common.so",
    ]
    common_library = next((path for path in common_candidates if path.is_file()), common_candidates[0])
    test_binary = build_root / "bin" / "test-backend-ops"
    result: dict[str, Any] = {}
    for name, path in (
        ("server", server),
        ("hip_library", library),
        ("llama_library", llama_library),
        ("common_library", common_library),
        ("test_backend_ops", test_binary),
    ):
        result[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path),
        }
    if library.is_file():
        try:
            binary = library.read_bytes()
            result["hip_library"]["gfx_targets"] = sorted({
                item.decode("ascii") for item in re.findall(rb"gfx[0-9a-f]{3,5}[a-z]*", binary)
            })
        except OSError:
            result["hip_library"]["gfx_targets"] = []
    else:
        result["hip_library"]["gfx_targets"] = []
    if llama_library.is_file():
        try:
            result["llama_library"]["qwen4exp_mtp_marker"] = (
                QWEN4EXP_MTP_MARKER.encode("ascii") in llama_library.read_bytes()
            )
            result["llama_library"]["qwen4exp_mtp_scheduling_marker"] = (
                QWEN4EXP_MTP_SCHED_MARKER.encode("ascii") in llama_library.read_bytes()
            )
        except OSError:
            result["llama_library"]["qwen4exp_mtp_marker"] = False
            result["llama_library"]["qwen4exp_mtp_scheduling_marker"] = False
    else:
        result["llama_library"]["qwen4exp_mtp_marker"] = False
        result["llama_library"]["qwen4exp_mtp_scheduling_marker"] = False
    if common_library.is_file():
        try:
            result["common_library"]["host_checkpoint_marker"] = (
                HOST_CHECKPOINT_MARKER.encode("ascii") in common_library.read_bytes()
            )
        except OSError:
            result["common_library"]["host_checkpoint_marker"] = False
    else:
        result["common_library"]["host_checkpoint_marker"] = False
    if server.is_file():
        try:
            result["server"]["mtp_vision_resync_marker"] = (
                MTP_VISION_RESYNC_MARKER.encode("ascii") in server.read_bytes()
            )
            result["server"]["qwen4exp_vision_strict_marker"] = (
                QWEN4EXP_VISION_STRICT_MARKER.encode("ascii") in server.read_bytes()
            )
            result["server"]["qwen4exp_vision_checkpoint_marker"] = (
                QWEN4EXP_VISION_CHECKPOINT_MARKER.encode("ascii") in server.read_bytes()
            )
        except OSError:
            result["server"]["mtp_vision_resync_marker"] = False
            result["server"]["qwen4exp_vision_strict_marker"] = False
            result["server"]["qwen4exp_vision_checkpoint_marker"] = False
    else:
        result["server"]["mtp_vision_resync_marker"] = False
        result["server"]["qwen4exp_vision_strict_marker"] = False
        result["server"]["qwen4exp_vision_checkpoint_marker"] = False
    return result


def inspect_rocmfp4_sources(repo: pathlib.Path) -> dict[str, Any]:
    runtime_roots = [repo / "ggml" / "src" / "ggml-cuda", repo / "ggml" / "src" / "ggml-hip"]
    tokens = ("Q4_0_ROCMFP4", "Q4_0_ROCMFP4_FAST")
    hits: dict[str, list[str]] = {token: [] for token in tokens}
    checked = 0
    for root in runtime_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            checked += 1
            for token in tokens:
                if token in text:
                    hits[token].append(str(path.relative_to(repo)))
    return {
        "repo": str(repo),
        "runtime_source_files_checked": checked,
        "token_hits": {token: sorted(set(paths)) for token, paths in hits.items()},
        "static_dispatch_ready": all(hits[token] for token in tokens),
        "note": "The standalone ggml/rocmfp4 directory does not count; markers must exist in a compiled HIP/CUDA runtime source tree.",
    }


def inspect_qwen4exp_mtp_sources(repo: pathlib.Path) -> dict[str, Any]:
    required = {
        "src/models/qwen4exp.cpp": (
            "LLM_GRAPH_TYPE_DECODER_MTP",
            "llm_graph_input_embd_h",
            QWEN4EXP_MTP_MARKER,
            QWEN4EXP_MTP_SCHED_MARKER,
            "nextn.eh_proj",
            "t_h_pre_norm",
        ),
        "src/llama-model.cpp": (
            "LLM_ARCH_QWEN4EXP",
            "n_embd_pre_norm",
        ),
        "src/llama-arch.cpp": (
            "llm_arch_supports_rs_rollback",
            "LLM_ARCH_QWEN4EXP",
        ),
    }
    files: dict[str, Any] = {}
    ready = True
    for relative, markers in required.items():
        path = repo / relative
        text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
        found = {marker: marker in text for marker in markers}
        files[relative] = {"exists": path.is_file(), "markers": found}
        ready = ready and path.is_file() and all(found.values())
    return {"ready": ready, "files": files}


def inspect_host_checkpoint_source(repo: pathlib.Path) -> dict[str, Any]:
    path = repo / "common" / "common.cpp"
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    markers = {
        HOST_CHECKPOINT_MARKER: HOST_CHECKPOINT_MARKER in text,
        "clear_on_device_flag": "~LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in text,
        "host_checkpoint_log": "forcing checkpoint state to host memory" in text,
    }
    return {
        "ready": path.is_file() and all(markers.values()),
        "file": str(path),
        "markers": markers,
    }


def inspect_mtp_vision_resync_source(repo: pathlib.Path) -> dict[str, Any]:
    server_path = repo / "tools" / "server" / "server-context.cpp"
    speculative_path = repo / "common" / "speculative.cpp"
    server_text = server_path.read_text(encoding="utf-8", errors="ignore") if server_path.is_file() else ""
    speculative_text = (
        speculative_path.read_text(encoding="utf-8", errors="ignore")
        if speculative_path.is_file() else ""
    )
    markers = {
        MTP_VISION_RESYNC_MARKER: MTP_VISION_RESYNC_MARKER in server_text,
        "target_conditioned_draft": "target_conditioned_draft" in server_text,
        "missing_boundary_resync": "resyncing after a non-token batch" in speculative_text,
    }
    return {
        "ready": server_path.is_file() and speculative_path.is_file() and all(markers.values()),
        "server_file": str(server_path),
        "speculative_file": str(speculative_path),
        "markers": markers,
    }


def inspect_qwen4exp_vision_strict_source(repo: pathlib.Path) -> dict[str, Any]:
    server_path = repo / "tools" / "server" / "server-context.cpp"
    common_path = repo / "common" / "common.cpp"
    common_header_path = repo / "common" / "common.h"
    server_text = server_path.read_text(encoding="utf-8", errors="ignore") if server_path.is_file() else ""
    common_text = common_path.read_text(encoding="utf-8", errors="ignore") if common_path.is_file() else ""
    common_header_text = (
        common_header_path.read_text(encoding="utf-8", errors="ignore")
        if common_header_path.is_file() else ""
    )
    markers = {
        QWEN4EXP_VISION_STRICT_MARKER: QWEN4EXP_VISION_STRICT_MARKER in server_text,
        QWEN4EXP_VISION_CHECKPOINT_MARKER: QWEN4EXP_VISION_CHECKPOINT_MARKER in server_text,
        "strict_qwen4exp_vision_mtp_verification": "strict_qwen4exp_vision_mtp_verification" in server_text,
        "llama_decode_with_ubatch": "llama_decode_with_ubatch(ctx_tgt, batch_view, 1)" in server_text,
        "explicit_strict_flag": "mtp_strict_qwen4exp_vision" in common_header_text,
        "rollback_disabled": "mtp_strict_qwen4exp_vision ? 0" in common_text,
    }
    return {
        "ready": server_path.is_file() and common_path.is_file() and common_header_path.is_file() and all(markers.values()),
        "server_file": str(server_path),
        "common_file": str(common_path),
        "common_header_file": str(common_header_path),
        "markers": markers,
    }


def backend_ops_passed(capture: dict[str, Any]) -> tuple[bool, int, int]:
    if capture.get("returncode") != 0:
        return False, 0, 0
    output = str(capture.get("stdout", "")) + "\n" + str(capture.get("stderr", ""))
    matches = re.findall(r"(\d+)\s*/\s*(\d+)\s+tests passed", output)
    if not matches:
        return False, 0, 0
    passed, total = (int(value) for value in matches[-1])
    return total > 0 and passed == total and "FAIL" not in output, passed, total


def run_rocm_audit(
    config: dict[str, Any],
    output: pathlib.Path,
    run_ops: bool,
    devices: list[str],
    timeout_s: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / "rocm-audit-logs"
    logs.mkdir(exist_ok=True)
    variables = config.get("variables", {})
    repo = pathlib.Path(str(variables.get("hip_repo") or variables.get("repo", "")))
    fingerprint = rocm_build_fingerprint(config)
    source = inspect_rocmfp4_sources(repo)
    qwen4exp_mtp_source = inspect_qwen4exp_mtp_sources(repo)
    host_checkpoint_source = inspect_host_checkpoint_source(repo)
    mtp_vision_resync_source = inspect_mtp_vision_resync_source(repo)
    qwen4exp_vision_strict_source = inspect_qwen4exp_vision_strict_source(repo)
    server = pathlib.Path(fingerprint["server"]["path"])
    test_binary = pathlib.Path(fingerprint["test_backend_ops"]["path"])
    gfx_targets = set(fingerprint["hip_library"].get("gfx_targets", []))
    dual_arch_ready = {"gfx1100", "gfx1151"}.issubset(gfx_targets)
    env = os.environ.copy()
    env.update({
        "ROCM_PATH": "/opt/rocm-10.0.0",
        "HIP_PATH": "/opt/rocm-10.0.0",
        "LD_LIBRARY_PATH": "/opt/rocm-10.0.0/lib:/opt/rocm-10.0.0/lib/rocm_sysdeps/lib",
    })
    device_capture = run_capture([str(server), "--list-devices"], env=env, timeout=60) if server.is_file() else {
        "command": [str(server), "--list-devices"], "error": "missing HIP llama-server",
    }
    write_capture(logs / "devices.txt", device_capture)
    tests: list[dict[str, Any]] = []
    if run_ops and test_binary.is_file():
        for device in devices:
            for operation in ("MUL_MAT", "MUL_MAT_ID"):
                quant_types = ["q8_0"]
                if source["static_dispatch_ready"]:
                    quant_types.extend(("q4_0_rocmfp4", "q4_0_rocmfp4_fast"))
                for quant_type in quant_types:
                    command = [
                        str(test_binary), "test", "-b", device, "-o", operation,
                        "-p", f"type_a={quant_type}",
                    ]
                    capture = run_capture(command, env=env, timeout=timeout_s)
                    passed, passed_n, total_n = backend_ops_passed(capture)
                    log_name = safe_name(f"{device}-{operation}-{quant_type}") + ".txt"
                    write_capture(logs / log_name, capture)
                    tests.append({
                        "gate": "standard_control" if quant_type == "q8_0" else "custom_rocmfp4",
                        "device": device,
                        "operation": operation,
                        "type_a": quant_type,
                        "passed": passed,
                        "passed_n": passed_n,
                        "total_n": total_n,
                        "returncode": capture.get("returncode"),
                        "error": capture.get("error"),
                        "log": str(pathlib.Path("rocm-audit-logs") / log_name),
                    })
    standard_tests = [item for item in tests if item["gate"] == "standard_control"]
    custom_tests = [item for item in tests if item["gate"] == "custom_rocmfp4"]
    expected_standard = len(devices) * 2
    expected_custom = len(devices) * 2 * 2
    standard_control_pass = (
        run_ops and len(standard_tests) == expected_standard and expected_standard > 0
        and all(item["passed"] for item in standard_tests)
    )
    custom_functional_pass = (
        run_ops and source["static_dispatch_ready"] and len(custom_tests) == expected_custom
        and expected_custom > 0 and all(item["passed"] for item in custom_tests)
    )
    functional_pass = standard_control_pass and custom_functional_pass
    compiled_qwen4exp_mtp = bool(
        fingerprint.get("llama_library", {}).get("qwen4exp_mtp_marker") and
        fingerprint.get("llama_library", {}).get("qwen4exp_mtp_scheduling_marker")
    )
    ready_for_model_benchmarks = bool(
        source["static_dispatch_ready"] and dual_arch_ready and functional_pass
    )
    ready_for_mtp_benchmarks = bool(
        ready_for_model_benchmarks and qwen4exp_mtp_source["ready"] and compiled_qwen4exp_mtp
    )
    compiled_host_checkpoints = bool(
        fingerprint.get("common_library", {}).get("host_checkpoint_marker")
    )
    ready_for_host_checkpoint_benchmarks = bool(
        ready_for_model_benchmarks and host_checkpoint_source["ready"] and compiled_host_checkpoints
    )
    compiled_mtp_vision_resync = bool(
        fingerprint.get("server", {}).get("mtp_vision_resync_marker")
    )
    compiled_qwen4exp_vision_strict = bool(
        fingerprint.get("server", {}).get("qwen4exp_vision_strict_marker") and
        fingerprint.get("server", {}).get("qwen4exp_vision_checkpoint_marker")
    )
    ready_for_mtp_vision_benchmarks = bool(
        ready_for_mtp_benchmarks and
        mtp_vision_resync_source["ready"] and compiled_mtp_vision_resync and
        qwen4exp_vision_strict_source["ready"] and compiled_qwen4exp_vision_strict
    )
    reasons: list[str] = []
    mtp_reasons: list[str] = []
    host_checkpoint_reasons: list[str] = []
    mtp_vision_reasons: list[str] = []
    if not source["static_dispatch_ready"]:
        reasons.append("ROCmFP4 and ROCmFP4_FAST are not both wired into the compiled HIP runtime source tree")
    if not test_binary.is_file():
        reasons.append(f"{test_binary} is missing; rebuild with tests enabled")
    if not dual_arch_ready:
        reasons.append("libggml-hip.so does not contain both gfx1100 and gfx1151 code objects")
    if not run_ops:
        reasons.append("functional backend-op tests were not requested; rerun with --run-ops after source integration")
    elif not standard_control_pass:
        reasons.append("ordinary Q8_0 MUL_MAT/MUL_MAT_ID did not pass on both GPUs; fix the ROCm runtime/build first")
    if run_ops and source["static_dispatch_ready"] and not custom_functional_pass:
        reasons.append("custom ROCmFP4 MUL_MAT/MUL_MAT_ID failed or matched zero cases on one or both GPUs")
    if not qwen4exp_mtp_source["ready"]:
        mtp_reasons.append("the pinned source tree lacks the complete qwen4exp MTP sidecar integration")
    if not compiled_qwen4exp_mtp:
        mtp_reasons.append("libllama.so lacks the compiled qwen4exp MTP integration or hidden-state scheduling marker; rebuild with build-rocm10-dual.sh")
    if not host_checkpoint_source["ready"]:
        host_checkpoint_reasons.append(
            "the pinned source tree lacks the LLAMA_CKPT_FORCE_HOST checkpoint-to-host integration"
        )
    if not compiled_host_checkpoints:
        host_checkpoint_reasons.append(
            "libllama-common.so lacks the compiled LLAMA_CKPT_FORCE_HOST marker; rebuild with build-rocm10-dual.sh"
        )
    if not mtp_vision_resync_source["ready"]:
        mtp_vision_reasons.append(
            "the pinned source tree lacks the MTP multimodal-resync integration"
        )
    if not compiled_mtp_vision_resync:
        mtp_vision_reasons.append(
            "llama-server lacks the compiled MTP multimodal-resync marker; rebuild with build-rocm10-dual.sh"
        )
    if not qwen4exp_vision_strict_source["ready"]:
        mtp_vision_reasons.append(
            "the pinned source tree lacks Qwen4Exp vision single-row MTP verification"
        )
    if not compiled_qwen4exp_vision_strict:
        mtp_vision_reasons.append(
            "llama-server lacks the compiled Qwen4Exp checkpoint-backed vision strict-verification markers; rebuild with build-rocm10-dual.sh"
        )
    if not ready_for_mtp_benchmarks:
        mtp_vision_reasons.append(
            "the prerequisite qwen4exp MTP audit gate has not passed"
        )
    report = {
        "schema": 8,
        "ts": utc_now(),
        "harness_version": VERSION,
        "ready_for_model_benchmarks": ready_for_model_benchmarks,
        "ready_for_mtp_benchmarks": ready_for_mtp_benchmarks,
        "ready_for_host_checkpoint_benchmarks": ready_for_host_checkpoint_benchmarks,
        "ready_for_mtp_vision_benchmarks": ready_for_mtp_vision_benchmarks,
        "source": source,
        "qwen4exp_mtp_source": qwen4exp_mtp_source,
        "host_checkpoint_source": host_checkpoint_source,
        "mtp_vision_resync_source": mtp_vision_resync_source,
        "qwen4exp_vision_strict_source": qwen4exp_vision_strict_source,
        "build_fingerprint": fingerprint,
        "devices_requested": devices,
        "device_list_log": str(pathlib.Path("rocm-audit-logs") / "devices.txt"),
        "functional_tests_requested": run_ops,
        "dual_arch_ready": dual_arch_ready,
        "standard_control_pass": standard_control_pass,
        "custom_functional_pass": custom_functional_pass,
        "functional_pass": functional_pass,
        "tests": tests,
        "reasons": reasons,
        "mtp_reasons": mtp_reasons,
        "host_checkpoint_reasons": host_checkpoint_reasons,
        "mtp_vision_reasons": mtp_vision_reasons,
    }
    atomic_json(output, report)
    return report


def validate_rocm_audit(
    config: dict[str, Any], *, require_mtp: bool = False, require_host_checkpoints: bool = False,
    require_mtp_vision: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    configured = config.get("variables", {}).get("rocm_audit")
    path = pathlib.Path(str(configured)) if configured else pathlib.Path(__file__).with_name("preflight") / "rocm-audit.json"
    if not path.is_file():
        return None, f"ROCm audit is missing at {path}; run `python3 qwen_bench.py rocm-audit --run-ops`"
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"cannot read ROCm audit {path}: {exc}"
    if not report.get("ready_for_model_benchmarks"):
        reasons = "; ".join(str(item) for item in report.get("reasons", []))
        return report, f"ROCm audit has not proven custom kernel coverage: {reasons}"
    current = rocm_build_fingerprint(config)
    audited = report.get("build_fingerprint", {})
    fingerprint_keys = ["server", "hip_library", "llama_library"]
    if require_mtp_vision:
        require_mtp = True
    if require_host_checkpoints:
        fingerprint_keys.append("common_library")
    for key in fingerprint_keys:
        expected = audited.get(key, {}).get("sha256")
        actual = current.get(key, {}).get("sha256")
        if not expected or expected != actual:
            return report, f"ROCm audit is stale because {key} changed; rerun `python3 qwen_bench.py rocm-audit --run-ops`"
    if require_mtp and int(report.get("schema", 0)) < 4:
        return report, "ROCm audit predates the hidden-state scheduling gate; rebuild and rerun `python3 qwen_bench.py rocm-audit --run-ops`"
    if require_mtp and not current.get("llama_library", {}).get("qwen4exp_mtp_scheduling_marker"):
        return report, "current libllama.so lacks the qwen4exp hidden-state scheduling fix; rerun `./build-rocm10-dual.sh` and the ROCm audit"
    if require_mtp and not report.get("ready_for_mtp_benchmarks"):
        reasons = "; ".join(str(item) for item in report.get("mtp_reasons", []))
        return report, f"ROCm audit has not proven qwen4exp MTP support: {reasons}"
    if require_host_checkpoints and int(report.get("schema", 0)) < 5:
        return report, "ROCm audit predates the host-checkpoint gate; rebuild and rerun `python3 qwen_bench.py rocm-audit --run-ops`"
    if require_host_checkpoints and not current.get("common_library", {}).get("host_checkpoint_marker"):
        return report, "current libllama-common.so lacks LLAMA_CKPT_FORCE_HOST; rerun `./build-rocm10-dual.sh` and the ROCm audit"
    if require_host_checkpoints and not report.get("ready_for_host_checkpoint_benchmarks"):
        reasons = "; ".join(str(item) for item in report.get("host_checkpoint_reasons", []))
        return report, f"ROCm audit has not proven host-resident prompt checkpoints: {reasons}"
    if require_mtp_vision and int(report.get("schema", 0)) < 8:
        return report, "ROCm audit predates checkpoint-backed Qwen4Exp vision strict verification; rebuild and rerun `python3 qwen_bench.py rocm-audit --run-ops`"
    if require_mtp_vision and not current.get("server", {}).get("mtp_vision_resync_marker"):
        return report, "current llama-server lacks the MTP multimodal-resync fix; rerun `./build-rocm10-dual.sh` and the ROCm audit"
    if require_mtp_vision and not current.get("server", {}).get("qwen4exp_vision_strict_marker"):
        return report, "current llama-server lacks Qwen4Exp vision strict verification; rerun `./build-rocm10-dual.sh` and the ROCm audit"
    if require_mtp_vision and not current.get("server", {}).get("qwen4exp_vision_checkpoint_marker"):
        return report, "current llama-server lacks checkpoint-backed Qwen4Exp vision verification; rerun `./build-rocm10-dual.sh` and the ROCm audit"
    if require_mtp_vision and not report.get("ready_for_mtp_vision_benchmarks"):
        reasons = "; ".join(str(item) for item in report.get("mtp_vision_reasons", []))
        return report, f"ROCm audit has not proven MTP plus vision support: {reasons}"
    return report, None


def write_capture(path: pathlib.Path, capture: dict[str, Any]) -> None:
    lines = [f"$ {command_text([str(x) for x in capture['command']])}\n"]
    if "error" in capture:
        lines.append(f"ERROR: {capture['error']}\n")
    else:
        lines.append(f"exit={capture['returncode']}\n")
        lines.append(capture.get("stdout", ""))
        if capture.get("stderr"):
            lines.append("\n[stderr]\n")
            lines.append(capture["stderr"])
    path.write_text("".join(lines), encoding="utf-8")


def active_llama_processes() -> list[str]:
    if os.name == "nt" or not shutil.which("pgrep"):
        return []
    capture = run_capture(["pgrep", "-af", "llama-(server|cli|bench)"])
    if capture.get("returncode") not in (0, 1):
        return []
    own_pid = str(os.getpid())
    return [line for line in capture.get("stdout", "").splitlines() if line and not line.startswith(own_pid + " ")]


def preflight(
    config: dict[str, Any],
    tier: dict[str, Any],
    experiments: list[dict[str, Any]],
    run_dir: pathlib.Path,
    allow_busy: bool,
    skip_path_check: bool,
) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    host = str(defaults.get("host", "127.0.0.1"))
    port = int(defaults.get("port", 8189))
    errors: list[str] = []
    warnings: list[str] = []
    rocm_audit: dict[str, Any] | None = None
    files: list[dict[str, Any]] = []
    mode = str(tier.get("mode", "text"))
    require_host_checkpoints = any(
        HOST_CHECKPOINT_MARKER in experiment.get("env", {}) for experiment in experiments
    )
    vision_mode = mode == "vision"
    quality_mode = mode == "quality"
    vision_cases: dict[str, dict[str, Any]] = {}
    if vision_mode:
        try:
            vision_cases = load_vision_cases(config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load vision cases: {exc}")
        selected_cases = list(tier.get("vision_cases", vision_cases))
        missing_cases = sorted(set(selected_cases) - set(vision_cases))
        if missing_cases:
            errors.append(f"vision tier references unknown cases: {missing_cases}")
        for case_name in selected_cases:
            case = vision_cases.get(case_name)
            if not case:
                continue
            image = pathlib.Path(str(case["image"]))
            exists = image.is_file()
            files.append({"vision_case": case_name, "kind": "image", "path": str(image), "exists": exists})
            if not skip_path_check and not exists:
                errors.append(f"vision case {case_name}: missing image {image}; run python3 qwen_vision.py fixtures")
    if quality_mode:
        try:
            _quality_filler, quality_cases = load_quality_cases(config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            quality_cases = {}
            errors.append(f"cannot load quality cases: {exc}")
        selected_quality_cases = list(tier.get("quality_cases", quality_cases))
        missing_quality_cases = sorted(set(selected_quality_cases) - set(quality_cases))
        if missing_quality_cases:
            errors.append(f"quality tier references unknown cases: {missing_quality_cases}")
        files.append({
            "kind": "quality_cases",
            "path": str(config.get("quality_cases_file", "")),
            "exists": bool(quality_cases),
            "selected": selected_quality_cases,
        })
    for experiment in experiments:
        effective_args = server_command(config, tier, experiment)[1:]
        effective_request = merged_request(config, tier, experiment)
        for error in server_arg_compatibility_errors(effective_args):
            errors.append(f"{experiment['name']}: {error}")
        if experiment.get("require_zero_draft"):
            if "-md" not in effective_args:
                errors.append(f"{experiment['name']}: require_zero_draft requires a loaded draft model")
            if effective_request.get("speculative.n_max") != 0:
                errors.append(
                    f"{experiment['name']}: require_zero_draft requires request speculative.n_max=0"
                )
        if HOST_CHECKPOINT_MARKER in experiment.get("env", {}):
            if option_value(effective_args, "--checkpoint-every-n-tokens") is None:
                errors.append(
                    f"{experiment['name']}: host-checkpoint experiments require "
                    "--checkpoint-every-n-tokens"
                )
        for kind in ("server", "model"):
            path = pathlib.Path(str(experiment[kind]))
            entry = {"experiment": experiment["name"], "kind": kind, "path": str(path)}
            entry["exists"] = path.exists()
            if path.exists() and path.is_file():
                entry["size_bytes"] = path.stat().st_size
            files.append(entry)
            if not skip_path_check and not path.exists():
                errors.append(f"{experiment['name']}: missing {kind} {path}")
        if "-md" in experiment.get("args", []):
            args = experiment["args"]
            sidecar = pathlib.Path(str(args[args.index("-md") + 1]))
            exists = sidecar.exists()
            files.append({"experiment": experiment["name"], "kind": "draft_model", "path": str(sidecar), "exists": exists})
            if not skip_path_check and not exists:
                errors.append(f"{experiment['name']}: missing draft model {sidecar}")
        if vision_mode:
            experiment_args = list(experiment.get("args", []))
            mmproj_value = option_value(experiment_args, "--mmproj") or option_value(experiment_args, "-mm")
            if not mmproj_value:
                errors.append(f"{experiment['name']}: vision experiment has no --mmproj")
            else:
                mmproj = pathlib.Path(mmproj_value)
                exists = mmproj.is_file()
                files.append({"experiment": experiment["name"], "kind": "mmproj", "path": str(mmproj), "exists": exists})
                if not skip_path_check and not exists:
                    errors.append(
                        f"{experiment['name']}: missing projector {mmproj}; "
                        "run python3 qwen_vision.py projectors"
                    )
    if not port_available(host, port):
        errors.append(f"{host}:{port} is already in use")
    active = active_llama_processes()
    if active and not allow_busy:
        errors.append("existing llama process(es) detected; stop them or pass --allow-busy: " + " | ".join(active))
    elif active:
        warnings.append("GPU benchmark may be contaminated by active processes: " + " | ".join(active))
    available = meminfo().get("MemAvailable")
    if available is not None and available < int(defaults.get("warn_mem_available_bytes", 32 * 1024**3)):
        warnings.append(f"MemAvailable is only {available / 1024**3:.1f} GiB before model load")
    cache_state = str(tier.get("cache_state", "unspecified"))
    slot_save_path: str | None = None
    if cache_state not in {"unspecified", "hot", "cold"}:
        errors.append(f"unsupported cache_state {cache_state!r}; expected hot, cold, or unspecified")
    if cache_state == "hot":
        if int(tier.get("warmups", 0)) < 1 or (
            not vision_mode and int(tier.get("warmup_depth", 0)) <= 0
        ):
            errors.append("hot text tiers require at least one nonzero-depth warm-up; vision tiers require one image warm-up")
        if not bool(tier.get("erase_slot_between_requests", defaults.get("erase_slot_between_requests", False))):
            errors.append("hot tiers must erase slot KV state between warm-up and measured requests")
    if cache_state == "cold" and int(tier.get("warmups", 0)) != 0:
        errors.append("cold tiers must set warmups to 0")
    concurrency = int(tier.get("concurrency", 1))
    if concurrency < 1:
        errors.append("concurrency must be at least 1")
    if quality_mode:
        if concurrency != 1:
            errors.append("deterministic quality tiers currently require concurrency 1")
        if not bool(tier.get("exact_prompt_tokens")):
            errors.append("deterministic quality tiers require exact_prompt_tokens=true")
        if any(int(value) <= 0 for value in tier.get("depths", [])):
            errors.append("deterministic quality tiers require positive exact prompt depths")
        if int(tier.get("n_predict", 0)) < 128:
            errors.append("deterministic quality tiers require n_predict >= 128 to avoid truncating reasoning")
        quality_request = dict(defaults.get("request", {}))
        quality_request.update(tier.get("request", {}))
        if quality_request.get("ignore_eos") is not False:
            errors.append(
                "deterministic quality tiers require request.ignore_eos=false so exact answers are not forced to continue"
            )
    if concurrency > 1:
        workloads = list(tier.get("workloads", []))
        if len(workloads) != concurrency:
            errors.append(
                f"concurrent tiers require exactly one workload per lane; got {len(workloads)} workloads for {concurrency} lanes"
            )
        for experiment in experiments:
            parallel = option_value(server_command(config, tier, experiment)[1:], "--parallel")
            if parallel != str(concurrency):
                errors.append(
                    f"{experiment['name']}: tier concurrency is {concurrency}, but effective --parallel is {parallel}"
                )
    text_quality_anchors = tier.get("text_quality_anchors")
    if text_quality_anchors is not None:
        if vision_mode or quality_mode or not isinstance(text_quality_anchors, dict):
            errors.append("text_quality_anchors must be a workload-to-anchor-list mapping on a text tier")
        else:
            selected_workloads = set(str(value) for value in tier.get("workloads", []))
            unknown = sorted(set(text_quality_anchors) - selected_workloads)
            if unknown:
                errors.append(f"text quality anchors reference unselected workloads: {unknown}")
            for workload, anchors in text_quality_anchors.items():
                if not isinstance(anchors, list) or not anchors or not all(
                    isinstance(anchor, str) and anchor for anchor in anchors
                ):
                    errors.append(f"text quality anchors for {workload!r} must be a non-empty string list")
        minimum = tier.get("text_quality_min_anchor_score")
        if not isinstance(minimum, (int, float)) or not 0.0 <= float(minimum) <= 1.0:
            errors.append("text_quality_min_anchor_score must be between 0 and 1")
    if tier.get("require_probability_metrics"):
        request = dict(defaults.get("request", {}))
        request.update(tier.get("request", {}))
        if not isinstance(request.get("n_probs"), int) or int(request["n_probs"]) < 1:
            errors.append("require_probability_metrics requires request.n_probs >= 1")
    if vision_mode:
        if concurrency != 1:
            errors.append("vision tiers currently require concurrency 1")
        if [int(value) for value in tier.get("depths", [0])] != [0]:
            errors.append("vision tiers currently require depths: [0]")
        if tier.get("exact_prompt_tokens"):
            errors.append("vision tiers cannot use exact_prompt_tokens")
        request = dict(defaults.get("request", {}))
        request.update(tier.get("request", {}))
        chat_kwargs = request.get("chat_template_kwargs")
        if not isinstance(chat_kwargs, dict) or chat_kwargs.get("enable_thinking") is not False:
            errors.append(
                "vision tiers require request.chat_template_kwargs.enable_thinking=false "
                "so the measured decode budget reaches a final answer"
            )
        min_anchor_score = tier.get("vision_min_anchor_score")
        if not isinstance(min_anchor_score, (int, float)) or not 0.0 <= float(min_anchor_score) <= 1.0:
            errors.append("vision tiers require vision_min_anchor_score between 0 and 1")
        if not skip_path_check:
            checked_servers: set[str] = set()
            for experiment in experiments:
                server_path = str(experiment["server"])
                if server_path in checked_servers or not pathlib.Path(server_path).is_file():
                    continue
                checked_servers.add(server_path)
                capture = run_capture([server_path, "--help"], env=merged_env(config, experiment), timeout=60)
                help_text = str(capture.get("stdout", "")) + str(capture.get("stderr", ""))
                if capture.get("returncode") != 0 or "--mmproj" not in help_text:
                    errors.append(f"{server_path}: build does not advertise --mmproj support")
    if require_host_checkpoints and not skip_path_check:
        checked_servers: set[str] = set()
        for experiment in experiments:
            server_path = str(experiment["server"])
            if server_path in checked_servers or not pathlib.Path(server_path).is_file():
                continue
            checked_servers.add(server_path)
            capture = run_capture([server_path, "--help"], env=merged_env(config, experiment), timeout=60)
            help_text = str(capture.get("stdout", "")) + str(capture.get("stderr", ""))
            missing = [
                option for option in ("--ctx-checkpoints", "--checkpoint-every-n-tokens")
                if option not in help_text
            ]
            if capture.get("returncode") != 0 or missing:
                errors.append(
                    f"{server_path}: build does not advertise required checkpoint options: "
                    + ", ".join(missing)
                )
    if tier.get("startup_only") and (int(tier.get("warmups", 0)) != 0 or concurrency != 1):
        errors.append("startup-only tiers require warmups 0 and concurrency 1")
    if tier.get("require_rocm_audit") or require_host_checkpoints or tier.get("require_rocm_mtp_vision"):
        rocm_audit, audit_error = validate_rocm_audit(
            config,
            require_mtp=bool(tier.get("require_rocm_mtp")),
            require_host_checkpoints=require_host_checkpoints,
            require_mtp_vision=bool(tier.get("require_rocm_mtp_vision")),
        )
        if audit_error:
            errors.append(audit_error)
    if bool(tier.get("erase_slot_between_requests", defaults.get("erase_slot_between_requests", False))):
        slot_save_path = effective_slot_save_path(config, tier)
        try:
            pathlib.Path(slot_save_path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create slot-save path {slot_save_path}: {exc}")
    report = {
        "ts": utc_now(),
        "version": VERSION,
        "host": host,
        "port": port,
        "tier": tier,
        "cache_state": cache_state,
        "slot_save_path": slot_save_path,
        "experiments": [item["name"] for item in experiments],
        "files": files,
        "drm_cards": discover_drm_cards(),
        "active_llama_processes": active,
        "rocm_audit": rocm_audit,
        "require_host_checkpoints": require_host_checkpoints,
        "warnings": warnings,
        "errors": errors,
    }
    atomic_json(run_dir / "preflight.json", report)
    if errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))
    return report


def capture_system(config: dict[str, Any], experiments: list[dict[str, Any]], run_dir: pathlib.Path) -> None:
    target = run_dir / "system"
    target.mkdir(exist_ok=True)
    commands = [
        ("uname.txt", ["uname", "-a"], 10),
        ("os-release.txt", ["cat", "/etc/os-release"], 10),
        ("lspci.txt", ["lspci", "-nn"], 20),
        ("vulkan-summary.txt", ["vulkaninfo", "--summary"], 30),
        ("amd-smi-version.txt", ["amd-smi", "version"], 20),
        ("amd-smi-list.txt", ["amd-smi", "list"], 20),
        ("amd-smi-topology.txt", ["amd-smi", "topology"], 30),
        ("rocminfo.txt", ["rocminfo"], 60),
    ]
    for filename, command, timeout in commands:
        write_capture(target / filename, run_capture(command, timeout=timeout))
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for experiment in experiments:
        server = str(experiment["server"])
        env = merged_env(config, experiment)
        env_identity = tuple(sorted((key, value) for key, value in experiment.get("env", {}).items()))
        identity = (server, env_identity)
        if identity in seen or not pathlib.Path(server).exists():
            continue
        seen.add(identity)
        filename = f"devices-{safe_name(pathlib.Path(server).parent.parent.name)}-{len(seen)}.txt"
        device_command = [server]
        device_command.extend(str(item) for item in experiment.get("launcher_args", []))
        device_command.append("--list-devices")
        write_capture(target / filename, run_capture(device_command, env=env, timeout=60))
        repo = pathlib.Path(server).resolve().parent.parent.parent
        if (repo / ".git").exists():
            git = run_capture(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=10)
            write_capture(target / f"git-{safe_name(repo.name)}.txt", git)


def probe_row(
    run_id: str,
    experiment: dict[str, Any],
    round_index: int,
    workload: str,
    requested_depth: int,
    n_predict: int,
    response: dict[str, Any],
    wall_ms: float,
    startup_seconds: float | None,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    answer, reasoning = response_text_parts(response)
    content = answer if answer else reasoning
    timing = extract_timing(response)
    predicted_n = timing.get("predicted_n")
    chat_response = isinstance(response.get("choices"), list)
    answer_missing = chat_response and not answer.strip()
    degenerate = (
        not isinstance(predicted_n, (int, float))
        or predicted_n < n_predict * 0.95
        or answer_missing
    )
    return {
        "schema": 1,
        "status": "ok",
        "ts": utc_now(),
        "run_id": run_id,
        "experiment": experiment["name"],
        "backend": experiment.get("backend"),
        "round": round_index,
        "workload": workload,
        "requested_depth_tokens": requested_depth,
        "n_predict_requested": n_predict,
        "http_wall_ms": round(wall_ms, 3),
        "server_startup_seconds": startup_seconds,
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "output_chars": len(content),
        "answer_chars": len(answer),
        "reasoning_chars": len(reasoning),
        "answer_missing": answer_missing,
        "stop": response.get("stop"),
        "stopped_eos": response.get("stopped_eos"),
        "stopped_limit": response.get("stopped_limit"),
        "degenerate": degenerate,
        "timing": timing,
        "telemetry": telemetry,
        **probability_metrics(response),
    }


def completed_keys(results_path: pathlib.Path) -> set[tuple[int, str, str, int]]:
    keys: set[tuple[int, str, str, int]] = set()
    for row in read_jsonl(results_path):
        is_scored_quality = isinstance(row.get("quality_failure"), bool)
        if row.get("status") == "ok" and (not row.get("degenerate") or is_scored_quality):
            keys.add((int(row["round"]), str(row["experiment"]), str(row["workload"]), int(row["requested_depth_tokens"])))
    return keys


def execute_run(args: argparse.Namespace) -> pathlib.Path:
    config_path = pathlib.Path(args.config).resolve()
    config = load_config(config_path)
    if args.tier not in config.get("tiers", {}):
        raise ValueError(f"unknown tier {args.tier!r}; choose from {', '.join(config.get('tiers', {}))}")
    tier = config["tiers"][args.tier]
    experiments = select_experiments(config, tier, args.experiments)
    mode = str(tier.get("mode", "text"))
    vision_mode = mode == "vision"
    quality_mode = mode == "quality"
    vision_cases_all = load_vision_cases(config) if vision_mode else {}
    quality_filler, quality_cases_all = load_quality_cases(config) if quality_mode else ("", {})
    workloads_all = (
        {name: str(case["prompt"]) for name, case in vision_cases_all.items()}
        if vision_mode else (
            {name: str(case["task"]) for name, case in quality_cases_all.items()}
            if quality_mode else load_workloads(config, config_path)
        )
    )
    workload_names = list(
        tier.get("vision_cases", vision_cases_all) if vision_mode
        else (
            tier.get("quality_cases", quality_cases_all) if quality_mode
            else tier.get("workloads", workloads_all)
        )
    )
    missing_workloads = [name for name in workload_names if name not in workloads_all]
    if missing_workloads:
        raise ValueError(f"tier references unknown workload(s): {missing_workloads}")
    corpus = quality_filler if quality_mode else load_context_corpus(config)
    if args.resume:
        run_dir = pathlib.Path(args.resume).resolve()
        if not run_dir.is_dir():
            raise ValueError(f"resume directory does not exist: {run_dir}")
        run_id = run_dir.name
    else:
        output_root = pathlib.Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = f"{stamp()}-{safe_name(args.tier)}"
        run_dir = output_root / run_id
        run_dir.mkdir()
    for child in ("logs", "telemetry", "responses", "system"):
        (run_dir / child).mkdir(exist_ok=True)
    defaults = config.get("defaults", {})
    host = str(defaults.get("host", "127.0.0.1"))
    port = int(defaults.get("port", 8189))
    base_url = f"http://{host}:{port}"
    effective_request = dict(defaults.get("request", {}))
    effective_request.update(tier.get("request", {}))
    manifest = {
        "schema": 1,
        "harness_version": VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_path": str(config_path),
        "tier_name": args.tier,
        "tier": tier,
        "experiments": experiments,
        "commands": {item["name"]: server_command(config, tier, item) for item in experiments},
        "request": effective_request,
        "requests": {item["name"]: merged_request(config, tier, item) for item in experiments},
        "vision_cases": {name: vision_cases_all[name] for name in workload_names} if vision_mode else None,
        "quality_cases": {name: quality_cases_all[name] for name in workload_names} if quality_mode else None,
        "argv": sys.argv,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    preflight(config, tier, experiments, run_dir, args.allow_busy, args.skip_path_check)
    if args.capture_system and not args.dry_run:
        capture_system(config, experiments, run_dir)
    if args.dry_run:
        for experiment in experiments:
            print(f"[{experiment['name']}]\n  {command_text(server_command(config, tier, experiment))}")
        print(f"Dry run complete; manifest: {run_dir / 'manifest.json'}")
        return run_dir
    results_path = run_dir / "results.jsonl"
    completed = completed_keys(results_path)
    rounds = int(tier.get("rounds", 1))
    depths = [int(value) for value in tier.get("depths", [0])]
    n_predict = int(tier.get("n_predict", 128))
    request_timeout_s = float(tier.get("request_timeout_s", defaults.get("request_timeout_s", 900)))
    warmups = int(tier.get("warmups", 1))
    concurrency = int(tier.get("concurrency", 1))
    startup_only = bool(tier.get("startup_only", False))
    exact_prompt_tokens = bool(tier.get("exact_prompt_tokens", False))
    erase_between_requests = bool(
        tier.get("erase_slot_between_requests", defaults.get("erase_slot_between_requests", False))
    )
    stop_event = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        if stop_event.is_set():
            raise KeyboardInterrupt
        print(f"Received signal {signum}; stopping after current request", file=sys.stderr, flush=True)
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        for round_index in range(rounds):
            # A deterministic rotation avoids giving one topology every coldest
            # or hottest round while preserving reproducibility.
            offset = round_index % len(experiments)
            ordered = experiments[offset:] + experiments[:offset]
            for experiment in ordered:
                if stop_event.is_set():
                    break
                pending = (
                    (
                        [("startup", 0)]
                        if (round_index, experiment["name"], "startup", 0) not in completed
                        else []
                    )
                    if startup_only
                    else [
                        (workload, depth)
                        for workload in workload_names
                        for depth in depths
                        if (round_index, experiment["name"], workload, depth) not in completed
                    ]
                )
                if not pending:
                    continue
                if not port_available(host, port):
                    raise RuntimeError(f"{host}:{port} became busy before starting {experiment['name']}")
                suffix = f"r{round_index + 1:02d}-{safe_name(experiment['name'])}"
                command = server_command(config, tier, experiment)
                effective_args = command[1:]
                extra_request = merged_request(config, tier, experiment)
                host_checkpoint_runtime_verified = False
                host_checkpoint_required = (
                    HOST_CHECKPOINT_MARKER in experiment.get("env", {})
                    and int(option_value(effective_args, "--ctx-checkpoints") or "32") > 0
                )
                checkpoint_every_nt = int(
                    option_value(effective_args, "--checkpoint-every-n-tokens") or "-1"
                )
                checkpoint_verification_threshold = checkpoint_every_nt if checkpoint_every_nt > 0 else 64
                print(f"\n=== {experiment['name']} round {round_index + 1}/{rounds} ===", flush=True)
                print(command_text(command), flush=True)
                server = ManagedServer(
                    command=command,
                    env=merged_env(config, experiment),
                    log_path=run_dir / "logs" / f"{suffix}.log",
                    health_url=base_url + "/health",
                    startup_timeout_s=float(tier.get("startup_timeout_s", defaults.get("startup_timeout_s", 900))),
                    telemetry_path=run_dir / "telemetry" / f"{suffix}.jsonl",
                    telemetry_interval_s=float(tier.get("telemetry_interval_s", defaults.get("telemetry_interval_s", 1.0))),
                    pcie_bdf=defaults.get("pcie_upstream_bdf"),
                    stop_event=stop_event,
                )
                try:
                    server.start()
                    settle = float(tier.get("settle_seconds", defaults.get("settle_seconds", 2)))
                    if settle > 0:
                        time.sleep(settle)
                    if startup_only:
                        samples = list(server.telemetry.samples) if server.telemetry else []
                        row = {
                            "schema": 1,
                            "status": "ok",
                            "mode": "startup_only",
                            "ts": utc_now(),
                            "run_id": run_id,
                            "experiment": experiment["name"],
                            "backend": experiment.get("backend"),
                            "round": round_index,
                            "workload": "startup",
                            "requested_depth_tokens": 0,
                            "n_predict_requested": 0,
                            "http_wall_ms": 0.0,
                            "server_startup_seconds": server.startup_seconds,
                            "output_sha256": "",
                            "output_chars": 0,
                            "degenerate": False,
                            "timing": {},
                            "telemetry": aggregate_telemetry(samples),
                        }
                        append_jsonl(results_path, row)
                        completed.add((round_index, experiment["name"], "startup", 0))
                        available = row["telemetry"].get("mem_available_min_bytes")
                        available_text = (
                            f", host available min {available / 1024**3:.2f} GiB"
                            if isinstance(available, (int, float)) else ""
                        )
                        print(
                            f"  capacity allocation READY in {server.startup_seconds:.1f}s{available_text}",
                            flush=True,
                        )
                        continue

                    prompt_cache: dict[tuple[str, int, int], tuple[str, int | None]] = {}

                    def prepared_prompt(workload: str, depth: int, lane: int) -> tuple[str, int | None]:
                        key = (workload, depth, lane)
                        if key not in prompt_cache:
                            if quality_mode:
                                prompt_cache[key] = (
                                    fit_quality_prompt_to_tokens(
                                        base_url, quality_cases_all[workload], depth,
                                        quality_filler, request_timeout_s,
                                    ) if exact_prompt_tokens else (
                                        make_quality_prompt(
                                            quality_cases_all[workload], depth, quality_filler,
                                        ),
                                        None,
                                    )
                                )
                            elif exact_prompt_tokens:
                                prompt_cache[key] = fit_prompt_to_tokens(
                                    base_url, workloads_all[workload], depth, corpus, request_timeout_s, lane,
                                )
                            else:
                                prompt = make_prompt(workloads_all[workload], depth, corpus)
                                if concurrency > 1:
                                    prompt = f"Independent benchmark lane {lane + 1}; lane marker {lane:08x}.\n" + prompt
                                prompt_cache[key] = (prompt, None)
                        return prompt_cache[key]

                    def request_one(workload: str, prompt: str, predict: int) -> tuple[dict[str, Any], float, dict[str, Any]]:
                        if not vision_mode:
                            response, wall_ms = completion_request(
                                base_url + "/completion", prompt, predict, request_timeout_s, extra_request,
                            )
                            return response, wall_ms, {}
                        log_offset = file_mark(server.log_path)
                        case = vision_cases_all[workload]
                        response, wall_ms = vision_completion_request(
                            base_url + "/v1/chat/completions",
                            prompt,
                            pathlib.Path(str(case["image"])),
                            predict,
                            request_timeout_s,
                            extra_request,
                        )
                        timing = extract_timing(response)
                        draft_n = timing.get("draft_n")
                        if (
                            experiment.get("require_zero_draft")
                            and isinstance(draft_n, (int, float))
                            and draft_n > 0
                        ):
                            raise RuntimeError(
                                f"vision request required speculation disabled, but server drafted {int(draft_n)} tokens"
                            )
                        log_metrics = extract_vision_log_metrics(file_since(server.log_path, log_offset))
                        if (
                            tier.get("require_rocm_mtp_vision")
                            and "-md" in effective_args
                            and not log_metrics["mtp_vision_resync"]
                        ):
                            raise RuntimeError(
                                "the MTP vision request completed without the runtime multimodal-resync marker"
                            )
                        if tier.get("require_rocm_mtp_vision") and "-md" in effective_args:
                            full_log = server.log_path.read_text(encoding="utf-8", errors="ignore")
                            if QWEN4EXP_VISION_STRICT_MARKER not in full_log:
                                raise RuntimeError(
                                    "the MTP vision server did not enable Qwen4Exp single-row target verification"
                                )
                            if QWEN4EXP_VISION_CHECKPOINT_MARKER not in full_log:
                                raise RuntimeError(
                                    "the MTP vision server did not disable recurrent rollback for checkpoint-backed verification"
                                )
                        answer, _reasoning = response_text_parts(response)
                        return response, wall_ms, {
                            "image": str(case["image"]),
                            "image_sha256": sha256_file(pathlib.Path(str(case["image"]))),
                            **anchor_metrics(answer, list(case.get("anchors", []))),
                            **log_metrics,
                        }

                    def verify_host_checkpoint_runtime(responses: list[dict[str, Any]]) -> None:
                        nonlocal host_checkpoint_runtime_verified
                        if not host_checkpoint_required or host_checkpoint_runtime_verified:
                            return
                        prompt_counts = [extract_timing(response).get("prompt_n") for response in responses]
                        if not any(
                            isinstance(value, (int, float)) and int(value) >= checkpoint_verification_threshold
                            for value in prompt_counts
                        ):
                            return
                        log_text = server.log_path.read_text(encoding="utf-8", errors="ignore")
                        if "forcing checkpoint state to host memory" not in log_text:
                            raise RuntimeError(
                                "a prompt crossed the configured checkpoint interval, but the runtime did not confirm "
                                "host-resident checkpoints; this production run is invalid"
                            )
                        host_checkpoint_runtime_verified = True

                    warmup_depth = int(tier.get("warmup_depth", 0))
                    for warmup_index in range(warmups):
                        mark = server.telemetry.mark() if server.telemetry else 0
                        if concurrency > 1:
                            warm_prompts = [
                                prepared_prompt(workload, warmup_depth, lane)[0]
                                for lane, workload in enumerate(workload_names)
                            ]
                            warm_responses, warm_walls, warm_group_wall = concurrent_completion_requests(
                                base_url + "/completion", warm_prompts, min(32, n_predict),
                                request_timeout_s, extra_request,
                            )
                            warm_vision = [{} for _ in warm_responses]
                        else:
                            warm_prompt = prepared_prompt(workload_names[0], warmup_depth, 0)[0]
                            warm_response, warm_wall_ms, warm_vision_metrics = request_one(
                                workload_names[0], warm_prompt, min(32, n_predict),
                            )
                            warm_responses, warm_walls, warm_group_wall = [warm_response], [warm_wall_ms], warm_wall_ms
                            warm_vision = [warm_vision_metrics]
                        verify_host_checkpoint_runtime(warm_responses)
                        samples = server.telemetry.slice(mark) if server.telemetry else []
                        warm_group = concurrent_metrics(warm_responses, warm_group_wall)
                        erased: dict[int, Any] = {}
                        if erase_between_requests:
                            for lane, response in enumerate(warm_responses):
                                slot = response_slot_id(response, lane)
                                if slot not in erased:
                                    erased[slot] = erase_slot(base_url, slot, request_timeout_s)
                        for lane, (workload, response, lane_wall) in enumerate(
                            zip(workload_names, warm_responses, warm_walls)
                        ):
                            warm_row = {
                                "schema": 1,
                                "status": "warmup",
                                "ts": utc_now(),
                                "run_id": run_id,
                                "experiment": experiment["name"],
                                "round": round_index,
                                "warmup": warmup_index,
                                "lane": lane,
                                "concurrency": concurrency,
                                "workload": workload,
                                "requested_depth_tokens": warmup_depth,
                                "http_wall_ms": round(lane_wall, 3),
                                "concurrent_group": warm_group,
                                "timing": extract_timing(response),
                                "telemetry": aggregate_telemetry(samples),
                            }
                            if vision_mode:
                                warm_row["vision"] = warm_vision[lane]
                            if erase_between_requests:
                                slot = response_slot_id(response, lane)
                                warm_row["slot_erase"] = erased[slot]
                            append_jsonl(run_dir / "warmups.jsonl", warm_row)
                    if concurrency > 1:
                        pending_depths = [
                            depth for depth in depths
                            if any(
                                (round_index, experiment["name"], workload, depth) not in completed
                                for workload in workload_names
                            )
                        ]
                        for depth in pending_depths:
                            if stop_event.is_set():
                                break
                            prompts = [
                                prepared_prompt(workload, depth, lane)[0]
                                for lane, workload in enumerate(workload_names)
                            ]
                            mark = server.telemetry.mark() if server.telemetry else 0
                            responses, lane_walls, group_wall = concurrent_completion_requests(
                                base_url + "/completion", prompts, n_predict,
                                request_timeout_s, extra_request,
                            )
                            verify_host_checkpoint_runtime(responses)
                            samples = server.telemetry.slice(mark) if server.telemetry else []
                            telemetry = aggregate_telemetry(samples)
                            group = concurrent_metrics(responses, group_wall)
                            group_id = f"{suffix}-d{depth}"
                            append_jsonl(run_dir / "concurrency-groups.jsonl", {
                                "schema": 1,
                                "status": "ok",
                                "ts": utc_now(),
                                "run_id": run_id,
                                "group_id": group_id,
                                "experiment": experiment["name"],
                                "round": round_index,
                                "requested_depth_tokens": depth,
                                "workloads": workload_names,
                                **group,
                                "telemetry": telemetry,
                            })
                            erased: dict[int, Any] = {}
                            if erase_between_requests:
                                for lane, response in enumerate(responses):
                                    slot = response_slot_id(response, lane)
                                    if slot not in erased:
                                        erased[slot] = erase_slot(base_url, slot, request_timeout_s)
                            for lane, (workload, response, lane_wall) in enumerate(
                                zip(workload_names, responses, lane_walls)
                            ):
                                key = (round_index, experiment["name"], workload, depth)
                                if key in completed:
                                    continue
                                row = probe_row(
                                    run_id, experiment, round_index, workload, depth, n_predict,
                                    response, lane_wall, server.startup_seconds, telemetry,
                                )
                                row.update({
                                    "lane": lane,
                                    "concurrency": concurrency,
                                    "concurrency_group_id": group_id,
                                    "concurrent_group": group,
                                })
                                response_name = f"{suffix}-{safe_name(workload)}-d{depth}-lane{lane}.json"
                                atomic_json(run_dir / "responses" / response_name, response)
                                row["response_file"] = str(pathlib.Path("responses") / response_name)
                                if erase_between_requests:
                                    row["slot_erase"] = erased[response_slot_id(response, lane)]
                                append_jsonl(results_path, row)
                                completed.add(key)
                            print(
                                f"  {concurrency}-user depth~{depth}: aggregate decode {fmt(group['aggregate_decode_tok_s'])} tok/s, "
                                f"aggregate prefill {fmt(group['aggregate_prefill_tok_s'])} tok/s, "
                                f"wall {group_wall / 1000.0:.1f}s",
                                flush=True,
                            )
                        continue
                    for workload, depth in pending:
                        if stop_event.is_set():
                            break
                        prompt, constructed_prompt_tokens = prepared_prompt(workload, depth, 0)
                        mark = server.telemetry.mark() if server.telemetry else 0
                        response, wall_ms, vision_metrics = request_one(workload, prompt, n_predict)
                        verify_host_checkpoint_runtime([response])
                        samples = server.telemetry.slice(mark) if server.telemetry else []
                        row = probe_row(
                            run_id, experiment, round_index, workload, depth, n_predict,
                            response, wall_ms, server.startup_seconds, aggregate_telemetry(samples),
                        )
                        if tier.get("require_probability_metrics") and not row.get("probabilities_sha256"):
                            raise RuntimeError(
                                "llama-server omitted completion_probabilities despite request.n_probs; "
                                "this run cannot fingerprint the prompt state"
                            )
                        if exact_prompt_tokens:
                            row["constructed_prompt_tokens"] = constructed_prompt_tokens
                        if quality_mode:
                            text_quality = quality_case_metrics(
                                response_content(response), quality_cases_all[workload]["validator"],
                            )
                            row["text_quality"] = text_quality
                            row["quality_failure"] = not bool(text_quality["quality_pass"])
                            if row["quality_failure"]:
                                row["degenerate"] = True
                        text_anchors = tier.get("text_quality_anchors", {}).get(workload, [])
                        if text_anchors:
                            text_quality = anchor_metrics(response_content(response), list(text_anchors))
                            minimum = float(tier.get("text_quality_min_anchor_score", 1.0))
                            score = text_quality.get("anchor_score")
                            quality_failure = (
                                not isinstance(score, (int, float)) or float(score) < minimum
                            )
                            text_quality.update({
                                "minimum_anchor_score": minimum,
                                "quality_pass": not quality_failure,
                            })
                            row["text_quality"] = text_quality
                            row["quality_failure"] = quality_failure
                            if quality_failure:
                                row["degenerate"] = True
                        if vision_mode:
                            row["vision"] = vision_metrics
                            min_anchor_score = float(tier.get("vision_min_anchor_score", 0.0))
                            anchor_score = vision_metrics.get("anchor_score")
                            quality_failure = (
                                not isinstance(anchor_score, (int, float))
                                or float(anchor_score) < min_anchor_score
                            )
                            row["quality_failure"] = quality_failure
                            row["vision"]["minimum_anchor_score"] = min_anchor_score
                            row["vision"]["quality_pass"] = not quality_failure
                            if quality_failure:
                                row["degenerate"] = True
                        response_name = f"{suffix}-{safe_name(workload)}-d{depth}.json"
                        atomic_json(run_dir / "responses" / response_name, response)
                        row["response_file"] = str(pathlib.Path("responses") / response_name)
                        if erase_between_requests:
                            row["slot_erase"] = erase_slot(
                                base_url, response_slot_id(response), request_timeout_s,
                            )
                        append_jsonl(results_path, row)
                        completed.add((round_index, experiment["name"], workload, depth))
                        speed = row["timing"].get("predicted_per_second")
                        prefill = row["timing"].get("prompt_per_second")
                        prompt_n = row["timing"].get("prompt_n")
                        prompt_ms = row["timing"].get("prompt_ms")
                        draft = row["timing"].get("draft_acceptance")
                        predicted_n = row["timing"].get("predicted_n")
                        status_parts: list[str] = []
                        if row.get("text_quality") and row.get("quality_failure"):
                            score = row["text_quality"].get("anchor_score")
                            score_text = f"{score:.0%}" if isinstance(score, (int, float)) else "unavailable"
                            if quality_mode:
                                status_parts.append(f"QUALITY FAIL ({score_text})")
                            else:
                                status_parts.append(f"INVALID text quality ({score_text} anchors)")
                        elif quality_mode and row.get("text_quality"):
                            status_parts.append("QUALITY PASS")
                        elif row.get("quality_failure"):
                            score = vision_metrics.get("anchor_score")
                            score_text = f"{score:.0%}" if isinstance(score, (int, float)) else "unavailable"
                            status_parts.append(f"INVALID vision quality ({score_text} anchors)")
                        elif row.get("answer_missing"):
                            status_parts.append("INVALID empty final answer")
                        elif row["degenerate"]:
                            actual = int(predicted_n) if isinstance(predicted_n, (int, float)) else "unknown"
                            status_parts.append(f"INVALID short decode ({actual}/{n_predict} tokens)")
                        status_parts.append(
                            f"decode {speed:.2f} tok/s"
                            if isinstance(speed, (int, float)) else "decode unavailable"
                        )
                        if isinstance(prefill, (int, float)):
                            prefill_text = f"prefill {prefill:.2f} tok/s"
                            if isinstance(prompt_n, (int, float)) and isinstance(prompt_ms, (int, float)):
                                prefill_text += f" ({int(prompt_n)} tok, {prompt_ms:.0f} ms)"
                            status_parts.append(prefill_text)
                        else:
                            status_parts.append("prefill unavailable")
                        if isinstance(draft, (int, float)):
                            status_parts.append(f"MTP accept {draft:.1%}")
                        probability_hash = row.get("probabilities_sha256")
                        if isinstance(probability_hash, str):
                            status_parts.append(f"probability hash {probability_hash[:12]}")
                        if vision_mode:
                            image_ms = vision_metrics.get("image_process_ms")
                            anchor_score = vision_metrics.get("anchor_score")
                            if isinstance(image_ms, (int, float)):
                                status_parts.append(f"image {image_ms:.0f} ms")
                            if isinstance(anchor_score, (int, float)):
                                status_parts.append(f"anchors {anchor_score:.0%}")
                        print(f"  {workload} depth~{depth}: {', '.join(status_parts)}", flush=True)
                except Exception as exc:
                    error = {
                        "schema": 1,
                        "status": "server_error",
                        "ts": utc_now(),
                        "run_id": run_id,
                        "experiment": experiment["name"],
                        "round": round_index,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(results_path, error)
                    print(f"ERROR in {experiment['name']}: {exc}", file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise
                finally:
                    server.stop()
            if stop_event.is_set():
                break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
    summarize(run_dir, config, experiments)
    print(f"Packaging complete run {run_dir.name}...", flush=True)
    archive_path = archive_results(run_dir)
    print(f"\nResults: {run_dir}")
    print(f"Archive: {archive_path}")
    print(f"Checksum: {archive_path}.sha256")
    return run_dir


def median_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(numeric) if numeric else None


def fmt(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = passed / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_p(regressions: int, improvements: int) -> float | None:
    """Two-sided exact McNemar p-value for discordant paired binary outcomes."""
    discordant = regressions + improvements
    if discordant <= 0:
        return None
    tail = sum(
        math.comb(discordant, index) for index in range(min(regressions, improvements) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def deterministic_quality_summary(
    run_dir: pathlib.Path,
    results: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> str:
    samples = [
        row for row in results
        if row.get("status") in SUCCESS_STATES
        and isinstance(row.get("text_quality", {}).get("quality_pass"), bool)
    ]
    if not samples:
        return ""
    baseline_names = [item["name"] for item in experiments if item.get("baseline")]
    baseline_name = baseline_names[0] if baseline_names else experiments[0]["name"]
    baseline_by_cell = {
        (str(row["workload"]), int(row["requested_depth_tokens"]), int(row["round"])):
            bool(row["text_quality"]["quality_pass"])
        for row in samples if row["experiment"] == baseline_name
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[str(row["experiment"])].append(row)
    baseline_rows = grouped.get(baseline_name, [])
    baseline_rate = (
        sum(bool(row["text_quality"]["quality_pass"]) for row in baseline_rows) / len(baseline_rows)
        if baseline_rows else None
    )
    quality_rows: list[dict[str, Any]] = []
    for experiment, rows in grouped.items():
        passed = sum(bool(row["text_quality"]["quality_pass"]) for row in rows)
        total = len(rows)
        rate = passed / total
        low, high = wilson_interval(passed, total)
        regressions = 0
        improvements = 0
        paired = 0
        for row in rows:
            key = (str(row["workload"]), int(row["requested_depth_tokens"]), int(row["round"]))
            if key not in baseline_by_cell:
                continue
            paired += 1
            current_pass = bool(row["text_quality"]["quality_pass"])
            baseline_pass = baseline_by_cell[key]
            regressions += int(baseline_pass and not current_pass)
            improvements += int(not baseline_pass and current_pass)
        quality_rows.append({
            "experiment": experiment,
            "passed": passed,
            "total": total,
            "pass_rate": rate,
            "wilson_low_95": low,
            "wilson_high_95": high,
            "delta_vs_baseline": rate - baseline_rate if baseline_rate is not None else None,
            "paired_cells": paired,
            "paired_regressions": regressions,
            "paired_improvements": improvements,
            "mcnemar_exact_p": exact_mcnemar_p(regressions, improvements),
            "median_decode_tok_s": median_or_none(
                row.get("timing", {}).get("predicted_per_second") for row in rows
            ),
            "median_prefill_tok_s": median_or_none(
                row.get("timing", {}).get("prompt_per_second") for row in rows
            ),
        })
    quality_rows.sort(key=lambda row: (-row["pass_rate"], row["experiment"]))
    with (run_dir / "quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)
    per_case_rows: list[dict[str, Any]] = []
    case_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        case_groups[(str(row["experiment"]), str(row["workload"]))].append(row)
    for (experiment, case_name), rows in sorted(case_groups.items()):
        passed = sum(bool(row["text_quality"]["quality_pass"]) for row in rows)
        per_case_rows.append({
            "experiment": experiment,
            "case": case_name,
            "passed": passed,
            "total": len(rows),
            "pass_rate": passed / len(rows),
        })
    with (run_dir / "quality-cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0]))
        writer.writeheader()
        writer.writerows(per_case_rows)
    md = [
        "## Deterministic functional quality\n\n",
        f"Ground-truth pass rate is the primary quality metric. Baseline for paired regressions: `{baseline_name}`. "
        "Intervals are 95% Wilson intervals; regressions count cells where the baseline passed and the candidate failed.\n\n",
        "| Experiment | Passed | Pass rate | 95% CI | Delta | Regressions | Improvements | McNemar p | Prefill tok/s | Decode tok/s |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in quality_rows:
        low = row["wilson_low_95"]
        high = row["wilson_high_95"]
        interval = f"{100.0 * low:.1f}%–{100.0 * high:.1f}%" if low is not None and high is not None else ""
        delta = row["delta_vs_baseline"]
        delta_text = f"{100.0 * delta:+.1f} pp" if isinstance(delta, (int, float)) else ""
        p_value = row["mcnemar_exact_p"]
        p_text = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "—"
        md.append(
            f"| {row['experiment']} | {row['passed']}/{row['total']} | {100.0 * row['pass_rate']:.1f}% | "
            f"{interval} | {delta_text} | {row['paired_regressions']}/{row['paired_cells']} | "
            f"{row['paired_improvements']}/{row['paired_cells']} | {p_text} | "
            f"{fmt(row['median_prefill_tok_s'])} | {fmt(row['median_decode_tok_s'])} |\n"
        )
    failures = [row for row in per_case_rows if row["passed"] < row["total"]]
    if failures:
        md.extend([
            "\n### Failed deterministic cases\n\n",
            "| Experiment | Case | Passed | Pass rate |\n",
            "|---|---|---:|---:|\n",
        ])
        for row in failures:
            md.append(
                f"| {row['experiment']} | {row['case']} | {row['passed']}/{row['total']} | "
                f"{100.0 * row['pass_rate']:.1f}% |\n"
            )
    return "".join(md) + "\n"


def summarize(run_dir: pathlib.Path, config: dict[str, Any] | None = None, experiments: list[dict[str, Any]] | None = None) -> None:
    results = read_jsonl(run_dir / "results.jsonl")
    if config is None:
        manifest = load_json(run_dir / "manifest.json")
        experiments = manifest.get("experiments", [])
    assert experiments is not None
    quality_md = deterministic_quality_summary(run_dir, results, experiments)
    valid = [row for row in results if row.get("status") in SUCCESS_STATES and not row.get("degenerate")]
    if not valid:
        if quality_md:
            (run_dir / "summary.md").write_text(
                f"# Benchmark summary: {run_dir.name}\n\n" + quality_md,
                encoding="utf-8",
            )
        print(f"No successful, non-degenerate probes to summarize in {run_dir}", file=sys.stderr)
        return
    baseline_names = [item["name"] for item in experiments if item.get("baseline")]
    baseline_name = baseline_names[0] if baseline_names else experiments[0]["name"]
    baseline_hashes: dict[tuple[str, int, int], str] = {}
    for row in valid:
        output_hash = row.get("output_sha256")
        if row["experiment"] == baseline_name and isinstance(output_hash, str) and output_hash:
            key = (row["workload"], int(row["requested_depth_tokens"]), int(row["round"]))
            baseline_hashes.setdefault(key, output_hash)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(row["experiment"], row["workload"], int(row["requested_depth_tokens"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (experiment, workload, depth), rows in sorted(groups.items()):
        speeds = [row["timing"].get("predicted_per_second") for row in rows]
        pp = [row["timing"].get("prompt_per_second") for row in rows]
        prompt_tokens = [row["timing"].get("prompt_n") for row in rows]
        prompt_ms = [row["timing"].get("prompt_ms") for row in rows]
        acceptance = [row["timing"].get("draft_acceptance") for row in rows]
        matches = []
        for row in rows:
            key = (workload, depth, int(row["round"]))
            if key in baseline_hashes and row.get("output_sha256"):
                matches.append(row["output_sha256"] == baseline_hashes[key])
        gpu_busy: dict[str, list[float]] = defaultdict(list)
        gpu_vram: dict[str, list[float]] = defaultdict(list)
        gpu_gtt: dict[str, list[float]] = defaultdict(list)
        gpu_power: dict[str, list[float]] = defaultdict(list)
        gpu_temp: dict[str, list[float]] = defaultdict(list)
        gpu_pcie_rx: dict[str, list[float]] = defaultdict(list)
        gpu_pcie_tx: dict[str, list[float]] = defaultdict(list)
        link_width: list[float] = []
        link_speed: list[float] = []
        rss_file: list[float] = []
        rss_anon: list[float] = []
        rss_shmem: list[float] = []
        rss_total: list[float] = []
        storage_read: list[float] = []
        major_faults: list[float] = []
        mem_available: list[float] = []
        host_cached: list[float] = []
        vision_encode: list[float] = []
        vision_decode: list[float] = []
        vision_process: list[float] = []
        vision_anchor_scores: list[float] = []
        for row in rows:
            telemetry = row.get("telemetry", {})
            vision = row.get("vision", {})
            if isinstance(vision.get("image_encode_ms"), (int, float)):
                vision_encode.append(float(vision["image_encode_ms"]))
            if isinstance(vision.get("image_decode_ms"), (int, float)):
                vision_decode.append(float(vision["image_decode_ms"]))
            if isinstance(vision.get("image_process_ms"), (int, float)):
                vision_process.append(float(vision["image_process_ms"]))
            if isinstance(vision.get("anchor_score"), (int, float)):
                vision_anchor_scores.append(float(vision["anchor_score"]))
            if isinstance(telemetry.get("pid_rss_file_max_bytes"), (int, float)):
                rss_file.append(float(telemetry["pid_rss_file_max_bytes"]))
            if isinstance(telemetry.get("pid_rss_anon_max_bytes"), (int, float)):
                rss_anon.append(float(telemetry["pid_rss_anon_max_bytes"]))
            if isinstance(telemetry.get("pid_rss_shmem_max_bytes"), (int, float)):
                rss_shmem.append(float(telemetry["pid_rss_shmem_max_bytes"]))
            if isinstance(telemetry.get("pid_rss_max_bytes"), (int, float)):
                rss_total.append(float(telemetry["pid_rss_max_bytes"]))
            if isinstance(telemetry.get("pid_read_bytes_delta"), (int, float)):
                storage_read.append(float(telemetry["pid_read_bytes_delta"]))
            if isinstance(telemetry.get("pid_major_faults_delta"), (int, float)):
                major_faults.append(float(telemetry["pid_major_faults_delta"]))
            if isinstance(telemetry.get("mem_available_min_bytes"), (int, float)):
                mem_available.append(float(telemetry["mem_available_min_bytes"]))
            if isinstance(telemetry.get("host_cached_max_bytes"), (int, float)):
                host_cached.append(float(telemetry["host_cached_max_bytes"]))
            for bdf, gpu in telemetry.get("gpus", {}).items():
                if isinstance(gpu.get("busy_mean_percent"), (int, float)):
                    gpu_busy[bdf].append(float(gpu["busy_mean_percent"]))
                if isinstance(gpu.get("vram_used_max_bytes"), (int, float)):
                    gpu_vram[bdf].append(float(gpu["vram_used_max_bytes"]))
                if isinstance(gpu.get("gtt_used_max_bytes"), (int, float)):
                    gpu_gtt[bdf].append(float(gpu["gtt_used_max_bytes"]))
                if isinstance(gpu.get("power_w_mean"), (int, float)):
                    gpu_power[bdf].append(float(gpu["power_w_mean"]))
                if isinstance(gpu.get("temp_c_max"), (int, float)):
                    gpu_temp[bdf].append(float(gpu["temp_c_max"]))
                if isinstance(gpu.get("pcie_rx_est_bytes_s_max"), (int, float)):
                    gpu_pcie_rx[bdf].append(float(gpu["pcie_rx_est_bytes_s_max"]))
                if isinstance(gpu.get("pcie_tx_est_bytes_s_max"), (int, float)):
                    gpu_pcie_tx[bdf].append(float(gpu["pcie_tx_est_bytes_s_max"]))
            if isinstance(telemetry.get("pcie_width_lanes_max"), (int, float)):
                link_width.append(float(telemetry["pcie_width_lanes_max"]))
            if isinstance(telemetry.get("pcie_speed_gt_s_max"), (int, float)):
                link_speed.append(float(telemetry["pcie_speed_gt_s_max"]))
        row_summary = {
            "experiment": experiment,
            "workload": workload,
            "depth_tokens_requested": depth,
            "samples": len(rows),
            "decode_tok_s_median": median_or_none(speeds),
            "decode_tok_s_min": min((float(x) for x in speeds if isinstance(x, (int, float))), default=None),
            "decode_tok_s_max": max((float(x) for x in speeds if isinstance(x, (int, float))), default=None),
            "prefill_tokens_median": median_or_none(prompt_tokens),
            "prefill_ms_median": median_or_none(prompt_ms),
            "prompt_tok_s_median": median_or_none(pp),
            "http_wall_ms_median": median_or_none(row.get("http_wall_ms") for row in rows),
            "vision_encode_ms_median": median_or_none(vision_encode),
            "vision_decode_ms_median": median_or_none(vision_decode),
            "vision_process_ms_median": median_or_none(vision_process),
            "vision_anchor_score_median": median_or_none(vision_anchor_scores),
            "startup_s_median": median_or_none(row.get("server_startup_seconds") for row in rows),
            "mtp_acceptance_median": median_or_none(acceptance),
            "baseline_output_match": (sum(matches) / len(matches)) if matches else None,
            "pcie_speed_gt_s_max": max(link_speed, default=None),
            "pcie_width_lanes_max": max(link_width, default=None),
            "rss_file_max_gib_median": median_or_none(value / 1024**3 for value in rss_file),
            "rss_anon_max_gib_median": median_or_none(value / 1024**3 for value in rss_anon),
            "rss_shmem_max_gib_median": median_or_none(value / 1024**3 for value in rss_shmem),
            "rss_total_max_gib_median": median_or_none(value / 1024**3 for value in rss_total),
            "storage_read_gib_median": median_or_none(value / 1024**3 for value in storage_read),
            "major_faults_median": median_or_none(major_faults),
            "mem_available_min_gib_median": median_or_none(value / 1024**3 for value in mem_available),
            "host_cached_max_gib_median": median_or_none(value / 1024**3 for value in host_cached),
            "gpu_busy_mean": json.dumps({bdf: round(statistics.fmean(vals), 2) for bdf, vals in gpu_busy.items()}, sort_keys=True),
            "gpu_vram_max_gib": json.dumps({bdf: round(max(vals) / 1024**3, 2) for bdf, vals in gpu_vram.items()}, sort_keys=True),
            "gpu_gtt_max_gib": json.dumps({bdf: round(max(vals) / 1024**3, 2) for bdf, vals in gpu_gtt.items()}, sort_keys=True),
            "gpu_power_mean_w": json.dumps({bdf: round(statistics.fmean(vals), 2) for bdf, vals in gpu_power.items()}, sort_keys=True),
            "gpu_temp_max_c": json.dumps({bdf: round(max(vals), 1) for bdf, vals in gpu_temp.items()}, sort_keys=True),
            "gpu_pcie_rx_est_max_mib_s": json.dumps({bdf: round(max(vals) / 1024**2, 2) for bdf, vals in gpu_pcie_rx.items()}, sort_keys=True),
            "gpu_pcie_tx_est_max_mib_s": json.dumps({bdf: round(max(vals) / 1024**2, 2) for bdf, vals in gpu_pcie_tx.items()}, sort_keys=True),
        }
        summary_rows.append(row_summary)
    baseline_speeds = {
        (row["workload"], row["depth_tokens_requested"]): row["decode_tok_s_median"]
        for row in summary_rows
        if row["experiment"] == baseline_name and row["decode_tok_s_median"]
    }
    baseline_prefill_speeds = {
        (row["workload"], row["depth_tokens_requested"]): row["prompt_tok_s_median"]
        for row in summary_rows
        if row["experiment"] == baseline_name and row["prompt_tok_s_median"]
    }
    for row in summary_rows:
        baseline_speed = baseline_speeds.get((row["workload"], row["depth_tokens_requested"]))
        baseline_prefill = baseline_prefill_speeds.get((row["workload"], row["depth_tokens_requested"]))
        speed = row["decode_tok_s_median"]
        prefill = row["prompt_tok_s_median"]
        row["decode_speedup_vs_baseline"] = (
            float(speed) / float(baseline_speed) if speed and baseline_speed else None
        )
        row["prefill_speedup_vs_baseline"] = (
            float(prefill) / float(baseline_prefill) if prefill and baseline_prefill else None
        )
    fields = list(summary_rows[0])
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    experiment_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        experiment_groups[row["experiment"]].append(row)
    gated_quality: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        quality_pass = row.get("text_quality", {}).get("quality_pass")
        if not isinstance(quality_pass, bool):
            quality_pass = row.get("vision", {}).get("quality_pass")
        if isinstance(quality_pass, bool):
            gated_quality[row["experiment"]].append(quality_pass)
    overall: list[dict[str, Any]] = []
    for experiment, rows in experiment_groups.items():
        decode_speedups = [float(row["decode_speedup_vs_baseline"]) for row in rows if row["decode_speedup_vs_baseline"] and row["decode_speedup_vs_baseline"] > 0]
        prefill_speedups = [float(row["prefill_speedup_vs_baseline"]) for row in rows if row["prefill_speedup_vs_baseline"] and row["prefill_speedup_vs_baseline"] > 0]
        matches = [float(row["baseline_output_match"]) for row in rows if row["baseline_output_match"] is not None]
        accepts = [float(row["mtp_acceptance_median"]) for row in rows if row["mtp_acceptance_median"] is not None]
        decode_geomean = math.exp(statistics.fmean(math.log(value) for value in decode_speedups)) if decode_speedups else None
        prefill_geomean = math.exp(statistics.fmean(math.log(value) for value in prefill_speedups)) if prefill_speedups else None
        quality = gated_quality.get(experiment, [])
        quality_passes = sum(quality)
        quality_eligible = not quality or quality_passes == len(quality)
        overall.append({
            "experiment": experiment,
            "cells": len(rows),
            "decode_geomean_speedup": decode_geomean,
            "prefill_geomean_speedup": prefill_geomean,
            "balanced_geomean_speedup": math.sqrt(decode_geomean * prefill_geomean) if decode_geomean and prefill_geomean else None,
            "hash_match": statistics.fmean(matches) if matches else None,
            "mtp_acceptance": statistics.median(accepts) if accepts else None,
            "quality_passes": quality_passes,
            "quality_samples": len(quality),
            "quality_eligible": quality_eligible,
        })
    overall.sort(
        key=lambda row: (row["quality_eligible"], row["balanced_geomean_speedup"] or -1),
        reverse=True,
    )
    md = [
        f"# Benchmark summary: {run_dir.name}\n\n",
        f"Baseline for output hashes: `{baseline_name}`. Medians exclude warm-ups and degenerate responses.\n\n",
        f"Declared cache state: `{load_json(run_dir / 'manifest.json').get('tier', {}).get('cache_state', 'unspecified')}`. "
        "Physical storage reads and major faults remain authoritative; a declared hot run is not hot if those counters stay high.\n\n",
        "## Overall comparable-cell ranking\n\n",
        "Experiments with any measured text or vision anchor failure are disqualified from the overall speed ranking. Their raw passing-cell throughput remains visible for diagnosis.\n\n",
        "| Experiment | Cells | Decode speedup | Prefill speedup | Balanced | Quality gates | Hash match | Median MTP accept |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in overall:
        decode_speedup = "" if row["decode_geomean_speedup"] is None else f"{row['decode_geomean_speedup']:.3f}x"
        prefill_speedup = "" if row["prefill_geomean_speedup"] is None else f"{row['prefill_geomean_speedup']:.3f}x"
        balanced = (
            "DISQUALIFIED"
            if not row["quality_eligible"]
            else ("" if row["balanced_geomean_speedup"] is None else f"{row['balanced_geomean_speedup']:.3f}x")
        )
        quality = ""
        if row["quality_samples"]:
            quality = f"{row['quality_passes']}/{row['quality_samples']} ({row['quality_passes'] / row['quality_samples']:.0%})"
        match = "" if row["hash_match"] is None else f"{row['hash_match']:.0%}"
        accept = "" if row["mtp_acceptance"] is None else f"{row['mtp_acceptance']:.1%}"
        md.append(f"| {row['experiment']} | {row['cells']} | {decode_speedup} | {prefill_speedup} | {balanced} | {quality} | {match} | {accept} |\n")
    if quality_md:
        md.extend(["\n", quality_md])
    md.extend([
        "\n## Per-workload results\n\n",
        "| Experiment | Workload | Requested depth | Prompt n | Samples | Decode tok/s | Prefill tok/s | Prefill ms | MTP accept | Hash match | File RSS GiB | Storage read GiB | Major faults | PCIe |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ])
    ranked = sorted(summary_rows, key=lambda row: row["decode_tok_s_median"] or -1, reverse=True)
    for row in ranked:
        accept = "" if row["mtp_acceptance_median"] is None else f"{row['mtp_acceptance_median']:.1%}"
        match = "" if row["baseline_output_match"] is None else f"{row['baseline_output_match']:.0%}"
        pcie = ""
        if row["pcie_speed_gt_s_max"] is not None:
            pcie = f"{row['pcie_speed_gt_s_max']:g} GT/s x{int(row['pcie_width_lanes_max'] or 0)}"
        md.append(
            f"| {row['experiment']} | {row['workload']} | {row['depth_tokens_requested']} | {fmt(row['prefill_tokens_median'], 0)} | {row['samples']} | "
            f"{fmt(row['decode_tok_s_median'])} | {fmt(row['prompt_tok_s_median'])} | {fmt(row['prefill_ms_median'])} | {accept} | {match} | "
            f"{fmt(row['rss_file_max_gib_median'])} | {fmt(row['storage_read_gib_median'])} | {fmt(row['major_faults_median'], 0)} | {pcie} |\n"
        )
    vision_rows = [row for row in ranked if row.get("vision_anchor_score_median") is not None]
    if vision_rows:
        md.extend([
            "\n## Vision correctness and image processing\n\n",
            "Anchor score checks known visible facts in the deterministic fixture; exact output hash is a stricter cross-backend reproducibility check. Blank image timing fields mean this server build did not emit the corresponding log marker.\n\n",
            "| Experiment | Case | Anchor score | Image encode ms | Image decode ms | Image total ms | Prompt tokens | Prompt ms | HTTP wall ms |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for row in vision_rows:
            md.append(
                f"| {row['experiment']} | {row['workload']} | {fmt(100.0 * row['vision_anchor_score_median'], 0)}% | "
                f"{fmt(row['vision_encode_ms_median'])} | {fmt(row['vision_decode_ms_median'])} | "
                f"{fmt(row['vision_process_ms_median'])} | {fmt(row['prefill_tokens_median'], 0)} | "
                f"{fmt(row['prefill_ms_median'])} | {fmt(row['http_wall_ms_median'])} |\n"
            )
    quality_failures = [
        row for row in results
        if row.get("status") in SUCCESS_STATES and row.get("quality_failure")
    ]
    if quality_failures:
        md.extend([
            "\n## Quality failures\n\n",
            "These requests completed normally but lost required task or image anchors. They are excluded from all speed rankings.\n\n",
            "| Experiment | Round | Workload | Depth | Gate | Score | Matched anchors |\n",
            "|---|---:|---:|---:|---:|---:|---|\n",
        ])
        for row in quality_failures:
            gate = row.get("text_quality") or row.get("vision") or {}
            kind = "text" if row.get("text_quality") else "vision"
            score = gate.get("anchor_score")
            matched = ", ".join(str(value) for value in gate.get("anchors_matched", []))
            md.append(
                f"| {row.get('experiment', '')} | {int(row.get('round', 0)) + 1} | "
                f"{row.get('workload', '')} | {row.get('requested_depth_tokens', '')} | {kind} | "
                f"{fmt(100.0 * score, 0) + '%' if isinstance(score, (int, float)) else ''} | {matched} |\n"
            )
    md.extend([
        "\n## Residency and capacity telemetry\n\n",
        "Host available memory includes reclaimable cache. GPU maps are keyed by PCI BDF.\n\n",
        "| Experiment | Workload | Depth | Host available min GiB | Host cached max GiB | Process RSS GiB | Anon RSS GiB | File RSS GiB | Shmem RSS GiB | GPU VRAM max GiB | GPU GTT max GiB |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n",
    ])
    for row in ranked:
        md.append(
            f"| {row['experiment']} | {row['workload']} | {row['depth_tokens_requested']} | "
            f"{fmt(row['mem_available_min_gib_median'])} | {fmt(row['host_cached_max_gib_median'])} | "
            f"{fmt(row['rss_total_max_gib_median'])} | {fmt(row['rss_anon_max_gib_median'])} | "
            f"{fmt(row['rss_file_max_gib_median'])} | {fmt(row['rss_shmem_max_gib_median'])} | "
            f"{row['gpu_vram_max_gib']} | {row['gpu_gtt_max_gib']} |\n"
        )
    concurrent_groups = [
        row for row in read_jsonl(run_dir / "concurrency-groups.jsonl")
        if row.get("status") == "ok"
    ]
    # v1.13 and earlier persisted the overlap upper bound under the aggregate
    # field names. Recompute from the archived per-lane timings so summarizing
    # an older run cannot silently relabel optimistic values as conservative.
    lanes_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        group_id = row.get("concurrency_group_id")
        if isinstance(group_id, str) and group_id:
            lanes_by_group[group_id].append(row)
    normalized_groups: list[dict[str, Any]] = []
    for group in concurrent_groups:
        group_id = group.get("group_id")
        lane_rows = lanes_by_group.get(group_id, []) if isinstance(group_id, str) else []
        if lane_rows:
            responses = [
                {"timings": row.get("timing", {})}
                for row in sorted(lane_rows, key=lambda row: int(row.get("lane", 0)))
            ]
            metrics = concurrent_metrics(responses, float(group.get("wall_ms", 0)))
            group = {**group, **metrics}
        normalized_groups.append(group)
    concurrent_groups = normalized_groups
    if concurrent_groups:
        concurrency_fields = [
            "experiment", "round", "requested_depth_tokens", "lanes", "prompt_n_total",
            "predicted_n_total", "aggregate_prefill_tok_s", "aggregate_decode_tok_s",
            "overlap_upper_prefill_tok_s", "overlap_upper_decode_tok_s",
            "lane_prefill_min_tok_s", "lane_prefill_max_tok_s", "prefill_fairness_ratio",
            "lane_decode_min_tok_s", "lane_decode_max_tok_s", "decode_fairness_ratio",
            "end_to_end_output_tok_s", "wall_ms",
        ]
        with (run_dir / "concurrency.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=concurrency_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(concurrent_groups)
        md.extend([
            "\n## True concurrent-request throughput\n\n",
            "Conservative aggregate rates divide total tokens by the sum of per-lane phase times. The overlap upper bound divides by the longest lane phase and is achievable only when those phases truly overlap. End-to-end output includes prompt processing.\n\n",
            "| Experiment | Round | Depth/lane | Lanes | Conservative prefill | Prefill overlap upper | Conservative decode | Decode overlap upper | Slow lane decode | Decode fairness | End-to-end output | Wall s |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for group in concurrent_groups:
            md.append(
                f"| {group['experiment']} | {int(group.get('round', 0)) + 1} | "
                f"{group.get('requested_depth_tokens', '')} | {group.get('lanes', '')} | "
                f"{fmt(group.get('aggregate_prefill_tok_s'))} | {fmt(group.get('overlap_upper_prefill_tok_s'))} | "
                f"{fmt(group.get('aggregate_decode_tok_s'))} | {fmt(group.get('overlap_upper_decode_tok_s'))} | "
                f"{fmt(group.get('lane_decode_min_tok_s'))} | {fmt(group.get('decode_fairness_ratio'), 3)} | "
                f"{fmt(group.get('end_to_end_output_tok_s'))} | {fmt(float(group.get('wall_ms', 0)) / 1000.0, 1)} |\n"
            )
    failures = [row for row in results if row.get("status") not in SUCCESS_STATES]
    if failures:
        md.extend(["\n## Failures\n\n"])
        for row in failures:
            md.append(f"- `{row.get('experiment', 'unknown')}` round {int(row.get('round', 0)) + 1}: {row.get('error', row.get('status'))}\n")
    (run_dir / "summary.md").write_text("".join(md), encoding="utf-8")


def execute_preflight(args: argparse.Namespace) -> pathlib.Path:
    config_path = pathlib.Path(args.config).resolve()
    config = load_config(config_path)
    tier = config.get("tiers", {}).get(args.tier)
    if tier is None:
        raise ValueError(f"unknown tier {args.tier!r}")
    experiments = select_experiments(config, tier, args.experiments)
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = preflight(config, tier, experiments, output, args.allow_busy, args.skip_path_check)
    if args.capture_system:
        capture_system(config, experiments, output)
    print(json.dumps(report, indent=2))
    return output


def execute_rocm_audit(args: argparse.Namespace) -> pathlib.Path:
    config_path = pathlib.Path(args.config).resolve()
    config = load_config(config_path)
    output = pathlib.Path(args.output).resolve()
    devices = parse_selector(args.devices)
    if not devices:
        raise ValueError("--devices must select at least one ROCm backend")
    report = run_rocm_audit(config, output, args.run_ops, devices, args.timeout)
    print(json.dumps(report, indent=2))
    archive_path = output.parent / f"rocm-audit-{stamp()}.tar.gz"
    audit_logs = output.parent / "rocm-audit-logs"
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        if output.is_file():
            archive.add(output, arcname=output.name)
        if audit_logs.is_dir():
            archive.add(audit_logs, arcname=audit_logs.name)
    digest = sha256_file(archive_path)
    archive_path.with_name(archive_path.name + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="ascii",
    )
    print(f"Archive: {archive_path}")
    print(f"Checksum: {archive_path}.sha256")
    if not report["ready_for_model_benchmarks"]:
        raise RuntimeError(
            "ROCm audit did not prove the build safe for model benchmarks; inspect "
            f"{output} and {output.parent / 'rocm-audit-logs'}"
        )
    return output


def execute_archive(args: argparse.Namespace) -> pathlib.Path:
    run_dir = pathlib.Path(args.run_dir).resolve()
    output = pathlib.Path(args.output).resolve() if args.output else None
    print(f"Packaging {run_dir}...", flush=True)
    archive = archive_results(run_dir, output)
    print(f"Archive: {archive}")
    print(f"Checksum: {archive}.sha256")
    return archive


def self_test() -> None:
    parsed = parse_link_status("LnkSta: Speed 16GT/s, Width x4\n")
    assert parsed == {"speed": "16GT/s", "width": "x4", "speed_gt_s": 16.0, "width_lanes": 4}
    timing = extract_timing({"timings": {
        "prompt_n": 4096,
        "prompt_ms": 2048.0,
        "prompt_per_second": 2000.0,
        "predicted_n": 10,
        "draft_n": 8,
        "draft_n_accepted": 6,
    }})
    assert timing["draft_acceptance"] == 0.75
    assert timing["prompt_n"] == 4096
    assert timing["prompt_ms"] == 2048.0
    assert timing["prompt_per_second"] == 2000.0
    canonical = canonicalize_server_args([
        "--cache-type-k", "q8_0", "--jinja", "--cache-type-k", "f16",
        "--spec-draft-n-max", "2", "--spec-draft-n-max", "4",
        "--no-kv-unified", "--kv-unified", "--no-mmproj-offload", "--mmproj-offload",
    ])
    assert canonical == [
        "--jinja", "--cache-type-k", "f16", "--spec-draft-n-max", "4",
        "--kv-unified", "--mmproj-offload",
    ]
    assert translate_server_options(
        ["--load-mode", "mmap", "--jinja", "--flash-attn", "on"],
        {"--load-mode": {"mmap": ["--mmap"], "none": ["--no-mmap"]}},
    ) == ["--mmap", "--jinja", "--flash-attn", "on"]
    assert translate_server_options(
        ["--load-mode", "none", "--jinja"],
        {"--load-mode": {"mmap": ["--mmap"], "none": ["--no-mmap"]}},
    ) == ["--no-mmap", "--jinja"]
    assert server_arg_compatibility_errors([
        "--flash-attn", "off", "--cache-type-v", "q8_0",
    ])
    assert not server_arg_compatibility_errors([
        "--flash-attn", "off", "--cache-type-v", "f16",
    ])
    oai = {
        "choices": [{"message": {"content": "A red circle and UNSLOTH 42"}}],
        "timings": {"predicted_n": 16},
        "__verbose": {"id_slot": 2},
    }
    assert response_content(oai) == "A red circle and UNSLOTH 42"
    assert response_text_parts(oai) == ("A red circle and UNSLOTH 42", "")
    reasoning_only = {"choices": [{"message": {"content": "", "reasoning_content": "visible reasoning"}}]}
    assert response_content(reasoning_only) == "visible reasoning"
    assert response_text_parts(reasoning_only) == ("", "visible reasoning")
    anchors = anchor_metrics(response_content(oai), ["red", "circle", "blue", "UNSLOTH 42"])
    assert anchors["anchor_score"] == 0.75
    exact_quality = quality_case_metrics(
        "<think>retrieve it</think>\nEMBER-731942\n",
        {"type": "exact", "expected": "EMBER-731942"},
    )
    assert exact_quality["quality_pass"] is True and exact_quality["anchor_score"] == 1.0
    json_quality = quality_case_metrics(
        "<think>assemble</think>\n```json\n{\"alpha\":\"COPPER-17\"}\n```",
        {"type": "json_equals", "expected": {"alpha": "COPPER-17"}},
    )
    assert json_quality["quality_pass"] is True and json_quality["observed"] == {"alpha": "COPPER-17"}
    quality_prompt = make_quality_prompt({
        "task": "Return only TEST-1.",
        "records": [{"fraction": 0.5, "text": "QUALITY RECORD TEST: TEST-1."}],
        "validator": {"type": "exact", "expected": "TEST-1"},
    }, 256, "Neutral archive text. ")
    assert quality_prompt.count("QUALITY RECORD TEST: TEST-1.") == 1
    assert quality_prompt.endswith("Return only TEST-1.")
    with tempfile.TemporaryDirectory() as raw_tmp:
        quality_results = pathlib.Path(raw_tmp) / "results.jsonl"
        append_jsonl(quality_results, {
            "status": "ok", "degenerate": True, "quality_failure": True,
            "round": 0, "experiment": "candidate", "workload": "case-a",
            "requested_depth_tokens": 32768,
        })
        assert completed_keys(quality_results) == {(0, "candidate", "case-a", 32768)}
    probability = probability_metrics({"completion_probabilities": [{
        "content": "x",
        "probs": [
            {"id": 1, "tok_str": "x", "prob": 0.75},
            {"id": 2, "tok_str": "y", "prob": 0.25},
        ],
    }]})
    assert probability["probability_steps"] == 1
    assert probability["first_token_probabilities"][0]["id"] == 1
    assert len(probability["probabilities_sha256"]) == 64
    current_probability = probability_metrics({"completion_probabilities": [{
        "id": 271,
        "token": "\n\n",
        "logprob": -0.25,
        "top_logprobs": [
            {"id": 271, "token": "\n\n", "logprob": -0.25},
            {"id": 262, "token": "   ", "logprob": -2.0},
        ],
    }]})
    assert current_probability["first_token_probabilities"][0] == {
        "id": 271, "token": "\n\n", "logprob": -0.25,
    }
    with tempfile.TemporaryDirectory() as raw_tmp:
        preflight_report = preflight(
            {"defaults": {"host": "127.0.0.1", "port": 0}},
            {
                "workloads": ["code"],
                "text_quality_anchors": {"code": ["def merge_intervals("]},
                "text_quality_min_anchor_score": 1.0,
                "warmups": 0,
            },
            [], pathlib.Path(raw_tmp), allow_busy=True, skip_path_check=True,
        )
        assert preflight_report["errors"] == []
    with tempfile.TemporaryDirectory() as raw_tmp:
        quality_preflight = preflight(
            {
                "defaults": {"host": "127.0.0.1", "port": 0},
                "quality_cases_file": str(pathlib.Path(__file__).with_name("quality_cases.json")),
            },
            {
                "mode": "quality",
                "quality_cases": ["passkey_early"],
                "depths": [32768],
                "exact_prompt_tokens": True,
                "n_predict": 256,
                "request": {"ignore_eos": False},
                "warmups": 0,
            },
            [], pathlib.Path(raw_tmp), allow_busy=True, skip_path_check=True,
        )
        assert quality_preflight["errors"] == []
    vision_log = extract_vision_log_metrics(
        "image/slice encoded in 12.5 ms\nimage decoded (batch 1/1) in 3.5 ms\nimage processed in 16.2 ms\n"
    )
    assert vision_log == {
        "image_encode_ms": 12.5,
        "image_decode_ms": 3.5,
        "image_process_ms": 16.2,
        "mtp_vision_resync": False,
    }
    assert extract_vision_log_metrics(MTP_VISION_RESYNC_MARKER)["mtp_vision_resync"] is True
    short = probe_row(
        "self-test", {"name": "test", "backend": "cpu"}, 0, "code", 0, 128,
        {"content": "", "timings": {"predicted_n": 0}}, 1.0, None, {},
    )
    full = probe_row(
        "self-test", {"name": "test", "backend": "cpu"}, 0, "code", 0, 128,
        {"content": "x", "timings": {"predicted_n": 128}}, 1.0, None, {},
    )
    assert short["degenerate"] is True
    assert full["degenerate"] is False
    empty_answer = probe_row(
        "self-test", {"name": "test", "backend": "cpu"}, 0, "vision", 0, 64,
        {"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}],
         "timings": {"predicted_n": 64}},
        1.0, None, {},
    )
    assert empty_answer["degenerate"] is True and empty_answer["answer_missing"] is True
    bw = parse_pcie_bw("100 200 256\n")
    assert bw["pcie_rx_est_bytes_s"] == 25_600 and bw["pcie_tx_est_bytes_s"] == 51_200
    expanded = expand_tree({"x": "{root}/file"}, {"root": "/tmp"})
    assert expanded["x"] == "/tmp/file"
    prompt = make_prompt("task", 100, "abcdef")
    assert prompt.endswith("task") and len(prompt) >= 400
    original_tokenize_count = globals()["tokenize_count"]
    try:
        globals()["tokenize_count"] = lambda _url, content, _timeout: len(content)
        exact_prompt, exact_count = fit_prompt_to_tokens(
            "http://self-test", "task", 4096, "abcdef", 1.0,
        )
        assert exact_count == 4096 and len(exact_prompt) == 4096
    finally:
        globals()["tokenize_count"] = original_tokenize_count
    aggregate = aggregate_telemetry([
        {
            "pid_rss_bytes": 10,
            "process": {
                "rss_file_bytes": 20,
                "rss_anon_bytes": 30,
                "rss_shmem_bytes": 15,
                "read_bytes": 100,
                "major_faults": 4,
            },
            "host": {"MemAvailable": 100, "Cached": 50},
            "gpus": {"0000:01:00.0": {"gpu_busy_percent": 50, "vram_used_bytes": 20, "gtt_used_bytes": 2}},
            "pcie": {"speed_gt_s": 16.0, "width_lanes": 4},
        },
        {
            "pid_rss_bytes": 12,
            "process": {
                "rss_file_bytes": 25,
                "rss_anon_bytes": 35,
                "rss_shmem_bytes": 18,
                "read_bytes": 140,
                "major_faults": 7,
            },
            "host": {"MemAvailable": 90, "Cached": 60},
            "gpus": {},
            "pcie": {},
        },
    ])
    assert aggregate["gpus"]["0000:01:00.0"]["vram_used_max_bytes"] == 20
    assert aggregate["pid_rss_file_max_bytes"] == 25
    assert aggregate["pid_rss_shmem_max_bytes"] == 18
    assert aggregate["pid_read_bytes_delta"] == 40
    assert aggregate["pid_major_faults_delta"] == 3
    assert aggregate["mem_available_min_bytes"] == 90
    assert aggregate["host_cached_max_bytes"] == 60
    assert response_slot_id({"id_slot": 3}) == 3
    assert response_slot_id(oai) == 2
    assert response_slot_id({}) == 0
    assert response_slot_id({}, 1) == 1
    concurrent = concurrent_metrics([
        {"timings": {"predicted_n": 100, "predicted_ms": 5000, "predicted_per_second": 20, "prompt_n": 1000, "prompt_ms": 2000, "prompt_per_second": 500}},
        {"timings": {"predicted_n": 100, "predicted_ms": 4000, "predicted_per_second": 25, "prompt_n": 1000, "prompt_ms": 2500, "prompt_per_second": 400}},
    ], 8000)
    assert math.isclose(concurrent["aggregate_decode_tok_s"], 200000 / 9000)
    assert math.isclose(concurrent["aggregate_prefill_tok_s"], 2000000 / 4500)
    assert concurrent["overlap_upper_decode_tok_s"] == 40.0
    assert concurrent["overlap_upper_prefill_tok_s"] == 800.0
    assert concurrent["decode_fairness_ratio"] == 0.8
    assert concurrent["prefill_fairness_ratio"] == 0.8
    assert concurrent["end_to_end_output_tok_s"] == 25.0
    passed, passed_n, total_n = backend_ops_passed({
        "returncode": 0, "stdout": "43/43 tests passed", "stderr": "",
    })
    assert passed and passed_n == 43 and total_n == 43
    assert not backend_ops_passed({"returncode": 0, "stdout": "0/0 tests passed", "stderr": ""})[0]
    with tempfile.TemporaryDirectory() as raw_tmp:
        source_root = pathlib.Path(raw_tmp)
        runtime_root = source_root / "ggml" / "src" / "ggml-cuda"
        runtime_root.mkdir(parents=True)
        (runtime_root / "custom.cu").write_text(
            "Q4_0_ROCMFP4 Q4_0_ROCMFP4_FAST", encoding="utf-8",
        )
        assert inspect_rocmfp4_sources(source_root)["static_dispatch_ready"] is True
    with tempfile.TemporaryDirectory() as raw_tmp:
        quality_root = pathlib.Path(raw_tmp)
        quality_markdown = deterministic_quality_summary(quality_root, [
            {
                "status": "ok", "experiment": "base", "workload": "case-a",
                "requested_depth_tokens": 32768, "round": 0,
                "text_quality": {"quality_pass": True},
            },
            {
                "status": "ok", "experiment": "candidate", "workload": "case-a",
                "requested_depth_tokens": 32768, "round": 0,
                "text_quality": {"quality_pass": False},
            },
        ], [{"name": "base", "baseline": True}, {"name": "candidate"}])
        assert "candidate | 0/1 | 0.0%" in quality_markdown
        assert "1/1 | 0/1" in quality_markdown
        assert (quality_root / "quality.csv").is_file()
        assert (quality_root / "quality-cases.csv").is_file()
        assert "candidate,case-a,0,1,0.0" in (
            quality_root / "quality-cases.csv"
        ).read_text(encoding="utf-8")
        low, high = wilson_interval(1, 1)
        assert low is not None and high == 1.0 and 0.0 < low < 1.0
        assert exact_mcnemar_p(0, 0) is None
        assert exact_mcnemar_p(1, 0) == 1.0
        assert exact_mcnemar_p(6, 0) == 0.03125
    with tempfile.TemporaryDirectory() as raw_tmp:
        summary_root = pathlib.Path(raw_tmp)
        atomic_json(summary_root / "manifest.json", {"tier": {"cache_state": "hot"}})
        append_jsonl(summary_root / "results.jsonl", {
            "status": "ok",
            "experiment": "test",
            "workload": "code",
            "requested_depth_tokens": 4096,
            "round": 0,
            "lane": 0,
            "concurrency_group_id": "test-group",
            "output_sha256": "abc",
            "http_wall_ms": 1.0,
            "server_startup_seconds": 2.0,
            "degenerate": False,
            "timing": {
                "predicted_n": 100,
                "predicted_ms": 5000.0,
                "predicted_per_second": 20.0,
                "prompt_n": 1000,
                "prompt_ms": 2000.0,
                "prompt_per_second": 500.0,
                "draft_acceptance": 0.75,
            },
            "telemetry": aggregate,
            "vision": {
                "anchor_score": 1.0,
                "quality_pass": True,
                "image_encode_ms": 12.5,
                "image_decode_ms": 3.5,
                "image_process_ms": 16.2,
            },
        })
        append_jsonl(summary_root / "results.jsonl", {
            "status": "ok",
            "experiment": "test",
            "workload": "code",
            "requested_depth_tokens": 4096,
            "round": 0,
            "lane": 1,
            "concurrency_group_id": "test-group",
            "output_sha256": "abc",
            "http_wall_ms": 1.0,
            "server_startup_seconds": 2.0,
            "degenerate": False,
            "timing": {
                "predicted_n": 100,
                "predicted_ms": 4000.0,
                "predicted_per_second": 25.0,
                "prompt_n": 1000,
                "prompt_ms": 2500.0,
                "prompt_per_second": 400.0,
                "draft_acceptance": 0.75,
            },
            "telemetry": aggregate,
            "vision": {
                "anchor_score": 1.0,
                "quality_pass": True,
                "image_encode_ms": 13.5,
                "image_decode_ms": 4.5,
                "image_process_ms": 18.2,
            },
        })
        append_jsonl(summary_root / "concurrency-groups.jsonl", {
            "status": "ok",
            "experiment": "test",
            "group_id": "test-group",
            "round": 0,
            "requested_depth_tokens": 4096,
            "lanes": 2,
            "prompt_n_total": 2000,
            "predicted_n_total": 200,
            "aggregate_prefill_tok_s": 800.0,
            "aggregate_decode_tok_s": 40.0,
            "end_to_end_output_tok_s": 25.0,
            "wall_ms": 8000.0,
        })
        append_jsonl(summary_root / "results.jsonl", {
            "status": "ok",
            "experiment": "test",
            "workload": "failed-vision-case",
            "requested_depth_tokens": 0,
            "round": 0,
            "degenerate": True,
            "quality_failure": True,
            "vision": {"anchor_score": 0.5, "quality_pass": False},
        })
        summarize(summary_root, {}, [{"name": "test", "baseline": True}])
        rendered = (summary_root / "summary.md").read_text(encoding="utf-8")
        assert "Declared cache state: `hot`" in rendered
        assert "Residency and capacity telemetry" in rendered
        assert "Process RSS GiB" in rendered and "Shmem RSS GiB" in rendered
        assert "Vision correctness and image processing" in rendered
        assert "100% | 13.00 | 4.00 | 17.20" in rendered
        assert "DISQUALIFIED | 2/3 (67%)" in rendered
        assert "True concurrent-request throughput" in rendered
        assert (summary_root / "concurrency.csv").is_file()
        concurrency_csv = (summary_root / "concurrency.csv").read_text(encoding="utf-8")
        assert "22.22222222222222" in concurrency_csv
        assert "444.44444444444446" in concurrency_csv
        assert ",800.0,40.0," in concurrency_csv
    with tempfile.TemporaryDirectory() as raw_tmp:
        archive_root = pathlib.Path(raw_tmp)
        archived_run = archive_root / "example-run"
        archived_run.mkdir()
        atomic_json(archived_run / "manifest.json", {"run_id": "example-run"})
        (archived_run / "sample.txt").write_text("payload\n", encoding="utf-8")
        archive_path = archive_results(archived_run)
        assert archive_path == archive_root / "example-run.tar.gz"
        assert archive_path.is_file()
        assert archive_path.with_name(archive_path.name + ".sha256").is_file()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name for member in archive.getmembers()}
        assert "example-run/manifest.json" in members
        assert "example-run/archive-manifest.json" in members
        assert "example-run/sample.txt" in members
    inherited = resolve_experiment_inheritance([
        {"name": "base", "args": ["a"], "env": {"X": "1"}},
        {"name": "child", "extends": "base", "args_append": ["b"], "env": {"Y": "2"}},
    ])
    child = next(item for item in inherited if item["name"] == "child")
    assert child["args"] == ["a", "b"] and child["env"] == {"X": "1", "Y": "2"}
    try:
        validate_experiment_topologies([{
            "name": "bad",
            "args": ["--split-mode", "none", "--override-tensor", "^x$=Vulkan1"],
        }])
    except ValueError:
        pass
    else:
        raise AssertionError("non-primary tensor override with split-mode none was not rejected")
    ordered = select_experiments(
        {"experiments": [{"name": "a"}, {"name": "b"}]},
        {"experiments": ["b", "a"]},
        None,
    )
    assert [item["name"] for item in ordered] == ["b", "a"]
    shipped_config = load_json(pathlib.Path(__file__).with_name("matrix.json"))
    shipped_defaults = shipped_config.get("defaults", {})
    for tier_name, tier in shipped_config.get("tiers", {}).items():
        cache_state = str(tier.get("cache_state", "unspecified"))
        if cache_state == "hot":
            assert int(tier.get("warmups", 0)) >= 1, f"{tier_name}: hot tier has no warm-up"
            if str(tier.get("mode", "text")) != "vision":
                assert int(tier.get("warmup_depth", 0)) > 0, (
                    f"{tier_name}: hot text tier has no nonzero warm-up depth"
                )
            assert bool(tier.get(
                "erase_slot_between_requests",
                shipped_defaults.get("erase_slot_between_requests", False),
            )), f"{tier_name}: hot tier does not erase slot KV state"
        if cache_state == "cold":
            assert int(tier.get("warmups", 0)) == 0, f"{tier_name}: cold tier has warm-ups"
    expanded_shipped_config = load_config(pathlib.Path(__file__).with_name("matrix.json"))
    quality_filler, shipped_quality_cases = load_quality_cases(expanded_shipped_config)
    assert "neutral" in quality_filler.casefold()
    assert len(shipped_quality_cases) == 8
    assert shipped_quality_cases["json_retrieval"]["validator"]["type"] == "json_equals"
    for tier_name in (
        "rocm-smoke", "rocm-depth", "backend-smoke-matched", "rocm-placement", "rocm-mtp",
    ):
        rocm_tier = expanded_shipped_config["tiers"][tier_name]
        assert rocm_tier.get("cache_state") == "hot"
        assert int(rocm_tier.get("warmup_depth", 0)) >= max(
            int(depth) for depth in rocm_tier["depths"]
        ), f"{tier_name}: deepest prompt is not pre-warmed"
    for experiment in expanded_shipped_config["experiments"]:
        if experiment.get("backend") != "rocm":
            continue
        assert not str(experiment["model"]).endswith("-PLE16.gguf"), (
            f"{experiment['name']}: ROCm must use the native joined PLE model"
        )
        rocm_override = option_value(list(experiment.get("args", [])), "--override-tensor") or ""
        assert "ple_ngram_embd" not in rocm_override, (
            f"{experiment['name']}: split-PLE overrides are Vulkan-only"
        )
    matched_tier = expanded_shipped_config["tiers"]["backend-smoke-matched"]
    matched_experiments = select_experiments(expanded_shipped_config, matched_tier, None)
    assert len(matched_experiments) == 4
    matched_backends: defaultdict[str, int] = defaultdict(int)
    for experiment in matched_experiments:
        matched_args = server_command(expanded_shipped_config, matched_tier, experiment)[1:]
        backend = str(experiment["backend"])
        matched_backends[backend] += 1
        assert option_value(matched_args, "--cache-type-k") == "f16"
        assert option_value(matched_args, "--cache-type-v") == "f16"
        assert not server_arg_compatibility_errors(matched_args)
        model = option_value(matched_args, "-m") or ""
        override = option_value(matched_args, "--override-tensor") or ""
        if backend == "vulkan":
            assert model.endswith("-PLE16.gguf")
            assert "ple_ngram_embd" in override and "=CPU" in override
            assert option_value(matched_args, "--load-mode") == "mmap"
        elif backend == "rocm":
            assert model.endswith("-STRIX.gguf") and not model.endswith("-PLE16.gguf")
            assert "ple_ngram_embd" not in override
            assert "per_layer_token_embd" not in override
            assert "--mmap" in matched_args and "--load-mode" not in matched_args
        else:
            raise AssertionError(f"unexpected matched-tier backend: {backend}")
    assert dict(matched_backends) == {"vulkan": 2, "rocm": 2}
    placement_tier = expanded_shipped_config["tiers"]["rocm-placement"]
    placement_experiments = select_experiments(expanded_shipped_config, placement_tier, None)
    assert [item["name"] for item in placement_experiments] == [
        "apu_hip_joined_f16kv_no_mtp",
        "layer_82_18_hip_f16kv_no_mtp",
        "expert_hip_f16kv_no_mtp",
        "expert_shared_hip_f16kv_no_mtp",
    ]
    for experiment in placement_experiments:
        placement_args = server_command(expanded_shipped_config, placement_tier, experiment)[1:]
        assert option_value(placement_args, "--batch-size") == "2048"
        assert option_value(placement_args, "--cache-type-k") == "f16"
        assert option_value(placement_args, "--cache-type-v") == "f16"
    shared_override = option_value(
        list(placement_experiments[-1].get("args", [])), "--override-tensor",
    ) or ""
    assert "(exps|shexp)" in shared_override and "ffn_gate_inp_shexp" in shared_override
    mtp_tier = expanded_shipped_config["tiers"]["rocm-mtp"]
    assert mtp_tier.get("require_rocm_mtp") is True
    mtp_experiments = select_experiments(expanded_shipped_config, mtp_tier, None)
    assert [item["name"] for item in mtp_experiments] == [
        "expert_hip_f16kv_no_mtp",
        "expert_hip_f16kv_mtp_n4_igpu",
        "expert_hip_f16kv_mtp_n4_dgpu",
    ]
    assert option_value(list(mtp_experiments[1]["args"]), "--spec-draft-device") == "ROCm1"
    assert option_value(list(mtp_experiments[2]["args"]), "--spec-draft-device") == "ROCm0"
    for experiment in mtp_experiments[1:]:
        mtp_args = server_command(expanded_shipped_config, mtp_tier, experiment)[1:]
        assert option_value(mtp_args, "--spec-draft-n-max") == "4"
        assert option_value(mtp_args, "--spec-draft-p-min") == "0.75"
        assert option_value(mtp_args, "--cache-type-k") == "f16"
        assert option_value(mtp_args, "--cache-type-v") == "f16"
    mtp_window_tier = expanded_shipped_config["tiers"]["rocm-mtp-window"]
    mtp_window_experiments = select_experiments(expanded_shipped_config, mtp_window_tier, None)
    assert [item["name"] for item in mtp_window_experiments] == [
        "expert_hip_f16kv_mtp_n4_dgpu",
        "expert_hip_f16kv_mtp_n3_dgpu",
        "expert_hip_f16kv_mtp_n2_dgpu",
    ]
    assert [
        option_value(server_command(expanded_shipped_config, mtp_window_tier, item)[1:], "--spec-draft-n-max")
        for item in mtp_window_experiments
    ] == ["4", "3", "2"]
    mtp_prefill_tier = expanded_shipped_config["tiers"]["rocm-mtp-prefill"]
    mtp_prefill_experiments = select_experiments(expanded_shipped_config, mtp_prefill_tier, None)
    assert [item["name"] for item in mtp_prefill_experiments] == [
        "expert_hip_f16kv_mtp_n3_dgpu",
        "expert_hip_f16kv_mtp_n3_dgpu_ub1024",
        "expert_hip_f16kv_mtp_n3_dgpu_ub2048",
        "expert_hip_f16kv_mtp_n3_dgpu_q8draftkv",
    ]
    assert all(
        option_value(server_command(expanded_shipped_config, mtp_prefill_tier, item)[1:], "--spec-draft-n-max") == "3"
        for item in mtp_prefill_experiments
    )
    assert [
        option_value(server_command(expanded_shipped_config, mtp_prefill_tier, item)[1:], "--ubatch-size")
        for item in mtp_prefill_experiments[:3]
    ] == ["512", "1024", "2048"]
    q8_draft_args = server_command(expanded_shipped_config, mtp_prefill_tier, mtp_prefill_experiments[-1])[1:]
    assert option_value(q8_draft_args, "--cache-type-k") == "f16"
    assert option_value(q8_draft_args, "--cache-type-v") == "f16"
    assert option_value(q8_draft_args, "--spec-draft-type-k") == "q8_0"
    assert option_value(q8_draft_args, "--spec-draft-type-v") == "q8_0"
    mtp_finalists_tier = expanded_shipped_config["tiers"]["rocm-mtp-finalists"]
    mtp_finalists = select_experiments(expanded_shipped_config, mtp_finalists_tier, None)
    assert [item["name"] for item in mtp_finalists] == [
        "expert_hip_f16kv_no_mtp_ub2048",
        "expert_hip_f16kv_mtp_n3_dgpu_ub2048",
        "expert_hip_f16kv_mtp_n3_dgpu_ub2048_q8draftkv",
        "expert_hip_f16kv_no_mtp_ub4096",
        "expert_hip_f16kv_mtp_n3_dgpu_ub4096_q8draftkv",
    ]
    finalist_args = [
        server_command(expanded_shipped_config, mtp_finalists_tier, item)[1:]
        for item in mtp_finalists
    ]
    assert [option_value(args, "--ubatch-size") for args in finalist_args] == [
        "2048", "2048", "2048", "4096", "4096",
    ]
    assert [option_value(args, "--batch-size") for args in finalist_args] == [
        "2048", "2048", "2048", "4096", "4096",
    ]
    for args in (finalist_args[1], finalist_args[2], finalist_args[4]):
        assert option_value(args, "--spec-draft-n-max") == "3"
        assert option_value(args, "--spec-draft-device") == "ROCm0"
    for args in (finalist_args[2], finalist_args[4]):
        assert option_value(args, "--spec-draft-type-k") == "q8_0"
        assert option_value(args, "--spec-draft-type-v") == "q8_0"
    rocm_ple_tier = expanded_shipped_config["tiers"]["rocm-ple-ssd"]
    assert rocm_ple_tier.get("require_rocm_audit") is True
    assert rocm_ple_tier.get("require_rocm_mtp") is True
    assert int(rocm_ple_tier["warmup_depth"]) >= max(int(x) for x in rocm_ple_tier["depths"])
    rocm_ple_experiments = select_experiments(expanded_shipped_config, rocm_ple_tier, None)
    assert [item["name"] for item in rocm_ple_experiments] == [
        "expert_hip_f16kv_mtp_n3_dgpu_ub2048",
        "expert_hip_f16kv_mtp_n3_dgpu_ub2048_ple_cpu",
    ]
    rocm_ple_args = [
        server_command(expanded_shipped_config, rocm_ple_tier, item)[1:]
        for item in rocm_ple_experiments
    ]
    assert all("--mmap" in args and "--load-mode" not in args for args in rocm_ple_args)
    assert all(option_value(args, "--spec-draft-device") == "ROCm0" for args in rocm_ple_args)
    assert all(option_value(args, "--spec-draft-n-max") == "3" for args in rocm_ple_args)
    assert all(option_value(args, "--spec-draft-type-k") == "f16" for args in rocm_ple_args)
    assert all(option_value(args, "--spec-draft-type-v") == "f16" for args in rocm_ple_args)
    rocm_ple_override = option_value(rocm_ple_args[1], "--override-tensor") or ""
    assert "ffn_(down|gate|up)_exps" in rocm_ple_override
    assert "per_layer_token_embd" in rocm_ple_override and "=CPU" in rocm_ple_override
    placement_names = [
        "prod_hip_256k_base",
        "prod_hip_256k_token_igpu",
        "prod_hip_256k_token_shared_igpu",
        "prod_hip_256k_token_shared_output_igpu",
        "prod_hip_256k_token_shared_output_fullattn_igpu",
    ]
    tail_names = [
        "prod_hip_256k_base",
        "prod_hip_256k_tail_88_12",
        "prod_hip_256k_tail_84_16",
        "prod_hip_256k_tail_76_24",
    ]
    fit_capacity_names = [
        "prod_hip_256k_tail_87_13",
        "prod_hip_256k_tail_86_14",
        "prod_hip_256k_tail_85_15",
        "prod_hip_256k_tail_88_12_ub1792",
        "prod_hip_256k_tail_88_12_ub1536",
        "prod_hip_256k_tail_88_12_ub1024",
    ]
    fit_quality_names = [
        "prod_hip_256k_tail_88_12_ub1792",
        "prod_hip_256k_tail_88_12_ub1536",
        "prod_hip_256k_tail_88_12_ub1024",
    ]
    fit_full_names = [
        "prod_hip_256k_tail_88_12_ub1536",
        "prod_hip_256k_tail_88_12_ub1024",
    ]
    target_fingerprint_names = [
        "prod_hip_256k_tail_88_12_no_mtp_ub2048",
        "prod_hip_256k_tail_88_12_no_mtp_ub1792",
        "prod_hip_256k_tail_88_12_no_mtp_ub1536",
        "prod_hip_256k_tail_88_12_no_mtp_ub1024",
        "prod_hip_256k_tail_88_12_no_mtp_ub512",
    ]
    mtp_repeatability_names = [
        "prod_hip_256k_tail_88_12",
        "prod_hip_256k_tail_88_12_ub1792",
        "prod_hip_256k_tail_88_12_ub1536",
        "prod_hip_256k_tail_88_12_ub1024",
        "prod_hip_256k_tail_88_12_ub512",
    ]
    production_base = next(
        item for item in expanded_shipped_config["experiments"]
        if item["name"] == "prod_hip_256k_base"
    )
    production_base_args = server_command(
        expanded_shipped_config,
        expanded_shipped_config["tiers"]["rocm-256k-fit-full"],
        production_base,
    )[1:]
    assert option_value(production_base_args, "--cache-ram") == "0"
    assert option_value(production_base_args, "--ctx-checkpoints") == "8"
    assert option_value(production_base_args, "--checkpoint-every-n-tokens") == "32768"
    assert "--checkpoint-min-step" not in production_base_args
    assert production_base.get("env", {}).get(HOST_CHECKPOINT_MARKER) == "1"
    ple_storage_screen = expanded_shipped_config["tiers"]["rocm-ple-storage-screen"]
    ple_ram_capacity = expanded_shipped_config["tiers"]["rocm-ple-ram-capacity"]
    ple_storage_full = expanded_shipped_config["tiers"]["rocm-ple-storage-full"]
    ple_storage_names = [
        "prod_hip_256k_tail_88_12_ub1536",
        "prod_hip_256k_tail_88_12_ub1536_ple_ram",
    ]
    assert ple_storage_screen["experiments"] == ple_storage_names
    assert ple_storage_full["experiments"] == ple_storage_names
    assert ple_ram_capacity["experiments"] == [ple_storage_names[1]]
    assert int(ple_storage_screen["ctx_size"]) == 65536
    assert ple_ram_capacity.get("startup_only") is True
    assert int(ple_ram_capacity["ctx_size"]) == 262144
    assert ple_storage_full.get("exact_prompt_tokens") is True
    assert ple_storage_full["depths"] == [253952]
    ple_storage_experiments = select_experiments(
        expanded_shipped_config, ple_storage_screen, None,
    )
    ple_storage_commands = [
        server_command(expanded_shipped_config, ple_storage_screen, item)
        for item in ple_storage_experiments
    ]
    assert "--mmap" in ple_storage_commands[0] and "--no-mmap" not in ple_storage_commands[0]
    assert "--no-mmap" in ple_storage_commands[1] and "--mmap" not in ple_storage_commands[1]
    assert [
        token for token in ple_storage_commands[0] if token not in {"--mmap", "--no-mmap"}
    ] == [
        token for token in ple_storage_commands[1] if token not in {"--mmap", "--no-mmap"}
    ]
    for experiment, command in zip(ple_storage_experiments, ple_storage_commands):
        override = option_value(command[1:], "--override-tensor") or ""
        assert "per_layer_token_embd" in override and "=CPU" in override
        assert experiment.get("env", {}).get(HOST_CHECKPOINT_MARKER) == "1"
    placement_screen = expanded_shipped_config["tiers"]["rocm-256k-placement-screen"]
    tail_screen = expanded_shipped_config["tiers"]["rocm-256k-tail-screen"]
    capacity_screen = expanded_shipped_config["tiers"]["rocm-256k-capacity"]
    full_context = expanded_shipped_config["tiers"]["rocm-256k-full"]
    fit_capacity = expanded_shipped_config["tiers"]["rocm-256k-fit-capacity"]
    fit_quality = expanded_shipped_config["tiers"]["rocm-256k-fit-quality"]
    fit_full = expanded_shipped_config["tiers"]["rocm-256k-fit-full"]
    target_fingerprint = expanded_shipped_config["tiers"]["rocm-ubatch-target-fingerprint"]
    target_correctness = expanded_shipped_config["tiers"]["rocm-ubatch-target-correctness"]
    mtp_repeatability = expanded_shipped_config["tiers"]["rocm-ubatch-mtp-repeatability"]
    quality_target = expanded_shipped_config["tiers"]["rocm-ubatch-quality-target-screen"]
    quality_mtp = expanded_shipped_config["tiers"]["rocm-ubatch-quality-mtp-screen"]
    assert placement_screen["experiments"] == placement_names
    assert tail_screen["experiments"] == tail_names
    assert capacity_screen["experiments"] == tail_names
    assert full_context["experiments"] == tail_names
    assert fit_capacity["experiments"] == fit_capacity_names
    assert fit_quality["experiments"] == fit_quality_names
    assert fit_full["experiments"] == fit_full_names
    assert target_fingerprint["experiments"] == target_fingerprint_names
    assert target_correctness["experiments"] == target_fingerprint_names
    assert mtp_repeatability["experiments"] == mtp_repeatability_names
    assert quality_target["experiments"] == target_fingerprint_names
    assert quality_mtp["experiments"] == mtp_repeatability_names
    assert quality_target.get("mode") == "quality"
    assert quality_mtp.get("mode") == "quality"
    assert len(quality_target["quality_cases"]) == 8
    assert quality_target["quality_cases"] == quality_mtp["quality_cases"]
    assert quality_target.get("exact_prompt_tokens") is True
    assert quality_mtp.get("exact_prompt_tokens") is True
    assert int(quality_target["n_predict"]) == 256
    assert int(quality_mtp["n_predict"]) == 256
    assert quality_target["request"]["ignore_eos"] is False
    assert quality_mtp["request"]["ignore_eos"] is False
    assert int(placement_screen["ctx_size"]) == 65536
    assert int(tail_screen["ctx_size"]) == 65536
    assert int(capacity_screen["ctx_size"]) == 262144
    assert capacity_screen.get("startup_only") is True
    assert int(capacity_screen["warmups"]) == 0
    assert int(full_context["ctx_size"]) == 262144
    assert full_context.get("exact_prompt_tokens") is True
    assert max(int(value) for value in full_context["depths"]) == 253952
    assert int(fit_capacity["ctx_size"]) == 262144
    assert fit_capacity.get("startup_only") is True
    assert int(fit_capacity["warmups"]) == 0
    assert int(fit_quality["ctx_size"]) == 65536
    assert int(fit_quality["warmup_depth"]) == 32768
    assert int(fit_full["ctx_size"]) == 262144
    assert fit_full.get("exact_prompt_tokens") is True
    assert [int(value) for value in fit_full["depths"]] == [253952]
    assert target_fingerprint["depths"] == [32768]
    assert int(target_fingerprint["rounds"]) == 2
    assert int(target_fingerprint["n_predict"]) == 4
    assert int(target_fingerprint["request"]["n_probs"]) == 20
    assert target_fingerprint.get("require_probability_metrics") is True
    assert "text_quality_anchors" not in target_fingerprint
    assert target_correctness["depths"] == [32768]
    assert int(target_correctness["rounds"]) == 2
    assert int(target_correctness["n_predict"]) == 512
    assert "n_probs" not in target_correctness.get("request", {})
    assert target_correctness["text_quality_anchors"]["code"] == [
        "def merge_intervals(", "return",
    ]
    assert mtp_repeatability["depths"] == [32768]
    assert int(mtp_repeatability["rounds"]) == 2
    placement_experiments = select_experiments(
        expanded_shipped_config, placement_screen, None,
    )
    tail_experiments = select_experiments(expanded_shipped_config, tail_screen, None)
    for tier, experiments in (
        (placement_screen, placement_experiments),
        (tail_screen, tail_experiments),
    ):
        for experiment in experiments:
            args = server_command(expanded_shipped_config, tier, experiment)[1:]
            assert option_value(args, "--cache-ram") == "0"
            assert option_value(args, "--batch-size") == "2048"
            assert option_value(args, "--ubatch-size") == "2048"
            assert option_value(args, "--parallel") == "1"
            assert option_value(args, "--cache-type-k") == "f16"
            assert option_value(args, "--cache-type-v") == "f16"
            assert option_value(args, "--spec-draft-type-k") == "f16"
            assert option_value(args, "--spec-draft-type-v") == "f16"
            assert option_value(args, "--spec-draft-device") == "ROCm0"
            assert option_value(args, "--spec-draft-n-max") == "3"
            assert "--no-kv-unified" in args and "--cont-batching" in args
            override = option_value(args, "--override-tensor") or ""
            assert (
                "ffn_(down|gate|up)_exps" in override
                or "ffn_(down|gate|up)_(exps|shexp)" in override
            ) and "=ROCm1" in override
            assert "per_layer_token_embd" in override and "=CPU" in override
    production_overrides = [
        option_value(server_command(expanded_shipped_config, placement_screen, item)[1:], "--override-tensor") or ""
        for item in placement_experiments
    ]
    assert "^token_embd\\.weight" not in production_overrides[0]
    assert "^token_embd\\.weight" in production_overrides[1]
    assert "shexp" in production_overrides[2]
    assert "output" in production_overrides[3]
    assert "output" in production_overrides[4]
    assert "3|7|11|15|19|23|27|31|35|39|43|47" in production_overrides[4]
    tail_args = [
        server_command(expanded_shipped_config, tail_screen, item)[1:]
        for item in tail_experiments
    ]
    assert all(option_value(args, "--device") == "ROCm0,ROCm1" for args in tail_args)
    assert [option_value(args, "--tensor-split") for args in tail_args] == [
        "1,0", "88,12", "84,16", "76,24",
    ]
    fit_experiments = select_experiments(expanded_shipped_config, fit_capacity, None)
    fit_args = [
        server_command(expanded_shipped_config, fit_capacity, item)[1:]
        for item in fit_experiments
    ]
    assert all(option_value(args, "--device") == "ROCm0,ROCm1" for args in fit_args)
    assert [option_value(args, "--tensor-split") for args in fit_args] == [
        "87,13", "86,14", "85,15", "88,12", "88,12", "88,12",
    ]
    assert [option_value(args, "--ubatch-size") for args in fit_args] == [
        "2048", "2048", "2048", "1792", "1536", "1024",
    ]
    for args in fit_args:
        assert option_value(args, "--cache-ram") == "0"
        assert option_value(args, "--batch-size") == "2048"
        assert option_value(args, "--cache-type-k") == "f16"
        assert option_value(args, "--cache-type-v") == "f16"
        assert option_value(args, "--spec-draft-type-k") == "f16"
        assert option_value(args, "--spec-draft-type-v") == "f16"
        assert option_value(args, "--spec-draft-device") == "ROCm0"
        assert option_value(args, "--spec-draft-n-max") == "3"
        override = option_value(args, "--override-tensor") or ""
        assert "ffn_(down|gate|up)_exps" in override and "=ROCm1" in override
        assert "per_layer_token_embd" in override and "=CPU" in override
    target_experiments = select_experiments(
        expanded_shipped_config, target_fingerprint, None,
    )
    target_args = [
        server_command(expanded_shipped_config, target_fingerprint, item)[1:]
        for item in target_experiments
    ]
    assert [option_value(args, "--ubatch-size") for args in target_args] == [
        "2048", "1792", "1536", "1024", "512",
    ]
    for args in target_args:
        assert "-md" not in args
        assert option_value(args, "--tensor-split") == "88,12"
        assert option_value(args, "--cache-ram") == "0"
        assert option_value(args, "--cache-type-k") == "f16"
        assert option_value(args, "--cache-type-v") == "f16"
        override = option_value(args, "--override-tensor") or ""
        assert "ffn_(down|gate|up)_exps" in override and "=ROCm1" in override
        assert "per_layer_token_embd" in override and "=CPU" in override
    correctness_experiments = select_experiments(
        expanded_shipped_config, target_correctness, None,
    )
    correctness_args = [
        server_command(expanded_shipped_config, target_correctness, item)[1:]
        for item in correctness_experiments
    ]
    assert correctness_args == target_args
    mtp_experiments = select_experiments(
        expanded_shipped_config, mtp_repeatability, None,
    )
    mtp_args = [
        server_command(expanded_shipped_config, mtp_repeatability, item)[1:]
        for item in mtp_experiments
    ]
    assert [option_value(args, "--ubatch-size") for args in mtp_args] == [
        "2048", "1792", "1536", "1024", "512",
    ]
    for args in mtp_args:
        assert "-md" in args
        assert option_value(args, "--tensor-split") == "88,12"
        assert option_value(args, "--spec-draft-device") == "ROCm0"
        assert option_value(args, "--spec-draft-n-max") == "3"
        assert option_value(args, "--spec-draft-type-k") == "f16"
        assert option_value(args, "--spec-draft-type-v") == "f16"
    rocm_vision_smoke = expanded_shipped_config["tiers"]["rocm-vision-smoke"]
    assert rocm_vision_smoke.get("mode") == "vision"
    assert rocm_vision_smoke.get("require_rocm_audit") is True
    assert rocm_vision_smoke.get("require_rocm_mtp") is True
    assert rocm_vision_smoke.get("require_rocm_mtp_vision") is True
    rocm_vision_experiments = select_experiments(
        expanded_shipped_config, rocm_vision_smoke, None,
    )
    assert [item["name"] for item in rocm_vision_experiments] == [
        "vision_bf16_cpu_hip_no_mtp",
        "vision_bf16_igpu_hip_no_mtp",
        "vision_bf16_igpu_hip_mtp_n3",
    ]
    rocm_vision_args = [
        server_command(expanded_shipped_config, rocm_vision_smoke, item)[1:]
        for item in rocm_vision_experiments
    ]
    assert all(item.get("backend") == "rocm" for item in rocm_vision_experiments)
    assert all(item.get("env", {}).get(HOST_CHECKPOINT_MARKER) == "1" for item in rocm_vision_experiments)
    assert all(option_value(args, "--mmproj").endswith("-BF16.gguf") for args in rocm_vision_args)
    assert "--no-mmproj-offload" in rocm_vision_args[0]
    assert all("--mmproj-offload" in args for args in rocm_vision_args[1:])
    assert all(
        item.get("env", {}).get("MTMD_BACKEND_DEVICE") == "ROCm1"
        for item in rocm_vision_experiments[1:]
    )
    rocm_vision_mtp_args = rocm_vision_args[-1]
    assert option_value(rocm_vision_mtp_args, "--spec-draft-device") == "ROCm0"
    assert option_value(rocm_vision_mtp_args, "--spec-draft-n-max") == "3"
    assert option_value(rocm_vision_mtp_args, "--spec-draft-type-k") == "f16"
    assert option_value(rocm_vision_mtp_args, "--spec-draft-type-v") == "f16"
    assert "--spec-mtp-strict-qwen4exp-vision" in rocm_vision_mtp_args
    assert all("--spec-mtp-strict-qwen4exp-vision" not in args for args in rocm_vision_args[:-1])
    mtp_resync_tier = expanded_shipped_config["tiers"]["rocm-vision-mtp-resync-smoke"]
    assert mtp_resync_tier.get("require_rocm_mtp_vision") is True
    mtp_resync_experiments = select_experiments(expanded_shipped_config, mtp_resync_tier, None)
    assert [item["name"] for item in mtp_resync_experiments] == [
        "vision_bf16_igpu_hip_mtp_n3",
    ]
    mtp_strict_ab_tier = expanded_shipped_config["tiers"]["rocm-vision-mtp-strict-ab"]
    assert mtp_strict_ab_tier.get("require_rocm_mtp_vision") is True
    assert int(mtp_strict_ab_tier["rounds"]) == 2
    mtp_strict_ab_experiments = select_experiments(
        expanded_shipped_config, mtp_strict_ab_tier, None,
    )
    assert [item["name"] for item in mtp_strict_ab_experiments] == [
        "vision_bf16_igpu_hip_no_mtp_strict_ab",
        "vision_bf16_igpu_hip_mtp_n3_strict",
    ]
    assert mtp_strict_ab_experiments[0].get("baseline") is True
    assert mtp_strict_ab_experiments[1].get("baseline") is False
    mtp_strict_ab_args = [
        server_command(expanded_shipped_config, mtp_strict_ab_tier, item)[1:]
        for item in mtp_strict_ab_experiments
    ]
    assert all(
        option_value(args, "--checkpoint-every-n-tokens") == "32768"
        for args in mtp_strict_ab_args
    )
    assert "--spec-mtp-strict-qwen4exp-vision" not in mtp_strict_ab_args[0]
    assert "--spec-mtp-strict-qwen4exp-vision" in mtp_strict_ab_args[1]
    request_disable_tier = expanded_shipped_config["tiers"]["rocm-vision-mtp-request-disable-ab"]
    assert request_disable_tier.get("require_rocm_mtp") is True
    assert request_disable_tier.get("require_rocm_mtp_vision") is not True
    request_disable_experiments = select_experiments(
        expanded_shipped_config, request_disable_tier, None,
    )
    assert [item["name"] for item in request_disable_experiments] == [
        "vision_bf16_igpu_hip_no_mtp_strict_ab",
        "vision_bf16_igpu_hip_mtp_loaded_request_disabled",
    ]
    request_disable_args = [
        server_command(expanded_shipped_config, request_disable_tier, item)[1:]
        for item in request_disable_experiments
    ]
    assert "-md" not in request_disable_args[0]
    assert "-md" in request_disable_args[1]
    assert all("--spec-mtp-strict-qwen4exp-vision" not in args for args in request_disable_args)
    assert request_disable_experiments[1].get("require_zero_draft") is True
    assert merged_request(
        expanded_shipped_config, request_disable_tier, request_disable_experiments[1],
    ).get("speculative.n_max") == 0
    rocm_vision_full = expanded_shipped_config["tiers"]["rocm-vision"]
    assert rocm_vision_full.get("require_rocm_mtp_vision") is True
    assert int(rocm_vision_full["ctx_size"]) == 262144
    rocm_vision_full_experiments = select_experiments(
        expanded_shipped_config, rocm_vision_full, None,
    )
    assert [item["name"] for item in rocm_vision_full_experiments] == [
        "vision_bf16_igpu_prod_hip_no_mtp",
        "vision_bf16_igpu_prod_hip_mtp_n3",
    ]
    for item in rocm_vision_full_experiments:
        full_args = server_command(expanded_shipped_config, rocm_vision_full, item)[1:]
        assert option_value(full_args, "--ctx-size") == "262144"
        assert option_value(full_args, "--tensor-split") == "88,12"
        assert option_value(full_args, "--ubatch-size") == "1536"
        assert option_value(full_args, "--cache-type-k") == "f16"
        assert option_value(full_args, "--cache-type-v") == "f16"
        assert option_value(full_args, "--checkpoint-every-n-tokens") == "32768"
        assert option_value(full_args, "--mmproj").endswith("-BF16.gguf")
        assert "--mmproj-offload" in full_args
        assert item.get("env", {}).get("MTMD_BACKEND_DEVICE") == "ROCm1"
        assert item.get("env", {}).get(HOST_CHECKPOINT_MARKER) == "1"
    assert "-md" not in server_command(
        expanded_shipped_config, rocm_vision_full, rocm_vision_full_experiments[0],
    )[1:]
    assert "--spec-mtp-strict-qwen4exp-vision" not in server_command(
        expanded_shipped_config, rocm_vision_full, rocm_vision_full_experiments[0],
    )[1:]
    full_mtp_args = server_command(
        expanded_shipped_config, rocm_vision_full, rocm_vision_full_experiments[1],
    )[1:]
    assert option_value(full_mtp_args, "--spec-draft-device") == "ROCm0"
    assert option_value(full_mtp_args, "--spec-draft-n-max") == "3"
    assert option_value(full_mtp_args, "--spec-draft-type-k") == "f16"
    assert option_value(full_mtp_args, "--spec-draft-type-v") == "f16"
    assert "--spec-mtp-strict-qwen4exp-vision" in full_mtp_args
    for tier_name in (
        "rocm-ple-ssd",
        "rocm-ple-storage-screen",
        "rocm-ple-ram-capacity",
        "rocm-ple-storage-full",
        "rocm-256k-placement-screen",
        "rocm-256k-tail-screen",
        "rocm-256k-capacity",
        "rocm-256k-full",
        "rocm-vision-smoke",
        "rocm-vision-mtp-resync-smoke",
        "rocm-vision",
    ):
        production_tier = expanded_shipped_config["tiers"][tier_name]
        for experiment in select_experiments(expanded_shipped_config, production_tier, None):
            production_args = server_command(
                expanded_shipped_config, production_tier, experiment,
            )[1:]
            assert option_value(production_args, "--spec-draft-type-k") != "q8_0"
            assert option_value(production_args, "--spec-draft-type-v") != "q8_0"
    mtp_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-qwen4exp-mtp.patch"
    assert mtp_patch.is_file()
    mtp_patch_text = mtp_patch.read_text(encoding="utf-8")
    assert QWEN4EXP_MTP_MARKER in mtp_patch_text
    assert "src/models/qwen4exp.cpp" in mtp_patch_text
    assert "src/llama-model.cpp" in mtp_patch_text
    mtp_sched_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-qwen4exp-mtp-schedule-output.patch"
    assert mtp_sched_patch.is_file()
    assert QWEN4EXP_MTP_SCHED_MARKER in mtp_sched_patch.read_text(encoding="utf-8")
    mtp_vision_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-mtp-vision-resync.patch"
    assert mtp_vision_patch.is_file()
    mtp_vision_patch_text = mtp_vision_patch.read_text(encoding="utf-8")
    assert MTP_VISION_RESYNC_MARKER in mtp_vision_patch_text
    assert "common_speculative_need_embd_pre_norm" in mtp_vision_patch_text
    mtp_vision_strict_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-qwen4exp-vision-strict.patch"
    assert mtp_vision_strict_patch.is_file()
    mtp_vision_strict_patch_text = mtp_vision_strict_patch.read_text(encoding="utf-8")
    assert QWEN4EXP_VISION_STRICT_MARKER in mtp_vision_strict_patch_text
    assert "strict_qwen4exp_vision_mtp_verification" in mtp_vision_strict_patch_text
    assert "llama_decode_with_ubatch(ctx_tgt, batch_view, 1)" in mtp_vision_strict_patch_text
    mtp_vision_checkpoint_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-qwen4exp-vision-strict-checkpoint.patch"
    assert mtp_vision_checkpoint_patch.is_file()
    mtp_vision_checkpoint_patch_text = mtp_vision_checkpoint_patch.read_text(encoding="utf-8")
    assert QWEN4EXP_VISION_CHECKPOINT_MARKER in mtp_vision_checkpoint_patch_text
    assert "mtp_strict_qwen4exp_vision ? 0" in mtp_vision_checkpoint_patch_text
    assert "--spec-mtp-strict-qwen4exp-vision" in mtp_vision_checkpoint_patch_text
    host_checkpoint_patch = pathlib.Path(__file__).with_name("patches") / "rocmfpx-host-checkpoints.patch"
    assert host_checkpoint_patch.is_file()
    host_checkpoint_patch_text = host_checkpoint_patch.read_text(encoding="utf-8")
    assert HOST_CHECKPOINT_MARKER in host_checkpoint_patch_text
    assert "~LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in host_checkpoint_patch_text
    assert "forcing checkpoint state to host memory" in host_checkpoint_patch_text
    assert not re.search(r"^@@ -\d+,0 ", host_checkpoint_patch_text, re.MULTILINE)
    assert host_checkpoint_patch_text.count(
        "flags = common_prompt_checkpoint_maybe_host_flags(flags);"
    ) == 4
    build_script = pathlib.Path(__file__).with_name("build-rocm10-dual.sh").read_text(encoding="utf-8")
    assert "git -C \"$SOURCE_DIR\" apply" in build_script
    assert QWEN4EXP_MTP_MARKER in build_script
    assert QWEN4EXP_MTP_SCHED_MARKER in build_script
    assert HOST_CHECKPOINT_MARKER in build_script
    assert MTP_VISION_RESYNC_MARKER in build_script
    assert QWEN4EXP_VISION_STRICT_MARKER in build_script
    assert QWEN4EXP_VISION_CHECKPOINT_MARKER in build_script
    assert "rocmfpx-qwen4exp-vision-strict.patch" in build_script
    assert "rocmfpx-qwen4exp-vision-strict-checkpoint.patch" in build_script
    assert 'SERVER_BINARY="$BUILD_DIR/bin/llama-server"' in build_script
    assert "libllama-server-impl.so" not in build_script
    assert "rocmfpx-host-checkpoints-v1-broken.patch" in build_script
    assert "apply --reverse --check --unidiff-zero" in build_script
    with tempfile.TemporaryDirectory() as raw_tmp:
        fake_bin = pathlib.Path(raw_tmp) / "bin"
        fake_bin.mkdir()
        fake_server = fake_bin / "llama-server"
        fake_server.write_bytes(
            b"runtime\x00" + MTP_VISION_RESYNC_MARKER.encode("ascii") + b"\x00" +
            QWEN4EXP_VISION_STRICT_MARKER.encode("ascii") + b"\x00" +
            QWEN4EXP_VISION_CHECKPOINT_MARKER.encode("ascii")
        )
        fake_fingerprint = rocm_build_fingerprint({
            "variables": {"hip_server": str(fake_server)},
        })
        assert fake_fingerprint["server"]["mtp_vision_resync_marker"] is True
        assert fake_fingerprint["server"]["qwen4exp_vision_strict_marker"] is True
        assert fake_fingerprint["server"]["qwen4exp_vision_checkpoint_marker"] is True
        assert "server_impl_library" not in fake_fingerprint
    print(f"qwen_bench.py {VERSION}: self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark tier")
    run.add_argument("--config", default=str(pathlib.Path(__file__).with_name("matrix.json")))
    run.add_argument("--tier", default="smoke")
    run.add_argument("--experiments", help="comma-separated experiment names or glob patterns")
    run.add_argument("--output-root", default=str(pathlib.Path(__file__).with_name("results")))
    run.add_argument("--resume", help="resume an existing run directory")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-path-check", action="store_true", help="only for reviewing commands off-host")
    run.add_argument("--allow-busy", action="store_true", help="allow other llama processes; results may be invalid")
    run.add_argument("--capture-system", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--fail-fast", action="store_true")

    check = sub.add_parser("preflight", help="validate files, devices, memory, and port without loading a model")
    check.add_argument("--config", default=str(pathlib.Path(__file__).with_name("matrix.json")))
    check.add_argument("--tier", default="smoke")
    check.add_argument("--experiments")
    check.add_argument("--output", default=str(pathlib.Path(__file__).with_name("preflight")))
    check.add_argument("--skip-path-check", action="store_true")
    check.add_argument("--allow-busy", action="store_true")
    check.add_argument("--capture-system", action=argparse.BooleanOptionalAction, default=True)

    summary = sub.add_parser("summarize", help="regenerate CSV and Markdown from results.jsonl")
    summary.add_argument("run_dir")

    rocm = sub.add_parser(
        "rocm-audit",
        help="prove ROCmFP4 source dispatch and filtered HIP backend-op correctness before model loading",
    )
    rocm.add_argument("--config", default=str(pathlib.Path(__file__).with_name("matrix.json")))
    rocm.add_argument(
        "--output", default=str(pathlib.Path(__file__).with_name("preflight") / "rocm-audit.json"),
    )
    rocm.add_argument("--devices", default="ROCm0,ROCm1")
    rocm.add_argument("--run-ops", action="store_true")
    rocm.add_argument("--timeout", type=float, default=600.0, help="timeout per filtered backend-op test")

    archive = sub.add_parser("archive", help="package a complete or partial run as one .tar.gz plus SHA-256")
    archive.add_argument("run_dir")
    archive.add_argument("--output", help="output .tar.gz path; defaults beside the run directory")

    sub.add_parser("self-test", help="run parser and aggregation unit checks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            execute_run(args)
        elif args.command == "preflight":
            execute_preflight(args)
        elif args.command == "summarize":
            summarize(pathlib.Path(args.run_dir).resolve())
        elif args.command == "rocm-audit":
            execute_rocm_audit(args)
        elif args.command == "archive":
            execute_archive(args)
        elif args.command == "self-test":
            self_test()
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("QWEN_BENCH_TRACEBACK") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
