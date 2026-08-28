#!/usr/bin/env python3
"""Config-driven llama.cpp topology and speculative-decoding benchmark harness.

The runner uses only Python's standard library.  It is designed for very large
models where process isolation, cold server starts, exact command capture, and
recoverable partial results matter more than shaving a few seconds off a run.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import csv
import datetime as dt
import fnmatch
import hashlib
import http.client
import json
import math
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
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Iterable


VERSION = "1.12.0"
SUCCESS_STATES = {"ok"}
SINGLE_VALUE_SERVER_OPTIONS = {
    "-m",
    "-md",
    "-ot",
    "--batch-size",
    "--cache-ram",
    "--cache-type-k",
    "--cache-type-v",
    "--ctx-size",
    "--device",
    "--fit",
    "--flash-attn",
    "--host",
    "--load-mode",
    "--main-gpu",
    "--n-gpu-layers",
    "--override-tensor",
    "--parallel",
    "--port",
    "--spec-draft-device",
    "--spec-draft-ngl",
    "--spec-draft-n-max",
    "--spec-draft-p-min",
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
    """Build a prompt at or just below an exact per-slot token budget."""
    lane_prefix = f"Independent benchmark lane {lane + 1}; lane marker {lane:08x}.\n"
    if target_tokens <= 0:
        prompt = lane_prefix + base
        return prompt, tokenize_count(base_url, prompt, timeout_s)

    low_chars = 0
    high_chars = max(1024, target_tokens * 5)

    def construct(filler_chars: int) -> str:
        prefix = "Reference material follows. Read it, then answer the task after END REFERENCE.\n\n"
        repeated = (corpus + "\n\n") * max(1, math.ceil(filler_chars / max(1, len(corpus))))
        return f"{lane_prefix}{prefix}{repeated[:filler_chars]}\nEND REFERENCE\n\n{base}"

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

    best_prompt, best_count = low_prompt, low_count
    for _ in range(8):
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
        if count <= target_tokens:
            low_chars, low_prompt, low_count = probe_chars, prompt, count
            if count > best_count:
                best_prompt, best_count = prompt, count
            if target_tokens - count <= 16:
                break
        else:
            high_chars, high_prompt, high_count = probe_chars, prompt, count
    return best_prompt, best_count


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
    return [command[0], *canonicalize_server_args(command[1:])]


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


def response_slot_id(response: dict[str, Any], fallback: int = 0) -> int:
    """Return the llama-server slot used by a completion response."""
    for field in ("id_slot", "slot_id"):
        value = response.get(field)
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
    return {
        "lanes": len(responses),
        "wall_ms": round(wall_ms, 3),
        "predicted_n_total": int(sum(predicted_n)) if predicted_n else None,
        "prompt_n_total": int(sum(prompt_n)) if prompt_n else None,
        "aggregate_decode_tok_s": (
            sum(predicted_n) * 1000.0 / max(predicted_ms) if predicted_n and predicted_ms and max(predicted_ms) > 0 else None
        ),
        "aggregate_prefill_tok_s": (
            sum(prompt_n) * 1000.0 / max(prompt_ms) if prompt_n and prompt_ms and max(prompt_ms) > 0 else None
        ),
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


def rocm_build_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    server = pathlib.Path(str(config.get("variables", {}).get("hip_server", "")))
    build_root = server.parent.parent if server.name else pathlib.Path()
    candidates = [build_root / "bin" / "libggml-hip.so", build_root / "lib" / "libggml-hip.so"]
    library = next((path for path in candidates if path.is_file()), candidates[0])
    test_binary = build_root / "bin" / "test-backend-ops"
    result: dict[str, Any] = {}
    for name, path in (("server", server), ("hip_library", library), ("test_backend_ops", test_binary)):
        result[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path),
        }
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
    repo = pathlib.Path(str(config.get("variables", {}).get("repo", "")))
    fingerprint = rocm_build_fingerprint(config)
    source = inspect_rocmfp4_sources(repo)
    server = pathlib.Path(fingerprint["server"]["path"])
    test_binary = pathlib.Path(fingerprint["test_backend_ops"]["path"])
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
    if run_ops and source["static_dispatch_ready"] and test_binary.is_file():
        for device in devices:
            for operation in ("MUL_MAT", "MUL_MAT_ID"):
                for quant_type in ("q8_0", "q4_0_rocmfp4", "q4_0_rocmfp4_fast"):
                    command = [
                        str(test_binary), "test", "-b", device, "-o", operation,
                        "-p", f"type_a={quant_type}",
                    ]
                    capture = run_capture(command, env=env, timeout=timeout_s)
                    passed, passed_n, total_n = backend_ops_passed(capture)
                    log_name = safe_name(f"{device}-{operation}-{quant_type}") + ".txt"
                    write_capture(logs / log_name, capture)
                    tests.append({
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
    expected_tests = len(devices) * 2 * 3
    functional_pass = (
        run_ops and len(tests) == expected_tests and expected_tests > 0 and all(item["passed"] for item in tests)
    )
    reasons: list[str] = []
    if not source["static_dispatch_ready"]:
        reasons.append("ROCmFP4 and ROCmFP4_FAST are not both wired into the compiled HIP runtime source tree")
    if not test_binary.is_file():
        reasons.append("build-hip10/bin/test-backend-ops is missing; rebuild with tests enabled")
    if not run_ops:
        reasons.append("functional backend-op tests were not requested; rerun with --run-ops after source integration")
    elif tests and not functional_pass:
        reasons.append("one or more filtered MUL_MAT/MUL_MAT_ID tests failed or matched zero cases")
    report = {
        "schema": 1,
        "ts": utc_now(),
        "harness_version": VERSION,
        "ready_for_model_benchmarks": bool(source["static_dispatch_ready"] and functional_pass),
        "source": source,
        "build_fingerprint": fingerprint,
        "devices_requested": devices,
        "device_list_log": str(pathlib.Path("rocm-audit-logs") / "devices.txt"),
        "functional_tests_requested": run_ops,
        "functional_pass": functional_pass,
        "tests": tests,
        "reasons": reasons,
    }
    atomic_json(output, report)
    return report


def validate_rocm_audit(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
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
    for key in ("server", "hip_library"):
        expected = audited.get(key, {}).get("sha256")
        actual = current.get(key, {}).get("sha256")
        if not expected or expected != actual:
            return report, f"ROCm audit is stale because {key} changed; rerun `python3 qwen_bench.py rocm-audit --run-ops`"
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
    for experiment in experiments:
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
        if int(tier.get("warmups", 0)) < 1 or int(tier.get("warmup_depth", 0)) <= 0:
            errors.append("hot tiers require at least one nonzero-depth warm-up")
        if not bool(tier.get("erase_slot_between_requests", defaults.get("erase_slot_between_requests", False))):
            errors.append("hot tiers must erase slot KV state between warm-up and measured requests")
    if cache_state == "cold" and int(tier.get("warmups", 0)) != 0:
        errors.append("cold tiers must set warmups to 0")
    concurrency = int(tier.get("concurrency", 1))
    if concurrency < 1:
        errors.append("concurrency must be at least 1")
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
    if tier.get("startup_only") and (int(tier.get("warmups", 0)) != 0 or concurrency != 1):
        errors.append("startup-only tiers require warmups 0 and concurrency 1")
    if tier.get("require_rocm_audit"):
        rocm_audit, audit_error = validate_rocm_audit(config)
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
    content = str(response.get("content", ""))
    timing = extract_timing(response)
    predicted_n = timing.get("predicted_n")
    degenerate = not isinstance(predicted_n, (int, float)) or predicted_n < n_predict * 0.95
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
        "stop": response.get("stop"),
        "stopped_eos": response.get("stopped_eos"),
        "stopped_limit": response.get("stopped_limit"),
        "degenerate": degenerate,
        "timing": timing,
        "telemetry": telemetry,
    }


def completed_keys(results_path: pathlib.Path) -> set[tuple[int, str, str, int]]:
    keys: set[tuple[int, str, str, int]] = set()
    for row in read_jsonl(results_path):
        if row.get("status") == "ok" and not row.get("degenerate"):
            keys.add((int(row["round"]), str(row["experiment"]), str(row["workload"]), int(row["requested_depth_tokens"])))
    return keys


def execute_run(args: argparse.Namespace) -> pathlib.Path:
    config_path = pathlib.Path(args.config).resolve()
    config = load_config(config_path)
    if args.tier not in config.get("tiers", {}):
        raise ValueError(f"unknown tier {args.tier!r}; choose from {', '.join(config.get('tiers', {}))}")
    tier = config["tiers"][args.tier]
    experiments = select_experiments(config, tier, args.experiments)
    workloads_all = load_workloads(config, config_path)
    workload_names = list(tier.get("workloads", workloads_all))
    missing_workloads = [name for name in workload_names if name not in workloads_all]
    if missing_workloads:
        raise ValueError(f"tier references unknown workload(s): {missing_workloads}")
    corpus = load_context_corpus(config)
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
    extra_request = effective_request
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
                print(f"\n=== {experiment['name']} round {round_index + 1}/{rounds} ===", flush=True)
                print(command_text(command), flush=True)
                server = ManagedServer(
                    command=command,
                    env=merged_env(config, experiment),
                    log_path=run_dir / "logs" / f"{suffix}.log",
                    health_url=base_url + "/health",
                    startup_timeout_s=float(tier.get("startup_timeout_s", defaults.get("startup_timeout_s", 900))),
                    telemetry_path=run_dir / "telemetry" / f"{suffix}.jsonl",
                    telemetry_interval_s=float(defaults.get("telemetry_interval_s", 1.0)),
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
                            if exact_prompt_tokens:
                                prompt_cache[key] = fit_prompt_to_tokens(
                                    base_url, workloads_all[workload], depth, corpus, request_timeout_s, lane,
                                )
                            else:
                                prompt = make_prompt(workloads_all[workload], depth, corpus)
                                if concurrency > 1:
                                    prompt = f"Independent benchmark lane {lane + 1}; lane marker {lane:08x}.\n" + prompt
                                prompt_cache[key] = (prompt, None)
                        return prompt_cache[key]

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
                        else:
                            warm_prompt = prepared_prompt(workload_names[0], warmup_depth, 0)[0]
                            warm_response, warm_wall_ms = completion_request(
                                base_url + "/completion", warm_prompt, min(32, n_predict),
                                request_timeout_s, extra_request,
                            )
                            warm_responses, warm_walls, warm_group_wall = [warm_response], [warm_wall_ms], warm_wall_ms
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
                        prompt = prepared_prompt(workload, depth, 0)[0]
                        mark = server.telemetry.mark() if server.telemetry else 0
                        response, wall_ms = completion_request(
                            base_url + "/completion", prompt, n_predict, request_timeout_s, extra_request
                        )
                        samples = server.telemetry.slice(mark) if server.telemetry else []
                        row = probe_row(
                            run_id, experiment, round_index, workload, depth, n_predict,
                            response, wall_ms, server.startup_seconds, aggregate_telemetry(samples),
                        )
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
                        if row["degenerate"]:
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
    print(f"\nResults: {run_dir}")
    return run_dir


def median_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(numeric) if numeric else None


def fmt(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def summarize(run_dir: pathlib.Path, config: dict[str, Any] | None = None, experiments: list[dict[str, Any]] | None = None) -> None:
    results = read_jsonl(run_dir / "results.jsonl")
    valid = [row for row in results if row.get("status") in SUCCESS_STATES and not row.get("degenerate")]
    if not valid:
        print(f"No successful, non-degenerate probes to summarize in {run_dir}", file=sys.stderr)
        return
    if config is None:
        manifest = load_json(run_dir / "manifest.json")
        experiments = manifest.get("experiments", [])
    assert experiments is not None
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
        storage_read: list[float] = []
        major_faults: list[float] = []
        mem_available: list[float] = []
        host_cached: list[float] = []
        for row in rows:
            telemetry = row.get("telemetry", {})
            if isinstance(telemetry.get("pid_rss_file_max_bytes"), (int, float)):
                rss_file.append(float(telemetry["pid_rss_file_max_bytes"]))
            if isinstance(telemetry.get("pid_rss_anon_max_bytes"), (int, float)):
                rss_anon.append(float(telemetry["pid_rss_anon_max_bytes"]))
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
            "startup_s_median": median_or_none(row.get("server_startup_seconds") for row in rows),
            "mtp_acceptance_median": median_or_none(acceptance),
            "baseline_output_match": (sum(matches) / len(matches)) if matches else None,
            "pcie_speed_gt_s_max": max(link_speed, default=None),
            "pcie_width_lanes_max": max(link_width, default=None),
            "rss_file_max_gib_median": median_or_none(value / 1024**3 for value in rss_file),
            "rss_anon_max_gib_median": median_or_none(value / 1024**3 for value in rss_anon),
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
    overall: list[dict[str, Any]] = []
    for experiment, rows in experiment_groups.items():
        decode_speedups = [float(row["decode_speedup_vs_baseline"]) for row in rows if row["decode_speedup_vs_baseline"] and row["decode_speedup_vs_baseline"] > 0]
        prefill_speedups = [float(row["prefill_speedup_vs_baseline"]) for row in rows if row["prefill_speedup_vs_baseline"] and row["prefill_speedup_vs_baseline"] > 0]
        matches = [float(row["baseline_output_match"]) for row in rows if row["baseline_output_match"] is not None]
        accepts = [float(row["mtp_acceptance_median"]) for row in rows if row["mtp_acceptance_median"] is not None]
        decode_geomean = math.exp(statistics.fmean(math.log(value) for value in decode_speedups)) if decode_speedups else None
        prefill_geomean = math.exp(statistics.fmean(math.log(value) for value in prefill_speedups)) if prefill_speedups else None
        overall.append({
            "experiment": experiment,
            "cells": len(rows),
            "decode_geomean_speedup": decode_geomean,
            "prefill_geomean_speedup": prefill_geomean,
            "balanced_geomean_speedup": math.sqrt(decode_geomean * prefill_geomean) if decode_geomean and prefill_geomean else None,
            "hash_match": statistics.fmean(matches) if matches else None,
            "mtp_acceptance": statistics.median(accepts) if accepts else None,
        })
    overall.sort(key=lambda row: row["balanced_geomean_speedup"] or -1, reverse=True)
    md = [
        f"# Benchmark summary: {run_dir.name}\n\n",
        f"Baseline for output hashes: `{baseline_name}`. Medians exclude warm-ups and degenerate responses.\n\n",
        f"Declared cache state: `{load_json(run_dir / 'manifest.json').get('tier', {}).get('cache_state', 'unspecified')}`. "
        "Physical storage reads and major faults remain authoritative; a declared hot run is not hot if those counters stay high.\n\n",
        "## Overall comparable-cell ranking\n\n",
        "| Experiment | Cells | Decode speedup | Prefill speedup | Balanced | Hash match | Median MTP accept |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in overall:
        decode_speedup = "" if row["decode_geomean_speedup"] is None else f"{row['decode_geomean_speedup']:.3f}x"
        prefill_speedup = "" if row["prefill_geomean_speedup"] is None else f"{row['prefill_geomean_speedup']:.3f}x"
        balanced = "" if row["balanced_geomean_speedup"] is None else f"{row['balanced_geomean_speedup']:.3f}x"
        match = "" if row["hash_match"] is None else f"{row['hash_match']:.0%}"
        accept = "" if row["mtp_acceptance"] is None else f"{row['mtp_acceptance']:.1%}"
        md.append(f"| {row['experiment']} | {row['cells']} | {decode_speedup} | {prefill_speedup} | {balanced} | {match} | {accept} |\n")
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
    md.extend([
        "\n## Residency and capacity telemetry\n\n",
        "Host available memory includes reclaimable cache. GPU maps are keyed by PCI BDF.\n\n",
        "| Experiment | Workload | Depth | Host available min GiB | Host cached max GiB | Anon RSS GiB | GPU VRAM max GiB | GPU GTT max GiB |\n",
        "|---|---:|---:|---:|---:|---:|---|---|\n",
    ])
    for row in ranked:
        md.append(
            f"| {row['experiment']} | {row['workload']} | {row['depth_tokens_requested']} | "
            f"{fmt(row['mem_available_min_gib_median'])} | {fmt(row['host_cached_max_gib_median'])} | "
            f"{fmt(row['rss_anon_max_gib_median'])} | {row['gpu_vram_max_gib']} | {row['gpu_gtt_max_gib']} |\n"
        )
    concurrent_groups = [
        row for row in read_jsonl(run_dir / "concurrency-groups.jsonl")
        if row.get("status") == "ok"
    ]
    if concurrent_groups:
        concurrency_fields = [
            "experiment", "round", "requested_depth_tokens", "lanes", "prompt_n_total",
            "predicted_n_total", "aggregate_prefill_tok_s", "aggregate_decode_tok_s",
            "end_to_end_output_tok_s", "wall_ms",
        ]
        with (run_dir / "concurrency.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=concurrency_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(concurrent_groups)
        md.extend([
            "\n## True concurrent-request throughput\n\n",
            "Aggregate prefill and decode divide both lanes' tokens by the longest overlapping phase; end-to-end output includes prompt processing.\n\n",
            "| Experiment | Round | Depth/lane | Lanes | Prompt n total | Output n total | Aggregate prefill tok/s | Aggregate decode tok/s | End-to-end output tok/s | Wall s |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for group in concurrent_groups:
            md.append(
                f"| {group['experiment']} | {int(group.get('round', 0)) + 1} | "
                f"{group.get('requested_depth_tokens', '')} | {group.get('lanes', '')} | "
                f"{group.get('prompt_n_total', '')} | {group.get('predicted_n_total', '')} | "
                f"{fmt(group.get('aggregate_prefill_tok_s'))} | {fmt(group.get('aggregate_decode_tok_s'))} | "
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
    if not report["ready_for_model_benchmarks"]:
        raise RuntimeError(
            "ROCm audit did not prove the build safe for model benchmarks; inspect "
            f"{output} and {output.parent / 'rocm-audit-logs'}"
        )
    return output


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
    ])
    assert canonical == ["--jinja", "--cache-type-k", "f16", "--spec-draft-n-max", "4"]
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
    bw = parse_pcie_bw("100 200 256\n")
    assert bw["pcie_rx_est_bytes_s"] == 25_600 and bw["pcie_tx_est_bytes_s"] == 51_200
    expanded = expand_tree({"x": "{root}/file"}, {"root": "/tmp"})
    assert expanded["x"] == "/tmp/file"
    prompt = make_prompt("task", 100, "abcdef")
    assert prompt.endswith("task") and len(prompt) >= 400
    aggregate = aggregate_telemetry([
        {
            "pid_rss_bytes": 10,
            "process": {
                "rss_file_bytes": 20,
                "rss_anon_bytes": 30,
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
    assert aggregate["pid_read_bytes_delta"] == 40
    assert aggregate["pid_major_faults_delta"] == 3
    assert aggregate["mem_available_min_bytes"] == 90
    assert aggregate["host_cached_max_bytes"] == 60
    assert response_slot_id({"id_slot": 3}) == 3
    assert response_slot_id({}) == 0
    assert response_slot_id({}, 1) == 1
    concurrent = concurrent_metrics([
        {"timings": {"predicted_n": 100, "predicted_ms": 5000, "prompt_n": 1000, "prompt_ms": 2000}},
        {"timings": {"predicted_n": 100, "predicted_ms": 4000, "prompt_n": 1000, "prompt_ms": 2500}},
    ], 8000)
    assert concurrent["aggregate_decode_tok_s"] == 40.0
    assert concurrent["aggregate_prefill_tok_s"] == 800.0
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
        summary_root = pathlib.Path(raw_tmp)
        atomic_json(summary_root / "manifest.json", {"tier": {"cache_state": "hot"}})
        append_jsonl(summary_root / "results.jsonl", {
            "status": "ok",
            "experiment": "test",
            "workload": "code",
            "requested_depth_tokens": 4096,
            "round": 0,
            "output_sha256": "abc",
            "http_wall_ms": 1.0,
            "server_startup_seconds": 2.0,
            "degenerate": False,
            "timing": {
                "predicted_per_second": 10.0,
                "prompt_per_second": 20.0,
                "prompt_n": 4096,
                "prompt_ms": 204.8,
                "draft_acceptance": 0.75,
            },
            "telemetry": aggregate,
        })
        append_jsonl(summary_root / "concurrency-groups.jsonl", {
            "status": "ok",
            "experiment": "test",
            "round": 0,
            "requested_depth_tokens": 4096,
            "lanes": 2,
            "prompt_n_total": 8192,
            "predicted_n_total": 256,
            "aggregate_prefill_tok_s": 40.0,
            "aggregate_decode_tok_s": 20.0,
            "end_to_end_output_tok_s": 10.0,
            "wall_ms": 25600.0,
        })
        summarize(summary_root, {}, [{"name": "test", "baseline": True}])
        rendered = (summary_root / "summary.md").read_text(encoding="utf-8")
        assert "Declared cache state: `hot`" in rendered
        assert "Residency and capacity telemetry" in rendered
        assert "True concurrent-request throughput" in rendered
        assert (summary_root / "concurrency.csv").is_file()
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
