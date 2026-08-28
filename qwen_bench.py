#!/usr/bin/env python3
"""Config-driven llama.cpp topology and speculative-decoding benchmark harness.

The runner uses only Python's standard library.  It is designed for very large
models where process isolation, cold server starts, exact command capture, and
recoverable partial results matter more than shaving a few seconds off a run.
"""

from __future__ import annotations

import argparse
import copy
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
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Iterable


VERSION = "1.5.0"
SUCCESS_STATES = {"ok"}
SINGLE_VALUE_SERVER_OPTIONS = {
    "-m",
    "-md",
    "-ot",
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
    return [command[0], *canonicalize_server_args(command[1:])]


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


def process_rss_bytes(pid: int) -> int | None:
    try:
        text = pathlib.Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


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
        sample: dict[str, Any] = {
            "ts": utc_now(),
            "monotonic_s": time.monotonic(),
            "pid_rss_bytes": process_rss_bytes(self.pid),
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
    available: list[float] = []
    link_speeds: list[float] = []
    link_widths: list[float] = []
    for sample in samples:
        if isinstance(sample.get("pid_rss_bytes"), (int, float)):
            rss.append(float(sample["pid_rss_bytes"]))
        host_available = sample.get("host", {}).get("MemAvailable")
        if isinstance(host_available, (int, float)):
            available.append(float(host_available))
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
    if available:
        result["mem_available_min_bytes"] = int(min(available))
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
    report = {
        "ts": utc_now(),
        "version": VERSION,
        "host": host,
        "port": port,
        "tier": tier,
        "experiments": [item["name"] for item in experiments],
        "files": files,
        "drm_cards": discover_drm_cards(),
        "active_llama_processes": active,
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
                pending = [
                    (workload, depth)
                    for workload in workload_names
                    for depth in depths
                    if (round_index, experiment["name"], workload, depth) not in completed
                ]
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
                    warm_prompt = make_prompt(workloads_all[workload_names[0]], 0, corpus)
                    for _ in range(warmups):
                        completion_request(
                            base_url + "/completion",
                            warm_prompt,
                            min(32, n_predict),
                            request_timeout_s,
                            extra_request,
                        )
                    for workload, depth in pending:
                        if stop_event.is_set():
                            break
                        prompt = make_prompt(workloads_all[workload], depth, corpus)
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
        if row["experiment"] == baseline_name:
            key = (row["workload"], int(row["requested_depth_tokens"]), int(row["round"]))
            baseline_hashes.setdefault(key, row["output_sha256"])
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
            if key in baseline_hashes:
                matches.append(row["output_sha256"] == baseline_hashes[key])
        gpu_busy: dict[str, list[float]] = defaultdict(list)
        gpu_vram: dict[str, list[float]] = defaultdict(list)
        gpu_power: dict[str, list[float]] = defaultdict(list)
        gpu_temp: dict[str, list[float]] = defaultdict(list)
        gpu_pcie_rx: dict[str, list[float]] = defaultdict(list)
        gpu_pcie_tx: dict[str, list[float]] = defaultdict(list)
        link_width: list[float] = []
        link_speed: list[float] = []
        for row in rows:
            telemetry = row.get("telemetry", {})
            for bdf, gpu in telemetry.get("gpus", {}).items():
                if isinstance(gpu.get("busy_mean_percent"), (int, float)):
                    gpu_busy[bdf].append(float(gpu["busy_mean_percent"]))
                if isinstance(gpu.get("vram_used_max_bytes"), (int, float)):
                    gpu_vram[bdf].append(float(gpu["vram_used_max_bytes"]))
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
            "gpu_busy_mean": json.dumps({bdf: round(statistics.fmean(vals), 2) for bdf, vals in gpu_busy.items()}, sort_keys=True),
            "gpu_vram_max_gib": json.dumps({bdf: round(max(vals) / 1024**3, 2) for bdf, vals in gpu_vram.items()}, sort_keys=True),
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
        "| Experiment | Workload | Requested depth | Prompt n | Samples | Decode tok/s | Prefill tok/s | Prefill ms | MTP accept | Hash match | PCIe |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
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
            f"{fmt(row['decode_tok_s_median'])} | {fmt(row['prompt_tok_s_median'])} | {fmt(row['prefill_ms_median'])} | {accept} | {match} | {pcie} |\n"
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
            "host": {"MemAvailable": 100},
            "gpus": {"0000:01:00.0": {"gpu_busy_percent": 50, "vram_used_bytes": 20, "gtt_used_bytes": 2}},
            "pcie": {"speed_gt_s": 16.0, "width_lanes": 4},
        }
    ])
    assert aggregate["gpus"]["0000:01:00.0"]["vram_used_max_bytes"] == 20
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
