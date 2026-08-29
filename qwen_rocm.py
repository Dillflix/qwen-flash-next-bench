#!/usr/bin/env python3
"""Collect reproducible ROCmFPX/HIP source and build evidence without loading a model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


VERSION = "1.0.0"
TOKENS = ("Q4_0_ROCMFP4", "Q4_0_ROCMFP4_FAST")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
SKIP_DIRS = {".git", ".cache", "__pycache__", "build", "build-hip10", "build-vulkan"}
CMAKE_KEYS = {
    "AMDGPU_TARGETS",
    "BUILD_SHARED_LIBS",
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_HIP_ARCHITECTURES",
    "CMAKE_HIP_COMPILER",
    "CMAKE_HIP_COMPILER_ROCM_ROOT",
    "CMAKE_PREFIX_PATH",
    "GGML_BUILD_TESTS",
    "GGML_CUDA",
    "GGML_HIP",
    "GGML_HIP_FORCE_MMQ",
    "GGML_HIP_GRAPHS",
    "GGML_HIP_MMQ_MFMA",
    "GGML_HIP_NO_VMM",
    "GGML_HIP_ROCWMMA_FATTN",
    "GGML_NATIVE",
    "GGML_VULKAN",
    "GPU_TARGETS",
    "LLAMA_BUILD_SERVER",
    "LLAMA_BUILD_TESTS",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(text: str) -> str:
    # Git remotes occasionally contain credentials. The archive is intended to be shareable.
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", text)
    text = re.sub(r"(?i)(token|password|passwd|secret)=([^\s]+)", r"\1=<redacted>", text)
    return text


def capture(command: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None,
            timeout: float = 60.0) -> dict[str, Any]:
    result: dict[str, Any] = {"command": command, "cwd": str(cwd) if cwd else None}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        result.update({
            "returncode": completed.returncode,
            "stdout": redact(completed.stdout),
            "stderr": redact(completed.stderr),
        })
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = redact(str(exc))
    return result


def write_capture(path: pathlib.Path, item: dict[str, Any]) -> None:
    command = " ".join(str(part) for part in item.get("command", []))
    lines = [f"$ {command}\n"]
    if item.get("cwd"):
        lines.append(f"cwd: {item['cwd']}\n")
    if "error" in item:
        lines.append(f"ERROR: {item['error']}\n")
    else:
        lines.append(f"exit: {item.get('returncode')}\n\n")
        lines.append(str(item.get("stdout", "")))
        if item.get("stderr"):
            lines.append("\n[stderr]\n")
            lines.append(str(item["stderr"]))
    path.write_text("".join(lines), encoding="utf-8")


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_cmake_cache(path: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith(("#", "//")) or "=" not in raw or ":" not in raw.split("=", 1)[0]:
            continue
        left, value = raw.split("=", 1)
        key, type_name = left.split(":", 1)
        if key in CMAKE_KEYS:
            result[key] = {"type": type_name, "value": value}
    return result


def source_files(repo: pathlib.Path):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith("build-")]
        root_path = pathlib.Path(root)
        for name in files:
            path = root_path / name
            if path.suffix.lower() in SOURCE_SUFFIXES or name == "CMakeLists.txt":
                yield path


def inspect_source(repo: pathlib.Path, build: pathlib.Path) -> dict[str, Any]:
    hit_files: dict[str, list[str]] = {token: [] for token in TOKENS}
    excerpts: list[dict[str, Any]] = []
    checked = 0
    for path in source_files(repo) if repo.is_dir() else []:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        checked += 1
        relative = path.relative_to(repo).as_posix()
        lines = text.splitlines()
        for token in TOKENS:
            if token not in text:
                continue
            hit_files[token].append(relative)
            for number, line in enumerate(lines, 1):
                if token in line:
                    excerpts.append({"path": relative, "line": number, "text": line.strip()[:500]})
                    if sum(1 for item in excerpts if item["path"] == relative and token in item["text"]) >= 8:
                        break

    runtime_prefixes = ("ggml/src/ggml-cuda/", "ggml/src/ggml-hip/")
    runtime_hits = {
        token: sorted({path for path in paths if path.startswith(runtime_prefixes)})
        for token, paths in hit_files.items()
    }
    cmake_candidates = [
        repo / "ggml" / "src" / "ggml-hip" / "CMakeLists.txt",
        repo / "ggml" / "src" / "CMakeLists.txt",
        repo / "ggml" / "CMakeLists.txt",
    ]
    cmake_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in cmake_candidates if path.is_file()
    )
    hip_cmake_rocmfp4 = "rocmfp4_hip.cu" in cmake_text
    compile_commands_path = build / "compile_commands.json"
    compiled_sources: set[str] = set()
    if compile_commands_path.is_file():
        try:
            rows = json.loads(compile_commands_path.read_text(encoding="utf-8", errors="replace"))
            for row in rows:
                file_name = row.get("file")
                if file_name:
                    compiled_sources.add(str(pathlib.Path(file_name).resolve()).replace("\\", "/"))
        except (OSError, ValueError, TypeError):
            pass
    build_ninja_path = build / "build.ninja"
    build_ninja = build_ninja_path.read_text(encoding="utf-8", errors="ignore") if build_ninja_path.is_file() else ""
    direct_sources = [
        repo / "ggml" / "rocmfp4" / "rocmfp4_hip.cu",
        repo / "ggml" / "src" / "ggml-cuda" / "mmq.cu",
        repo / "ggml" / "src" / "ggml-cuda" / "mmvq.cu",
        repo / "ggml" / "src" / "ggml-cuda" / "mmid.cu",
        repo / "ggml" / "src" / "ggml-cuda" / "convert.cu",
        repo / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu",
    ]
    build_graph = []
    for path in direct_sources:
        relative = path.relative_to(repo).as_posix()
        absolute = str(path.resolve()).replace("\\", "/")
        build_graph.append({
            "path": relative,
            "exists": path.is_file(),
            "in_compile_commands": absolute in compiled_sources,
            "mentioned_in_build_ninja": relative in build_ninja or path.name in build_ninja,
        })
    return {
        "repo": str(repo),
        "files_checked": checked,
        "token_hits": {key: sorted(set(value)) for key, value in hit_files.items()},
        "runtime_token_hits": runtime_hits,
        "excerpts": excerpts,
        "hip_cmake_references_rocmfp4_hip_cu": hip_cmake_rocmfp4,
        "compile_commands_exists": compile_commands_path.is_file(),
        "build_ninja_exists": build_ninja_path.is_file(),
        "build_graph": build_graph,
        "source_dispatch_ready": all(runtime_hits[token] for token in TOKENS) and hip_cmake_rocmfp4,
    }


def binary_evidence(path: pathlib.Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
        "ascii_markers": {},
        "gfx_targets": [],
    }
    if not path.is_file():
        return result
    try:
        data = path.read_bytes()
    except OSError as exc:
        result["error"] = str(exc)
        return result
    result["ascii_markers"] = {
        token: token.encode("ascii") in data or token.lower().encode("ascii") in data
        for token in TOKENS
    }
    result["gfx_targets"] = sorted({item.decode("ascii") for item in re.findall(rb"gfx[0-9a-f]{3,5}[a-z]*", data)})
    return result


def make_env(rocm: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm)
    env["HIP_PATH"] = str(rocm)
    library_paths = [rocm / "lib", rocm / "lib64", rocm / "lib" / "rocm_sysdeps" / "lib"]
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        library_paths.append(pathlib.Path(existing))
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths)
    return env


def collect(args: argparse.Namespace) -> pathlib.Path:
    repo = pathlib.Path(args.llama_dir).resolve()
    build = pathlib.Path(args.build_dir).resolve()
    rocm = pathlib.Path(args.rocm).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    output = output_root / f"rocm-forensics-{run_id()}"
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "commands"
    logs.mkdir()
    env = make_env(rocm)

    server = build / "bin" / "llama-server"
    hip_library_candidates = (build / "bin" / "libggml-hip.so", build / "lib" / "libggml-hip.so")
    hip_library = next((path for path in hip_library_candidates if path.is_file()), hip_library_candidates[0])
    test_ops = build / "bin" / "test-backend-ops"

    commands: list[tuple[str, list[str], pathlib.Path | None, float]] = [
        ("uname", ["uname", "-a"], None, 30),
        ("os-release", ["sh", "-c", "cat /etc/os-release"], None, 30),
        ("id", ["id"], None, 30),
        ("git-head", ["git", "rev-parse", "HEAD"], repo, 30),
        ("git-branch", ["git", "branch", "--show-current"], repo, 30),
        ("git-status", ["git", "status", "--short"], repo, 30),
        ("git-remotes", ["git", "remote", "-v"], repo, 30),
        ("git-log", ["git", "log", "-1", "--decorate=full", "--stat"], repo, 30),
        ("git-submodules", ["git", "submodule", "status"], repo, 60),
        ("cmake-version", ["cmake", "--version"], None, 30),
        ("ninja-version", ["ninja", "--version"], None, 30),
        ("hipcc-version", [str(rocm / "bin" / "hipcc"), "--version"], None, 60),
        ("hipconfig", [str(rocm / "bin" / "hipconfig"), "--full"], None, 60),
        ("rocminfo", [str(rocm / "bin" / "rocminfo")], None, 90),
        ("amd-smi-version", ["amd-smi", "version"], None, 60),
        ("amd-smi-list", ["amd-smi", "list"], None, 60),
        ("amd-smi-topology", ["amd-smi", "topology"], None, 60),
        ("server-version", [str(server), "--version"], None, 60),
        ("server-devices", [str(server), "--list-devices"], None, 60),
        ("server-ldd", ["ldd", str(server)], None, 60),
        ("hip-library-ldd", ["ldd", str(hip_library)], None, 60),
        ("hip-library-dynamic", ["readelf", "-d", str(hip_library)], None, 60),
        ("hip-library-sections", ["readelf", "-S", str(hip_library)], None, 60),
        ("test-backend-ops-help", [str(test_ops), "--help"], None, 60),
        ("cmake-cache-list", ["cmake", "-LAH", "-N", str(build)], None, 60),
    ]
    captures: dict[str, Any] = {}
    for name, command, cwd, timeout in commands:
        item = capture(command, cwd=cwd, env=env, timeout=timeout)
        captures[name] = item
        write_capture(logs / f"{name}.txt", item)

    source = inspect_source(repo, build)
    cmake_cache = parse_cmake_cache(build / "CMakeCache.txt")
    binaries = {
        "server": binary_evidence(server),
        "hip_library": binary_evidence(hip_library),
        "test_backend_ops": binary_evidence(test_ops),
    }
    requested_targets = set()
    for key in ("CMAKE_HIP_ARCHITECTURES", "GPU_TARGETS", "AMDGPU_TARGETS"):
        value = str(cmake_cache.get(key, {}).get("value", ""))
        requested_targets.update(part for part in re.split(r"[;,\s]+", value) if part.startswith("gfx"))
    emitted_targets = set(binaries["hip_library"].get("gfx_targets", []))
    linked = captures.get("server-ldd", {})
    linked_text = str(linked.get("stdout", "")) + str(linked.get("stderr", ""))
    hip_enabled = str(cmake_cache.get("GGML_HIP", {}).get("value", "")).upper() in {"ON", "1", "TRUE", "YES"}
    gates = {
        "source_dispatch_ready": source["source_dispatch_ready"],
        "hip_enabled_in_cache": hip_enabled,
        "requested_gfx1100_and_gfx1151": {"gfx1100", "gfx1151"}.issubset(requested_targets),
        "emitted_gfx1100_and_gfx1151": {"gfx1100", "gfx1151"}.issubset(emitted_targets),
        "linked_to_requested_rocm": str(rocm) in linked_text,
        "test_backend_ops_exists": test_ops.is_file(),
    }
    reasons = []
    labels = {
        "source_dispatch_ready": "custom ROCmFP4 dispatch is absent from the HIP/CUDA runtime source or HIP CMake list",
        "hip_enabled_in_cache": "GGML_HIP is not enabled in CMakeCache.txt",
        "requested_gfx1100_and_gfx1151": "the build was not configured for both gfx1100 and gfx1151",
        "emitted_gfx1100_and_gfx1151": "libggml-hip.so does not show code objects for both gfx1100 and gfx1151",
        "linked_to_requested_rocm": f"llama-server is not visibly linked to {rocm}",
        "test_backend_ops_exists": "test-backend-ops is missing",
    }
    for key, passed in gates.items():
        if not passed:
            reasons.append(labels[key])
    manifest = {
        "schema": 1,
        "created_at": utc_now(),
        "collector_version": VERSION,
        "paths": {"llama_dir": str(repo), "build_dir": str(build), "rocm": str(rocm)},
        "source": source,
        "cmake_cache": cmake_cache,
        "binaries": binaries,
        "requested_targets": sorted(requested_targets),
        "emitted_targets": sorted(emitted_targets),
        "gates": gates,
        "ready_for_backend_ops": all(gates.values()),
        "reasons": reasons,
        "command_logs": {name: f"commands/{name}.txt" for name in captures},
    }
    atomic_json(output / "manifest.json", manifest)
    summary = [
        "# ROCm/HIP forensic summary",
        "",
        f"Created: `{manifest['created_at']}`",
        f"Source: `{repo}`",
        f"Build: `{build}`",
        f"ROCm: `{rocm}`",
        "",
        "## Gates",
        "",
    ]
    for key, passed in gates.items():
        summary.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    summary.extend(["", "## Reasons", ""])
    summary.extend(f"- {reason}" for reason in reasons)
    if not reasons:
        summary.append("- Static/build evidence is complete; run the numerical backend-op gate next.")
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    archive_path = output.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(output, arcname=output.name)
    digest = sha256_file(archive_path)
    archive_path.with_name(archive_path.name + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="ascii",
    )
    print(json.dumps(manifest, indent=2))
    print(f"Archive: {archive_path}")
    print(f"Checksum: {archive_path}.sha256")
    return archive_path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw)
        repo = root / "src"
        build = root / "build"
        (repo / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
        (repo / "ggml" / "src" / "ggml-hip").mkdir(parents=True)
        (repo / "ggml" / "rocmfp4").mkdir(parents=True)
        build.mkdir()
        (repo / "ggml" / "src" / "ggml-cuda" / "mmq.cu").write_text(
            "Q4_0_ROCMFP4 Q4_0_ROCMFP4_FAST\n", encoding="utf-8",
        )
        (repo / "ggml" / "src" / "ggml-hip" / "CMakeLists.txt").write_text(
            'list(APPEND SOURCES "../../rocmfp4/rocmfp4_hip.cu")\n', encoding="utf-8",
        )
        (repo / "ggml" / "rocmfp4" / "rocmfp4_hip.cu").write_text("kernel\n", encoding="utf-8")
        evidence = inspect_source(repo, build)
        assert evidence["source_dispatch_ready"]
        cache = build / "CMakeCache.txt"
        cache.write_text(
            "GGML_HIP:BOOL=ON\nCMAKE_HIP_ARCHITECTURES:STRING=gfx1100;gfx1151\nIGNORED:BOOL=ON\n",
            encoding="utf-8",
        )
        parsed = parse_cmake_cache(cache)
        assert parsed["GGML_HIP"]["value"] == "ON"
        assert "IGNORED" not in parsed
        assert "<redacted>@" in redact("https://token@example.invalid/repo")
    print(f"qwen_rocm.py {VERSION}: self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=VERSION)
    sub = result.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="collect and archive source/build/runtime evidence")
    collect_parser.add_argument("--llama-dir", default=os.environ.get("QWEN_LLAMA_DIR", "/srv/llm/src/llama-qwen4exp"))
    collect_parser.add_argument("--build-dir", default=os.environ.get("QWEN_HIP_BUILD", "/srv/llm/src/llama-qwen4exp/build-hip10"))
    collect_parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm-10.0.0"))
    collect_parser.add_argument("--output-root", default=str(pathlib.Path(__file__).with_name("preflight")))
    sub.add_parser("self-test", help="run collector unit checks")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "collect":
            collect(args)
        else:
            self_test()
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
