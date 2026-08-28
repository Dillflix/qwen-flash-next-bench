#!/usr/bin/env python3
"""Build and verify the conservative H1 Qwen3.8-Flash-Next quant."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VERSION = "1.0.0"
GIB = 1024**3
HERE = Path(__file__).resolve().parent

DEFAULT_LLAMA_DIR = Path("/srv/llm/src/llama-qwen4exp")
DEFAULT_MODEL_DIR = Path("/srv/llm/models/qwen-flash-next")
DEFAULT_HF_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
DEFAULT_SOURCE = DEFAULT_MODEL_DIR / "Qwen3.8-Flash-Next-FP8-to-BF16-PLE16.gguf"
DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "Qwen3.8-Flash-Next-H1-ROCmFP4-STRIX-PLE16.gguf"
DEFAULT_IMATRIX = DEFAULT_MODEL_DIR / "calibration/qwen3.8-flash-next-h1-imatrix.gguf"
DEFAULT_CALIBRATION_MODEL = DEFAULT_MODEL_DIR / "Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX-PLE16.gguf"
DEFAULT_RECIPE = HERE / "quantization/h1.tensor-types.txt"

PRESET = "Q4_0_ROCMFP4_STRIX"
MIN_FREE_GIB = {
    "convert": 500,
    "imatrix": 5,
    "quantize": 145,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=False))


def first_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def split_source_candidates(prefix: Path) -> list[Path]:
    prefix = prefix.expanduser()
    candidates: list[Path] = []
    if prefix.is_file():
        candidates.append(prefix)
    parent = prefix.parent
    if parent.is_dir():
        stem = prefix.name[:-5] if prefix.name.endswith(".gguf") else prefix.name
        pattern = re.compile(rf"^{re.escape(stem)}-\d{{5}}-of-\d{{5}}\.gguf$")
        candidates.extend(p for p in parent.iterdir() if p.is_file() and pattern.match(p.name))
    return sorted(set(candidates))


def first_source_shard(prefix: Path) -> Path:
    if prefix.is_file():
        return prefix
    candidates = split_source_candidates(prefix)
    if not candidates:
        raise FileNotFoundError(f"no GGUF or split GGUF found for {prefix}")
    split = [p for p in candidates if re.search(r"-00001-of-\d{5}\.gguf$", p.name)]
    return split[0] if split else candidates[0]


def read_recipe(path: Path) -> list[tuple[re.Pattern[str], str, str]]:
    rules: list[tuple[re.Pattern[str], str, str]] = []
    for token in path.read_text(encoding="utf-8").split():
        if "=" not in token:
            raise ValueError(f"malformed recipe token: {token!r}")
        pattern, qtype = token.rsplit("=", 1)
        rules.append((re.compile(pattern), qtype.upper(), pattern))
    return rules


def binary_path(llama_dir: Path, name: str) -> Path:
    return llama_dir / "build-vulkan/bin" / name


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_text(command: Iterable[object]) -> str:
    import shlex

    return shlex.join(str(part) for part in command)


def new_log_dir(work_dir: Path, action: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = work_dir / f"{stamp}-{action}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_logged(command: list[str], log_dir: Path, metadata: dict[str, Any]) -> None:
    log_path = log_dir / "run.log"
    manifest = {
        "version": VERSION,
        "started": utc_now(),
        "command": command,
        **metadata,
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"$ {command_text(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    manifest["stopped"] = utc_now()
    manifest["exit_status"] = status
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if status != 0:
        raise SystemExit(f"command failed with status {status}; see {log_path}")
    print(f"Log: {log_path}")


def convert_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.llama_dir / "convert_hf_to_gguf.py"),
        "--remote",
        "--outfile",
        str(args.source),
        "--outtype",
        "bf16",
        "--no-mtp",
        "--split-max-size",
        args.split_max_size,
        args.hf_repo,
    ]


def imatrix_command(args: argparse.Namespace) -> list[str]:
    return [
        str(binary_path(args.llama_dir, "llama-imatrix")),
        "-m",
        str(args.calibration_model),
        "-f",
        str(args.calibration_file),
        "-o",
        str(args.imatrix),
        "--output-format",
        "gguf",
        "--no-ppl",
        "--chunks",
        str(args.chunks),
        "--n-gpu-layers",
        "999",
        "--device",
        "Vulkan1,Vulkan0",
        "--main-gpu",
        "0",
        "--split-mode",
        "layer",
        "--tensor-split",
        "82,18",
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.ubatch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--flash-attn",
        "on",
        "--threads",
        str(args.threads),
    ]


def quantize_command(args: argparse.Namespace, *, dry_run: bool) -> list[str]:
    source = first_source_shard(args.source)
    command = [str(binary_path(args.llama_dir, "llama-quantize"))]
    if args.imatrix.is_file():
        command.extend(["--imatrix", str(args.imatrix)])
    elif not args.allow_no_imatrix:
        raise FileNotFoundError(
            f"importance matrix missing: {args.imatrix}; generate it or pass --allow-no-imatrix"
        )
    command.extend(
        [
            "--output-tensor-type",
            "q8_0",
            "--token-embedding-type",
            "bf16",
            "--tensor-type-file",
            str(args.recipe),
        ]
    )
    if dry_run:
        command.append("--dry-run")
        command.extend([str(source), PRESET, str(args.threads)])
    else:
        command.extend([str(source), str(args.output), PRESET, str(args.threads)])
    return command


def fork_checks(llama_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    converter = llama_dir / "convert_hf_to_gguf.py"
    conversion = llama_dir / "conversion/qwen4exp.py"
    quant_source = llama_dir / "src/llama-quant.cpp"
    if not converter.is_file():
        errors.append(f"converter missing: {converter}")
    if not conversion.is_file():
        errors.append(f"Qwen3.8 conversion module missing: {conversion}")
    else:
        text = conversion.read_text(encoding="utf-8", errors="replace")
        for marker in ("_place_ple_shard", "PLE_NGRAM_EMBD", "_read_ple_weight_scale"):
            if marker not in text:
                errors.append(f"converter lacks required streamed FP8 PLE marker: {marker}")
    if not quant_source.is_file():
        errors.append(f"quantizer source missing: {quant_source}")
    else:
        text = quant_source.read_text(encoding="utf-8", errors="replace")
        for marker in ("stream_out", "max_band_bytes", "Q4_0_ROCMFP4_STRIX"):
            if marker not in text:
                errors.append(f"quantizer lacks required streamed ROCmFP4 marker: {marker}")
    if git_revision(llama_dir) is None:
        warnings.append("could not read llama.cpp git revision")
    return errors, warnings


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    errors, warnings = fork_checks(args.llama_dir)
    phase = args.phase
    required_bins: list[str] = []
    if phase == "quantize":
        required_bins.append("llama-quantize")
    if phase == "imatrix":
        required_bins.append("llama-imatrix")
    for name in required_bins:
        path = binary_path(args.llama_dir, name)
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"executable missing: {path}")

    try:
        rules = read_recipe(args.recipe)
        if len(rules) != 5:
            errors.append(f"H1 recipe should contain 5 rules, found {len(rules)}")
    except (OSError, ValueError, re.error) as exc:
        errors.append(f"invalid H1 recipe: {exc}")
        rules = []

    source_files = split_source_candidates(args.source)
    if phase == "quantize" and not source_files:
        errors.append(f"source GGUF is missing: {args.source}")
    if phase == "convert" and source_files:
        errors.append(f"source output already exists; refusing to overwrite: {source_files[0]}")
    if phase == "quantize" and not args.imatrix.is_file():
        message = f"importance matrix is missing: {args.imatrix}"
        (warnings if args.allow_no_imatrix else errors).append(message)
    if phase == "quantize" and args.output.exists():
        errors.append(f"H1 output already exists; refusing to overwrite: {args.output}")
    if phase == "imatrix":
        if args.calibration_file is None or not args.calibration_file.is_file():
            errors.append("--calibration-file must name an existing representative text corpus")
        if not args.calibration_model.is_file():
            errors.append(f"calibration model missing: {args.calibration_model}")
        if args.imatrix.exists():
            errors.append(f"importance-matrix output already exists: {args.imatrix}")

    disk_target = args.source.parent if phase == "convert" else args.output.parent
    disk = shutil.disk_usage(first_existing_parent(disk_target))
    required_gib = args.min_free_gib if args.min_free_gib is not None else MIN_FREE_GIB[phase]
    free_gib = disk.free / GIB
    if free_gib < required_gib:
        errors.append(f"only {free_gib:.1f} GiB free; {phase} requires at least {required_gib:.1f} GiB")

    report = {
        "ts": utc_now(),
        "version": VERSION,
        "phase": phase,
        "llama_dir": str(args.llama_dir),
        "llama_revision": git_revision(args.llama_dir),
        "source_repo": args.hf_repo,
        "source_prefix": str(args.source),
        "source_files": [{"path": str(p), "size_bytes": p.stat().st_size} for p in source_files],
        "output": str(args.output),
        "imatrix": str(args.imatrix),
        "recipe": str(args.recipe),
        "recipe_rules": [{"pattern": p.pattern, "type": q} for p, q, _ in rules],
        "disk_free_gib": round(free_gib, 2),
        "minimum_free_gib": required_gib,
        "warnings": warnings,
        "errors": errors,
    }
    return report


def verify(args: argparse.Namespace) -> dict[str, Any]:
    gguf_path = args.llama_dir / "gguf-py"
    sys.path.insert(0, str(gguf_path))
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"cannot import gguf-py from {gguf_path}: {exc}") from exc

    if not args.output.is_file():
        raise SystemExit(f"output not found: {args.output}")
    reader = GGUFReader(args.output, mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    type_counts = Counter(t.tensor_type.name for t in reader.tensors)
    errors: list[str] = []
    warnings: list[str] = []
    rule_counts: dict[str, int] = {}

    minima = [16, 144, 144, 48, 1]
    for (pattern, expected, raw), minimum in zip(read_recipe(args.recipe), minima):
        matched = [t for t in reader.tensors if pattern.search(t.name)]
        rule_counts[raw] = len(matched)
        if len(matched) < minimum:
            errors.append(f"{raw} matched {len(matched)} tensors; expected at least {minimum}")
        wrong = [(t.name, t.tensor_type.name) for t in matched if t.tensor_type.name != expected]
        if wrong:
            errors.append(f"{raw} has wrong types: {wrong[:8]}")

    for name, expected in (("token_embd.weight", "BF16"), ("output.weight", "Q8_0")):
        tensor = tensors.get(name)
        if tensor is None:
            errors.append(f"required tensor missing: {name}")
        elif tensor.tensor_type.name != expected:
            errors.append(f"{name} is {tensor.tensor_type.name}, expected {expected}")

    wrong_scalars = [
        (t.name, t.tensor_type.name)
        for t in reader.tensors
        if (len(t.shape) == 1 and ("norm" in t.name or t.name.endswith(".bias")))
        and t.tensor_type.name != "F32"
    ]
    if wrong_scalars:
        errors.append(f"norm/bias tensors not F32: {wrong_scalars[:8]}")

    attn_types = Counter(
        t.tensor_type.name
        for t in reader.tensors
        if re.search(r"^blk\.\d+\.attn_.*\.weight$", t.name) and len(t.shape) > 1
    )
    for expected in ("Q4_0_ROCMFP4", "Q4_0_ROCMFP4_FAST"):
        if attn_types[expected] == 0:
            errors.append(f"attention recipe contains no {expected} tensors")

    if any("mtp" in name or "nextn" in name for name in tensors):
        errors.append("target GGUF unexpectedly contains MTP/NextN tensors")

    imatrix_fields = {
        key: reader.fields[key].contents()
        for key in (
            "quantize.imatrix.file",
            "quantize.imatrix.dataset",
            "quantize.imatrix.entries_count",
            "quantize.imatrix.chunks_count",
        )
        if key in reader.fields
    }
    if not imatrix_fields:
        warnings.append("GGUF contains no importance-matrix provenance")

    result = {
        "ts": utc_now(),
        "version": VERSION,
        "output": str(args.output),
        "size_gib": round(args.output.stat().st_size / GIB, 3),
        "tensor_count": len(reader.tensors),
        "type_counts": dict(sorted(type_counts.items())),
        "attention_type_counts": dict(sorted(attn_types.items())),
        "recipe_match_counts": rule_counts,
        "imatrix": imatrix_fields,
        "warnings": warnings,
        "errors": errors,
    }
    return result


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imatrix", type=Path, default=DEFAULT_IMATRIX)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_MODEL_DIR / "h1-build")
    parser.add_argument("--allow-no-imatrix", action="store_true")


def add_imatrix_inputs(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--calibration-file", type=Path, required=required)
    parser.add_argument("--calibration-model", type=Path, default=DEFAULT_CALIBRATION_MODEL)
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--ubatch-size", type=int, default=2048)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-test", help="validate the H1 recipe and command construction")

    p = sub.add_parser("preflight", help="check fork features, paths, and free space")
    add_common_paths(p)
    add_imatrix_inputs(p)
    p.add_argument("--phase", choices=sorted(MIN_FREE_GIB), default="quantize")
    p.add_argument("--min-free-gib", type=float)

    p = sub.add_parser("convert", help="stream the official FP8 release into a BF16 plus PLE-Q8 source GGUF")
    add_common_paths(p)
    p.add_argument("--split-max-size", default="32G")

    p = sub.add_parser("imatrix", help="collect an importance matrix with the current working model")
    add_common_paths(p)
    add_imatrix_inputs(p, required=True)

    p = sub.add_parser("dry-run", help="show the final H1 tensor plan and estimated size")
    add_common_paths(p)

    p = sub.add_parser("quantize", help="build the H1 GGUF")
    add_common_paths(p)

    p = sub.add_parser("verify", help="verify every H1 tensor-family invariant")
    add_common_paths(p)

    return parser


def self_test() -> None:
    rules = read_recipe(DEFAULT_RECIPE)
    cases = {
        "ple_ngram_embd.15.weight": "Q8_0",
        "blk.47.ffn_down_exps.weight": "Q4_0_ROCMFP4_FAST",
        "blk.12.ffn_gate_shexp.weight": "Q4_0_ROCMFP4",
        "blk.7.ffn_gate_inp.weight": "Q8_0",
        "blk.44.indexer.q_proj.weight": "Q8_0",
    }
    for name, expected in cases.items():
        matches = [qtype for pattern, qtype, _ in rules if pattern.search(name)]
        if matches != [expected]:
            raise AssertionError(f"{name}: expected {[expected]}, got {matches}")
    if any(pattern.search("blk.1.attn_gate.weight") for pattern, _, _ in rules):
        raise AssertionError("large linear-attention gate must remain on the STRIX preset")

    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        source = temp / "source.gguf"
        first = temp / "source-00001-of-00002.gguf"
        second = temp / "source-00002-of-00002.gguf"
        first.touch()
        second.touch()
        if first_source_shard(source) != first:
            raise AssertionError("split source discovery did not select shard 1")
        imatrix = temp / "imatrix.gguf"
        imatrix.touch()
        args = argparse.Namespace(
            llama_dir=Path("/llama"),
            source=source,
            output=temp / "h1.gguf",
            imatrix=imatrix,
            recipe=DEFAULT_RECIPE,
            threads=16,
            allow_no_imatrix=False,
        )
        command = quantize_command(args, dry_run=False)
        for forbidden in ("--pure", "--allow-requantize"):
            if forbidden in command:
                raise AssertionError(f"unsafe H1 option present: {forbidden}")
        if command[-4:] != [str(first), str(args.output), PRESET, "16"]:
            raise AssertionError(f"unexpected quantizer command tail: {command[-4:]}")
    print(f"qwen_quant.py {VERSION}: self-test passed")


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.command == "preflight":
        report = preflight(args)
        json_print(report)
        raise SystemExit(1 if report["errors"] else 0)
    if args.command == "verify":
        result = verify(args)
        json_print(result)
        raise SystemExit(1 if result["errors"] else 0)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    action_dir = new_log_dir(args.work_dir, args.command)
    metadata = {
        "llama_dir": str(args.llama_dir),
        "llama_revision": git_revision(args.llama_dir),
        "source": str(args.source),
        "output": str(args.output),
        "imatrix": str(args.imatrix),
        "recipe": str(args.recipe),
    }
    if args.command == "convert":
        if split_source_candidates(args.source):
            raise SystemExit(f"source output already exists; refusing to overwrite: {args.source}")
        args.source.parent.mkdir(parents=True, exist_ok=True)
        command = convert_command(args)
    elif args.command == "imatrix":
        if args.imatrix.exists():
            raise SystemExit(f"importance-matrix output already exists: {args.imatrix}")
        args.imatrix.parent.mkdir(parents=True, exist_ok=True)
        command = imatrix_command(args)
    elif args.command == "dry-run":
        command = quantize_command(args, dry_run=True)
    elif args.command == "quantize":
        if args.output.exists():
            raise SystemExit(f"H1 output already exists; refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = quantize_command(args, dry_run=False)
    else:
        raise AssertionError(args.command)
    run_logged(command, action_dir, metadata)


if __name__ == "__main__":
    main()
