#!/usr/bin/env python3
"""Plan/build a joined-PLE Q5_K gate/up + IQ4_NL down screening quant.

Uses the standard-library inventory reader. Does not download weights, convert
HF models, change services, or enable requantization. Run requires a new output
directory; partial artifacts remain there on failure.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from qwen_inventory import inventory
from qwen_quant import command_text, read_recipe, run_logged

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "quantization/strix-q5k-iq4nl.tensor-types.txt"
PIN = "5f851647fe5ed795dfd6c0a3fba543114879e874"


def source_files(first: Path) -> list[Path]:
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
    if not match:
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", first.name):
            raise ValueError("Pass the first shard, not a later shard")
        return [first]
    total = int(match[2])
    if not 1 <= total <= 10000:
        raise ValueError("Invalid shard count")
    return [first.with_name(f"{match[1]}-{n:05d}-of-{total:05d}.gguf")
            for n in range(1, total + 1)]


def checked_tensors(files: list[Path], source: bool) -> dict:
    tensors = {}
    for path in files:
        report = inventory(path, include_tensors=True)
        if report["metadata"].get("general.architecture") != "qwen4exp":
            raise ValueError(f"Not a qwen4exp GGUF: {path}")
        for tensor in report["tensors"]:
            name = tensor["name"]
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            if source and tensor["type"] not in {"F32", "F16", "BF16"}:
                raise ValueError(f"Already quantized source: {name}={tensor['type']}")
            tensors[name] = tensor
    for layer in range(48):
        for projection in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            shape = [640, 2560, 512] if projection == "down" else [2560, 640, 512]
            if tensors.get(name, {}).get("dimensions") != shape:
                raise ValueError(f"Missing or unexpected expert shape: {name}")
    experts = [n for n in tensors if re.search(r"\.ffn_(gate|up|down)_exps\.weight$", n)]
    if len(experts) != 144:
        raise ValueError("Expected exactly 144 routed expert tensors; exclude MTP from target")
    ple = tensors.get("per_layer_token_embd.weight", {})
    if ple.get("dimensions") != [160, 320001536]:
        raise ValueError("Expected the joined PLE table [160, 320001536], not PLE16")
    return tensors


def verify(before: dict, after: dict) -> None:
    if before.keys() != after.keys():
        raise ValueError("Output tensor names differ from source")
    rules = read_recipe(RECIPE)
    for name, tensor in after.items():
        if tensor["dimensions"] != before[name]["dimensions"]:
            raise ValueError(f"Output shape changed: {name}")
        for pattern, qtype, _ in rules:
            if pattern.search(name) and tensor["type"] != qtype:
                raise ValueError(f"Wrong output type: {name}={tensor['type']}, expected {qtype}")
        if name in {"output.weight", "token_embd.weight"} and tensor["type"] != "Q8_0":
            raise ValueError(f"Expected Q8_0: {name}")
        protected = (len(tensor["dimensions"]) == 1 or "_norm.weight" in name or
                     "ffn_gate_inp.weight" in name or "ssm_conv1d" in name or
                     re.search(r"\.indexer\.(q|k)_proj\.weight$", name))
        if protected and tensor["type"] != before[name]["type"]:
            raise ValueError(f"Protected tensor precision changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="F16/BF16 GGUF or its first shard")
    parser.add_argument("--llama-dir", type=Path,
                        default=Path("/srv/llm/src/strix-llama-trial-5f851647"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--run", action="store_true", help="Without this flag, only inspect headers and print plan")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    before = checked_tensors(source_files(source), source=True)
    revision = subprocess.check_output(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected inspected Strix revision {PIN}, found {revision}")
    binary = args.llama_dir / "build-vulkan/bin/llama-quantize"
    if not binary.is_file():
        raise ValueError(f"Build quantizer first: cmake --build {args.llama_dir}/build-vulkan --target llama-quantize -j 4")
    if args.imatrix and not args.imatrix.is_file():
        raise ValueError(f"Missing imatrix: {args.imatrix}")
    options = [str(binary), "--max-buffer-size", "1024", "--tensor-type-file", str(RECIPE),
               "--output-tensor-type", "Q8_0", "--token-embedding-type", "Q8_0"]
    if args.imatrix:
        options += ["--imatrix", str(args.imatrix.resolve())]
    output = output_dir / "Qwen3.8-Flash-Next-Q5_K-IQ4_NL-PLE-Q8_0.gguf"
    command = options + [str(source), str(output), "Q8_0", str(args.threads)]
    print(command_text(command), flush=True)
    print("Calibration: " + ("supplied imatrix" if args.imatrix else "UNCALIBRATED screening candidate"))
    print("Expected about 77.8 GiB non-PLE weights plus 50.7 GiB SSD PLE; not total runtime RAM.")
    if not args.run:
        print("Header checks passed. No weights written. Add --run to quantize.")
        return
    parent = output_dir.parent
    if not parent.is_dir():
        raise ValueError(f"Create the output parent directory first: {parent}")
    if shutil.disk_usage(parent).free < 145 * 1024**3:
        raise ValueError("At least 145 GiB free output disk space required")
    output_dir.mkdir(exist_ok=False)
    metadata = {"tool": "qwen_strix_quant.py", "version": "1.0.0", "strix_revision": revision,
                "source_files": [str(p) for p in source_files(source)],
                "recipe": RECIPE.read_text(), "calibrated": bool(args.imatrix)}
    dry_dir = output_dir / "dry-run"
    dry_dir.mkdir()
    run_logged(options + ["--dry-run", str(source), "Q8_0", str(args.threads)], dry_dir, metadata)
    run_logged(command, output_dir, metadata)
    after = checked_tensors([output], source=False)
    verify(before, after)
    report = inventory(output, include_tensors=True)
    (output_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recipe verification PASS: {output}")
    print("This checks layout/types only; model correctness, quality, and GPU performance remain untested.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
