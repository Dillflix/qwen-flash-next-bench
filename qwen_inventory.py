#!/usr/bin/env python3
"""Byte-accurate GGUF tensor inventory using only the Python standard library.

The reported storage bytes are tensor data spans in the GGUF data section.  They
include the file's alignment padding, so family totals add up exactly to the data
section size without relying on local ggml quant-type definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import tarfile
import tempfile
from collections import defaultdict
from typing import Any, BinaryIO


VERSION = "1.0.0"
GGUF_MAGIC = b"GGUF"

VALUE_UINT8 = 0
VALUE_INT8 = 1
VALUE_UINT16 = 2
VALUE_INT16 = 3
VALUE_UINT32 = 4
VALUE_INT32 = 5
VALUE_FLOAT32 = 6
VALUE_BOOL = 7
VALUE_STRING = 8
VALUE_ARRAY = 9
VALUE_UINT64 = 10
VALUE_INT64 = 11
VALUE_FLOAT64 = 12

VALUE_FORMATS = {
    VALUE_UINT8: "B",
    VALUE_INT8: "b",
    VALUE_UINT16: "H",
    VALUE_INT16: "h",
    VALUE_UINT32: "I",
    VALUE_INT32: "i",
    VALUE_FLOAT32: "f",
    VALUE_BOOL: "?",
    VALUE_UINT64: "Q",
    VALUE_INT64: "q",
    VALUE_FLOAT64: "d",
}

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "TQ1_0",
    32: "TQ2_0",
    33: "MXFP4",
    100: "Q4_0_ROCMFP4",
    101: "Q4_0_ROCMFP4_FAST",
}


class GGUFReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.version = 0

    def read_exact(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("unexpected end of GGUF file")
        return data

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read_exact(size))[0]

    def count(self) -> int:
        return int(self.unpack("I" if self.version == 1 else "Q"))

    def string(self, capture: bool = True) -> str | None:
        length = self.count()
        if length > 1 << 32:
            raise ValueError(f"implausible GGUF string length: {length}")
        if not capture:
            self.handle.seek(length, 1)
            return None
        return self.read_exact(length).decode("utf-8", errors="replace")

    def value(self, value_type: int, capture: bool = False) -> Any:
        if value_type in VALUE_FORMATS:
            value = self.unpack(VALUE_FORMATS[value_type])
            return value if capture else None
        if value_type == VALUE_STRING:
            return self.string(capture=capture)
        if value_type == VALUE_ARRAY:
            element_type = int(self.unpack("I"))
            length = self.count()
            if length > 1 << 32:
                raise ValueError(f"implausible GGUF array length: {length}")
            values = [] if capture else None
            for _ in range(length):
                item = self.value(element_type, capture=capture)
                if capture:
                    values.append(item)
            return values
        raise ValueError(f"unknown GGUF metadata value type {value_type}")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_family(name: str) -> str:
    if name == "per_layer_token_embd.weight" or re.match(r"^ple_ngram_embd\.[0-9]+\.weight$", name):
        return "ple"
    if re.search(r"\.ffn_(down|gate|up)_exps\.weight$", name):
        return "routed_experts"
    if re.search(r"\.ffn_(down|gate|up)_shexp\.weight$", name) or "ffn_gate_inp_shexp" in name:
        return "shared_experts"
    if "ffn_gate_inp" in name or "router" in name:
        return "routers"
    if ".indexer." in name:
        return "qsa_indexer"
    if name == "token_embd.weight":
        return "token_embedding"
    if name == "output.weight":
        return "lm_head"
    if ".attn" in name:
        return "attention"
    if ".ssm" in name or ".conv" in name or ".receptance" in name:
        return "linear_attention_state"
    if ".ffn_" in name:
        return "other_ffn"
    if "norm" in name or name.endswith(".bias"):
        return "norms_biases"
    return "other"


def inventory(path: pathlib.Path, include_tensors: bool = False) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        reader = GGUFReader(handle)
        if reader.read_exact(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        reader.version = int(reader.unpack("I"))
        if reader.version not in {2, 3}:
            raise ValueError(f"{path}: unsupported GGUF version {reader.version}; expected 2 or 3")
        tensor_count = reader.count()
        metadata_count = reader.count()
        captured_metadata: dict[str, Any] = {}
        capture_keys = {"general.alignment", "general.architecture", "general.name", "general.file_type"}
        for _ in range(metadata_count):
            key = reader.string(capture=True)
            assert isinstance(key, str)
            value_type = int(reader.unpack("I"))
            capture = key in capture_keys
            value = reader.value(value_type, capture=capture)
            if capture:
                captured_metadata[key] = value
        tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name = reader.string(capture=True)
            assert isinstance(name, str)
            dimensions_n = int(reader.unpack("I"))
            if dimensions_n > 16:
                raise ValueError(f"{path}: tensor {name!r} has {dimensions_n} dimensions")
            dimensions = [int(reader.unpack("Q")) for _ in range(dimensions_n)]
            type_id = int(reader.unpack("I"))
            offset = int(reader.unpack("Q"))
            tensors.append({
                "name": name,
                "dimensions": dimensions,
                "parameters": math.prod(dimensions),
                "type_id": type_id,
                "type": GGML_TYPE_NAMES.get(type_id, f"TYPE_{type_id}"),
                "offset": offset,
                "family": tensor_family(name),
            })
        alignment = int(captured_metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"{path}: invalid GGUF alignment {alignment}")
        data_offset = align_up(handle.tell(), alignment)

    ordered = sorted(tensors, key=lambda item: (int(item["offset"]), str(item["name"])))
    for index, tensor in enumerate(ordered):
        start = data_offset + int(tensor["offset"])
        end = (
            data_offset + int(ordered[index + 1]["offset"])
            if index + 1 < len(ordered)
            else file_size
        )
        if start < data_offset or end < start or end > file_size:
            raise ValueError(f"{path}: invalid data span for tensor {tensor['name']!r}")
        tensor["storage_span_bytes"] = end - start

    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "parameters": 0, "storage_span_bytes": 0, "types": defaultdict(int)},
    )
    for tensor in ordered:
        family = families[str(tensor["family"])]
        family["tensor_count"] += 1
        family["parameters"] += int(tensor["parameters"])
        family["storage_span_bytes"] += int(tensor["storage_span_bytes"])
        family["types"][str(tensor["type"])] += int(tensor["storage_span_bytes"])

    family_output: dict[str, Any] = {}
    for name, values in sorted(families.items(), key=lambda item: -item[1]["storage_span_bytes"]):
        family_output[name] = {
            **{key: value for key, value in values.items() if key != "types"},
            "storage_span_gib": values["storage_span_bytes"] / 1024**3,
            "bits_per_parameter_including_alignment": (
                8.0 * values["storage_span_bytes"] / values["parameters"]
                if values["parameters"] else None
            ),
            "types": dict(sorted(values["types"].items())),
        }

    result: dict[str, Any] = {
        "schema": 1,
        "tool_version": VERSION,
        "file": str(path.resolve()),
        "file_size_bytes": file_size,
        "gguf_version": reader.version,
        "alignment": alignment,
        "metadata_count": metadata_count,
        "tensor_count": tensor_count,
        "data_offset": data_offset,
        "data_section_bytes": file_size - data_offset,
        "metadata": captured_metadata,
        "families": family_output,
    }
    if include_tensors:
        result["tensors"] = ordered
    return result


def write_test_gguf(path: pathlib.Path) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    payload = bytearray()
    payload += GGUF_MAGIC
    payload += struct.pack("<IQQ", 3, 2, 1)
    payload += string("general.alignment")
    payload += struct.pack("<II", VALUE_UINT32, 32)
    for name, offset in (("token_embd.weight", 0), ("output.weight", 32)):
        payload += string(name)
        payload += struct.pack("<IQQIQ", 2, 4, 4, 0, offset)
    payload += bytes(align_up(len(payload), 32) - len(payload))
    payload += bytes(range(64))
    path.write_bytes(payload)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "tiny.gguf"
        write_test_gguf(path)
        result = inventory(path, include_tensors=True)
        assert result["tensor_count"] == 2
        assert result["data_section_bytes"] == 64
        assert result["families"]["token_embedding"]["storage_span_bytes"] == 32
        assert result["families"]["lm_head"]["storage_span_bytes"] == 32
        assert [item["storage_span_bytes"] for item in result["tensors"]] == [32, 32]
        report = pathlib.Path(tmp) / "inventory.json"
        report.write_text(json.dumps(result), encoding="utf-8")
        archive, checksum = package_output(report)
        assert archive.is_file() and checksum.is_file()
        with tarfile.open(archive, "r:gz") as bundle:
            assert bundle.getnames() == [report.name]
    print(f"qwen_inventory.py {VERSION}: self-test passed")


def package_output(output: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    stem = output.with_suffix("") if output.suffix else output
    archive = stem.with_name(stem.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname=output.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="inventory one or more GGUF files")
    scan.add_argument("files", nargs="+")
    scan.add_argument("--include-tensors", action="store_true")
    scan.add_argument("--output")
    scan.add_argument("--archive", action="store_true", help="package the JSON and write a SHA-256 sidecar")
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    results = [inventory(pathlib.Path(value), include_tensors=args.include_tensors) for value in args.files]
    document = {"schema": 1, "tool_version": VERSION, "files": results}
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Inventory: {output.resolve()}")
        if args.archive:
            archive, checksum = package_output(output)
            print(f"Archive: {archive.resolve()}")
            print(f"Checksum: {checksum.resolve()}")
    elif args.archive:
        parser.error("--archive requires --output")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
