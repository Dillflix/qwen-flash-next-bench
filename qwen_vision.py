#!/usr/bin/env python3
"""Prepare deterministic vision fixtures and verified Qwen3.8 projector files."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import pathlib
import struct
import sys
import urllib.request
import zlib
from typing import Callable


VERSION = "1.0.0"
DEFAULT_MODEL_DIR = pathlib.Path("/srv/llm/models/qwen-flash-next")
PROJECTORS = {
    "q8": {
        "filename": "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
        "url": "https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
        "sha256": "b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad",
    },
    "bf16": {
        "filename": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-BF16.gguf",
        "sha256": "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355",
    },
}


FONT = {
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


class Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(color * (width * height))

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset:offset + 3] = bytes(color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height, y1)))
        row = bytes(color) * max(0, x1 - x0)
        for y in range(y0, y1):
            offset = (y * self.width + x0) * 3
            self.pixels[offset:offset + len(row)] = row

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.rect(x0 - width // 2, y0 - width // 2, x0 + (width + 1) // 2, y0 + (width + 1) // 2, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * err
            if twice >= dy:
                err += dy
                x0 += sx
            if twice <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        radius2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            span = int(max(0, radius2 - (y - cy) ** 2) ** 0.5)
            self.rect(cx - span, y, cx + span + 1, y + 1, color)

    def triangle(self, top: tuple[int, int], left: tuple[int, int], right: tuple[int, int], color: tuple[int, int, int]) -> None:
        tx, ty = top
        lx, ly = left
        rx, ry = right
        for y in range(ty, max(ly, ry) + 1):
            fraction = (y - ty) / max(1, max(ly, ry) - ty)
            xa = round(tx + (lx - tx) * fraction)
            xb = round(tx + (rx - tx) * fraction)
            self.rect(min(xa, xb), y, max(xa, xb) + 1, y + 1, color)

    def text(self, x: int, y: int, value: str, scale: int, color: tuple[int, int, int]) -> None:
        cursor = x
        for char in value.upper():
            glyph = FONT.get(char, FONT[" "])
            for row_index, row in enumerate(glyph):
                for column, enabled in enumerate(row):
                    if enabled == "1":
                        self.rect(
                            cursor + column * scale,
                            y + row_index * scale,
                            cursor + (column + 1) * scale,
                            y + (row_index + 1) * scale,
                            color,
                        )
            cursor += 6 * scale

    def png(self) -> bytes:
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            start = y * stride
            rows.append(b"\x00" + bytes(self.pixels[start:start + stride]))
        raw = b"".join(rows)

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def shapes_fixture() -> bytes:
    canvas = Canvas(1024, 768, (245, 247, 250))
    canvas.text(260, 42, "VISUAL TEST", 8, (25, 30, 40))
    canvas.circle(220, 295, 110, (220, 45, 45))
    canvas.rect(405, 185, 625, 405, (40, 95, 220))
    canvas.triangle((810, 175), (675, 420), (945, 420), (40, 165, 75))
    canvas.text(260, 575, "UNSLOTH 42", 12, (20, 20, 20))
    return canvas.png()


def invoice_fixture() -> bytes:
    canvas = Canvas(1024, 768, (232, 235, 240))
    canvas.rect(85, 45, 939, 723, (255, 255, 255))
    canvas.text(140, 95, "INVOICE 4827", 8, (20, 30, 55))
    canvas.text(140, 185, "NORTH STAR LABS", 6, (35, 45, 65))
    canvas.text(140, 250, "DATE 2026-08-28", 5, (35, 45, 65))
    canvas.line(140, 315, 880, 315, (70, 80, 100), 3)
    canvas.text(150, 350, "CAMERA   2   149.50", 5, (30, 35, 45))
    canvas.text(150, 415, "CABLE    4    12.25", 5, (30, 35, 45))
    canvas.line(140, 485, 880, 485, (70, 80, 100), 3)
    canvas.text(500, 535, "TOTAL 348.00", 7, (10, 25, 55))
    canvas.rect(140, 635, 455, 655, (35, 95, 185))
    return canvas.png()


def chart_fixture() -> bytes:
    canvas = Canvas(1024, 768, (250, 250, 252))
    canvas.text(210, 45, "QUARTERLY SALES", 8, (25, 35, 55))
    canvas.line(120, 650, 930, 650, (35, 40, 50), 4)
    canvas.line(120, 150, 120, 650, (35, 40, 50), 4)
    data = [("Q1", 42, (65, 125, 210)), ("Q2", 68, (55, 165, 105)), ("Q3", 55, (225, 155, 50)), ("Q4", 91, (190, 70, 85))]
    for index, (label, value, color) in enumerate(data):
        x0 = 180 + index * 190
        height = value * 5
        canvas.rect(x0, 650 - height, x0 + 120, 650, color)
        canvas.text(x0 + 30, 670, label, 5, (25, 30, 40))
        canvas.text(x0 + 27, 610 - height, str(value), 6, (25, 30, 40))
    return canvas.png()


FIXTURES: dict[str, Callable[[], bytes]] = {
    "shapes.png": shapes_fixture,
    "invoice.png": invoice_fixture,
    "chart.png": chart_fixture,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_fixtures(model_dir: pathlib.Path) -> list[dict[str, object]]:
    target = model_dir / "vision-assets"
    target.mkdir(parents=True, exist_ok=True)
    report = []
    for filename, builder in FIXTURES.items():
        data = builder()
        path = target / filename
        expected = sha256_bytes(data)
        if sha256_file(path) != expected:
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        report.append({"path": str(path), "size_bytes": len(data), "sha256": expected})
    return report


def download_projector(model_dir: pathlib.Path, name: str, force: bool = False) -> dict[str, object]:
    spec = PROJECTORS[name]
    path = model_dir / str(spec["filename"])
    expected = str(spec["sha256"])
    if not force and sha256_file(path) == expected:
        return {"name": name, "path": str(path), "status": "present", "sha256": expected}
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(str(spec["url"]), headers={"User-Agent": "qwen-flash-next-bench/vision"})
    print(f"Downloading {name} projector to {path}...", flush=True)
    digest = hashlib.sha256()
    total = 0
    next_report = 64 * 1024**2
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                block = response.read(4 * 1024**2)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                total += len(block)
                if total >= next_report:
                    print(f"  {total / 1024**2:.0f} MiB", flush=True)
                    next_report += 64 * 1024**2
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} projector SHA-256 mismatch: expected {expected}, got {actual}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"name": name, "path": str(path), "status": "downloaded", "size_bytes": total, "sha256": expected}


def selected_projectors(value: str) -> list[str]:
    if value == "both":
        return ["q8", "bf16"]
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in PROJECTORS]
    if invalid or not names:
        raise ValueError(f"unknown projector selection: {value}")
    return names


def verify(model_dir: pathlib.Path, projector_names: list[str]) -> dict[str, object]:
    fixtures = []
    for filename, builder in FIXTURES.items():
        expected = sha256_bytes(builder())
        path = model_dir / "vision-assets" / filename
        actual = sha256_file(path)
        fixtures.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "ok": actual == expected})
    projectors = []
    for name in projector_names:
        spec = PROJECTORS[name]
        path = model_dir / str(spec["filename"])
        actual = sha256_file(path)
        projectors.append({"name": name, "path": str(path), "expected_sha256": spec["sha256"], "actual_sha256": actual, "ok": actual == spec["sha256"]})
    return {
        "schema": 1,
        "version": VERSION,
        "model_dir": str(model_dir),
        "fixtures": fixtures,
        "projectors": projectors,
        "ok": all(item["ok"] for item in fixtures + projectors),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fixtures", help="generate deterministic local PNG fixtures")
    projectors = sub.add_parser("projectors", help="download and verify projector GGUFs")
    projectors.add_argument("--projectors", default="both", help="q8, bf16, or both")
    projectors.add_argument("--force", action="store_true")
    prepare = sub.add_parser("prepare", help="generate fixtures and download projector GGUFs")
    prepare.add_argument("--projectors", default="both", help="q8, bf16, or both")
    prepare.add_argument("--force", action="store_true")
    check = sub.add_parser("verify", help="verify fixtures and projector checksums")
    check.add_argument("--projectors", default="both", help="q8, bf16, or both")
    sub.add_parser("self-test", help="validate deterministic fixture generation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_dir = args.model_dir.resolve()
    try:
        if args.command == "self-test":
            first = {name: sha256_bytes(builder()) for name, builder in FIXTURES.items()}
            second = {name: sha256_bytes(builder()) for name, builder in FIXTURES.items()}
            assert first == second and len(first) == 3
            for builder in FIXTURES.values():
                data = builder()
                assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 1024
            print(f"qwen_vision.py {VERSION}: self-test passed")
            return 0
        names = selected_projectors(getattr(args, "projectors", "both"))
        if args.command in {"fixtures", "prepare"}:
            print(json.dumps({"fixtures": create_fixtures(model_dir)}, indent=2))
        if args.command in {"projectors", "prepare"}:
            report = [download_projector(model_dir, name, args.force) for name in names]
            print(json.dumps({"projectors": report}, indent=2))
        if args.command in {"verify", "prepare"}:
            report = verify(model_dir, names)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
