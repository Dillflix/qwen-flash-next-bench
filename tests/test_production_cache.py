"""Exercise the production cache gate without loading a model or exposing keys."""
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and pathlib.Path(BASH).is_file(), "bash is required")
class ProductionCache(unittest.TestCase):
    def launch(self, cache="8192", parallel="1", patched=True):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            (root / "bin").mkdir()
            server = root / "bin/llama-server"
            server.write_text("#!/bin/sh\n# prompt cache checkpoint candidate: source=ram\n")
            server.chmod(0o755)
            (root / "bin/libllama.so").write_text(
                "Qwen4Exp PLE snapshot version mismatch" if patched else "old build")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("LLAMA_", "QWEN_")) and k not in ("API_KEY", "MODEL", "MTP", "MMPROJ")}
            env.update(PATH="/usr/bin:/bin", LLAMA_SERVER=server.as_posix(),
                       MODEL=server.as_posix(), MTP=server.as_posix(), MMPROJ=server.as_posix(),
                       ROCM_PATH="/usr", HIP_PATH="/usr", LLAMA_MTP_MODE="off",
                       LLAMA_HOST="127.0.0.1", LLAMA_CACHE_RAM_MIB=cache,
                       LLAMA_PARALLEL=parallel, LLAMA_BACKING_CACHE_DIAGNOSTIC="0")
            return subprocess.run([BASH, "deployment/run-production.sh", "--check"],
                                  cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)

    def test_production_opt_in_needs_no_diagnostic_flag(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Backing prompt cache limit: 8192 MiB", result.stdout)

    def test_unpatched_runtime_is_rejected(self):
        result = self.launch(patched=False)
        self.assertEqual(result.returncode, 66)
        self.assertIn("checkpoint/PLE snapshot fix", result.stderr)

    def test_zero_remains_available_on_old_runtime(self):
        result = self.launch(cache="0", patched=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multi_slot_and_excessive_budget_rejected(self):
        self.assertEqual(self.launch(parallel="2").returncode, 64)
        self.assertEqual(self.launch(cache="8193").returncode, 64)

    def test_dropin_only_overrides_cache_settings(self):
        lines = (ROOT / "deployment/backing-cache.env").read_text().splitlines()
        settings = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
        self.assertEqual(settings, {"LLAMA_CACHE_RAM_MIB": "8192", "LLAMA_BACKING_CACHE_DIAGNOSTIC": "0"})
        self.assertIn("EnvironmentFile=/etc/qwen-flash-next-cache.env",
                      (ROOT / "deployment/20-backing-cache.conf").read_text())
