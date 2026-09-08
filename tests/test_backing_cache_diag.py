"""Ensure cold fallback and mismatched replies cannot qualify backing caching."""
import pathlib
import unittest
import tempfile
from unittest.mock import patch

import qwen_backing_cache_diag as diag


class BackingCacheGates(unittest.TestCase):
    def rows(self):
        return [{"case_id": f"{arm}-{case}", "arm": arm, "case": case,
                 "contract_pass": True, "draft_n": 12, "token_count": 20,
                 "completion_tokens": 20, "token_evidence": {"identity_complete": True},
                 "cache_n": 2900 if arm == "ram" else 0, "usage_prompt_n": 3000}
                for arm in ("off", "ram")
                for case in ("a-establish", "b-displace", "a-resume", "b-resume")]

    def classify(self, rows, match=True):
        with patch.object(diag.prefix, "exact_output_comparison", return_value={"passed": match}):
            return diag.report(pathlib.Path("unused"), rows)

    def test_zero_cache_cannot_pass_on_output_agreement(self):
        rows = self.rows()
        for row in rows:
            row["cache_n"] = 0
        self.assertFalse(self.classify(rows)["passed"])

    def test_missing_trace_cannot_pass_on_text_agreement(self):
        rows = self.rows()
        rows[-1]["token_count"] = 0
        self.assertFalse(self.classify(rows)["passed"])

    def test_mtp_bypass_cannot_qualify(self):
        rows = self.rows()
        for row in rows:
            if row["arm"] == "ram":
                row["draft_n"] = 0
        self.assertFalse(self.classify(rows)["passed"])

    def test_contamination_or_divergence_fails(self):
        self.assertFalse(self.classify(self.rows(), match=False)["passed"])
        rows = self.rows()
        rows[-1]["contract_pass"] = False
        self.assertFalse(self.classify(rows)["passed"])

    def test_complete_pass(self):
        self.assertTrue(self.classify(self.rows())["passed"])

    def test_small_reuse_needs_checkpoint_evidence(self):
        rows = self.rows()
        for row in rows:
            if row["arm"] == "ram":
                row["cache_n"] = 1225
        self.assertFalse(self.classify(rows)["passed"])
        with patch.object(diag, "checkpoint_evidence", return_value={"verified": True}):
            self.assertTrue(self.classify(rows)["passed"])

    def test_checkpoint_log_and_reused_count_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "request-logs").mkdir()
            path = root / "request-logs/ram-a-resume.log"
            candidate = "prompt cache checkpoint candidate: source=ram lcp=2740 reusable=1225 cached_tokens=2900\n"
            rollback = "prompt cache checkpoint rollback: lcp=2740 cached_tokens=2900 request_tokens=2765 checkpoint=[1224,1224] n_past=1225 spec_state=400\n"
            row = {"case_id": "ram-a-resume", "cache_n": 1225}
            path.write_text(candidate + rollback)
            self.assertTrue(diag.checkpoint_evidence(root, row)["verified"])
            row["cache_n"] = 0
            self.assertFalse(diag.checkpoint_evidence(root, row)["verified"])
            row["cache_n"] = 1225
            path.write_text(candidate)
            self.assertFalse(diag.checkpoint_evidence(root, row)["verified"])
            path.write_text(rollback)
            self.assertFalse(diag.checkpoint_evidence(root, row)["verified"])

    def test_runtime_patch_markers_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            server, library = root / "llama-server", root / "libllama.so"
            server.write_bytes(b"prompt cache checkpoint candidate: source=ram")
            library.write_bytes(b"old library")
            with patch.dict(diag.os.environ, {"LLAMA_SERVER": str(server)}):
                with self.assertRaisesRegex(RuntimeError, "lacks the backing-cache patch"):
                    diag.check_runtime_patch()
                library.write_bytes(b"Qwen4Exp PLE snapshot version mismatch")
                result = diag.check_runtime_patch()
                self.assertEqual(len(result["server"]["sha256"]), 64)

    def test_fixture_has_shared_prefix_and_separate_families(self):
        a0, _ = diag.payload("A", 0, 96)
        a1, _ = diag.payload("A", 1, 96)
        b0, _ = diag.payload("B", 0, 96)
        self.assertEqual(a0["messages"][:2], a1["messages"][:2])
        self.assertNotEqual(a0["messages"][0], b0["messages"][0])
        self.assertNotIn("id_slot", a0)  # exercise ordinary client scheduling


if __name__ == "__main__":
    unittest.main()
