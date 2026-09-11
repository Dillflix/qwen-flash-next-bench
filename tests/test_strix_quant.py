import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import qwen_strix_quant as quant


def fixture():
    tensors = []
    for layer in range(48):
        for projection in ("gate", "up", "down"):
            dims = [640, 2560, 512] if projection == "down" else [2560, 640, 512]
            tensors.append({"name": f"blk.{layer}.ffn_{projection}_exps.weight",
                            "dimensions": dims, "type": "BF16"})
    tensors.append({"name": "per_layer_token_embd.weight", "dimensions": [160, 320001536], "type": "BF16"})
    return {"metadata": {"general.architecture": "qwen4exp"}, "tensors": tensors}


class StrixQuantTests(unittest.TestCase):
    def test_shards(self):
        self.assertEqual(quant.source_files(Path("x-00001-of-00002.gguf")),
                         [Path("x-00001-of-00002.gguf"), Path("x-00002-of-00002.gguf")])
        with self.assertRaises(ValueError):
            quant.source_files(Path("x-00002-of-00002.gguf"))

    def test_source_and_exact_recipe(self):
        with patch.object(quant, "inventory", return_value=fixture()):
            before = quant.checked_tensors([Path("source")], source=True)
        after = copy.deepcopy(before)
        for tensor in after.values():
            for pattern, qtype, _ in quant.read_recipe(quant.RECIPE):
                if pattern.search(tensor["name"]):
                    tensor["type"] = qtype
                    break
        quant.verify(before, after)
        after["blk.0.ffn_down_exps.weight"]["type"] = "Q8_0"
        with self.assertRaisesRegex(ValueError, "Wrong output type"):
            quant.verify(before, after)

    def test_model_metadata_only_in_first_shard(self):
        report = fixture()
        shards = [
            {"metadata": report["metadata"], "tensors": report["tensors"][:-1]},
            {"metadata": {}, "tensors": report["tensors"][-1:]},
        ]
        with patch.object(quant, "inventory", side_effect=shards):
            tensors = quant.checked_tensors([Path("first"), Path("second")], source=True)
        self.assertEqual(len(tensors), 145)
        self.assertIn("per_layer_token_embd.weight", tensors)

    def test_first_shard_requires_architecture(self):
        report = fixture()
        for architecture in (None, "llama"):
            report["metadata"] = {} if architecture is None else {"general.architecture": architecture}
            with patch.object(quant, "inventory", return_value=report):
                with self.assertRaisesRegex(ValueError, "Not a qwen4exp"):
                    quant.checked_tensors([Path("first")], source=True)

    def test_later_shard_cannot_contradict_architecture(self):
        first = fixture()
        later = {"metadata": {"general.architecture": "llama"}, "tensors": []}
        with patch.object(quant, "inventory", side_effect=[first, later]):
            with self.assertRaisesRegex(ValueError, "Not a qwen4exp"):
                quant.checked_tensors([Path("first"), Path("second")], source=True)

    def test_rejects_requantization_and_bad_shape(self):
        report = fixture()
        report["tensors"][0]["type"] = "Q5_K"
        with patch.object(quant, "inventory", return_value=report):
            with self.assertRaisesRegex(ValueError, "Already quantized"):
                quant.checked_tensors([Path("source")], source=True)
        report = fixture()
        report["tensors"][0]["dimensions"][0] = 640
        with patch.object(quant, "inventory", return_value=report):
            with self.assertRaisesRegex(ValueError, "expert shape"):
                quant.checked_tensors([Path("source")], source=True)

    def test_rejects_ple16_and_extra_mtp(self):
        report = fixture()
        report["tensors"][-1]["name"] = "ple_ngram_embd.0.weight"
        with patch.object(quant, "inventory", return_value=report):
            with self.assertRaisesRegex(ValueError, "joined PLE"):
                quant.checked_tensors([Path("source")], source=True)
        report = fixture()
        report["tensors"].append({"name": "blk.48.ffn_down_exps.weight", "dimensions": [640, 2560, 512], "type": "BF16"})
        with patch.object(quant, "inventory", return_value=report):
            with self.assertRaisesRegex(ValueError, "exactly 144"):
                quant.checked_tensors([Path("source")], source=True)


if __name__ == "__main__":
    unittest.main()
