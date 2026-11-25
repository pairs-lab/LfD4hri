import json
from pathlib import Path
import tempfile
import unittest
from VLA.rdt.rebind_declared_prompt_cache import verified_metadata


class RdtPromptCacheRebindTest(unittest.TestCase):
    def metadata(self):
        return {"additional_prompts_scope": "declared_task_vocabulary_no_observations",
                "encoder_weight_precision": "bfloat16", "load_in_4bit": False,
                "load_in_8bit": False, "source_index": "/original/index.json",
                "source_index_sha256": "original-index-hash"}

    def test_preserves_original_binding_and_discloses_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory)/"index.json"
            index.write_text(json.dumps([{"task": {"prompt": "Touch the red apple"}}]))
            result = verified_metadata(self.metadata(), index, "original-manifest", "original-embeddings")
            self.assertEqual(result["original_source_index_sha256"], "original-index-hash")
            self.assertEqual(result["reused_cache_manifest_sha256"], "original-manifest")
            self.assertTrue(result["reused_cached_embeddings_not_recomputed"])
            self.assertEqual(result["training_prompt_count"], 1)

    def test_rejects_quantized_or_undeclared_grammar_cache(self):
        for key, value in (("load_in_4bit", True), ("encoder_weight_precision", "nf4_4bit"),
                           ("additional_prompts_scope", None)):
            old = self.metadata(); old[key] = value
            with self.assertRaises(ValueError):
                verified_metadata(old, Path("not-read-on-failed-cache"), "hash", "hash")
