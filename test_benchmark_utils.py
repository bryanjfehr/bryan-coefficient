import unittest
from benchmark_utils import inject_passkey

class TestBenchmarkUtils(unittest.TestCase):
    def test_inject_passkey_char_offset(self):
        haystack = "This is a test haystack."
        passkey = "12345"
        # Test character offset 10
        modified_text = inject_passkey(haystack, passkey, offset=10)
        expected_needle = " The secret passkey is 12345. "
        self.assertIn(expected_needle, modified_text)
        # Check if it was inserted at approximately the right place (char 10)
        self.assertEqual(modified_text[10:10+len(expected_needle)], expected_needle)

    def test_inject_passkey_deterministic(self):
        haystack = "Another test haystack for deterministic check."
        passkey = "99999"
        res1 = inject_passkey(haystack, passkey, offset=5)
        res2 = inject_passkey(haystack, passkey, offset=5)
        self.assertEqual(res1, res2)

    def test_inject_passkey_out_of_bounds(self):
        haystack = "Short"
        passkey = "000"
        # Should handle offset > len(haystack) gracefully, e.g., append at end or cap at len
        modified_text = inject_passkey(haystack, passkey, offset=100)
        self.assertTrue(modified_text.endswith(" The secret passkey is 000. "))

    def test_verify_passkey_success(self):
        from benchmark_utils import verify_passkey
        response = "The secret passkey is 12345. I hope that's correct."
        passkey = "12345"
        self.assertTrue(verify_passkey(response, passkey))

    def test_verify_passkey_failure(self):
        from benchmark_utils import verify_passkey
        response = "The secret passkey is 00000."
        passkey = "12345"
        self.assertFalse(verify_passkey(response, passkey))

    def test_verify_passkey_empty_response(self):
        from benchmark_utils import verify_passkey
        response = ""
        passkey = "12345"
        self.assertFalse(verify_passkey(response, passkey))

    def test_calculate_ppl_basic(self):
        import torch
        from benchmark_utils import calculate_ppl
        # Mock logits: (batch, seq, vocab) -> (1, 2, 3)
        # We want to predict token 1 (at index 1) poorly.
        # Logits at index 0 (predicting token 1): high prob for index 0, target is 1.
        logits = torch.tensor([[[10.0, -10.0, -10.0], [ 0.0, 0.0, 0.0]]])
        labels = torch.tensor([[0, 1]])
        
        ppl = calculate_ppl(logits, labels)
        self.assertIsInstance(ppl, float)
        self.assertGreater(ppl, 100.0) # Loss ~ 20, exp(20) is huge

    def test_calculate_ppl_perfect_prediction(self):
        import torch
        from benchmark_utils import calculate_ppl
        # We want to predict token 1 (at index 1) perfectly.
        # Logits at index 0 (predicting token 1): high prob for index 1, target is 1.
        logits = torch.tensor([[[ -50.0, 50.0, -50.0], [ 0.0, 0.0, 0.0]]])
        labels = torch.tensor([[0, 1]])
        ppl = calculate_ppl(logits, labels)
        self.assertAlmostEqual(ppl, 1.0, places=2)

    def test_load_benchmark_dataset_fallback(self):
        from benchmark_utils import load_benchmark_dataset
        # Should return a string or list of strings
        dataset = load_benchmark_dataset(split="test")
        self.assertIsInstance(dataset, str)
        self.assertGreater(len(dataset), 100)
        self.assertIn("Wiki", dataset) # Should look like Wiki text

    def test_persist_metrics(self):
        import sqlite3
        import os
        from benchmark_utils import persist_metrics
        from init_db import initialize_database
        
        db_name = "test_persist.db"
        if os.path.exists(db_name):
            os.remove(db_name)
        initialize_database(db_name)
        
        metrics = {
            "model_name": "test-model",
            "lambda_b": 0.05,
            "passkey_acc": 1.0,
            "ppl": 12.5,
            "peak_vram_mb": 4500.0,
            "ttft_ms": 120.0,
            "tpot_ms": 25.0,
            "cache_retention_pct": 0.85
        }
        
        persist_metrics(db_name, metrics)
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT model_name, lambda_b, passkey_retrieval_acc, perplexity_score FROM Intelligence_Metrics")
        intel_row = cursor.fetchone()
        self.assertEqual(intel_row[0], "test-model")
        self.assertEqual(intel_row[1], 0.05)
        self.assertEqual(intel_row[2], 1.0)
        self.assertEqual(intel_row[3], 12.5)
        
        cursor.execute("SELECT peak_vram_mb, ttft_ms, tpot_ms, cache_retention_pct FROM Hardware_Metrics")
        hw_row = cursor.fetchone()
        self.assertEqual(hw_row[0], 4500.0)
        self.assertEqual(hw_row[1], 120.0)
        self.assertEqual(hw_row[2], 25.0)
        self.assertEqual(hw_row[3], 0.85)
        
        conn.close()
        os.remove(db_name)

if __name__ == "__main__":
    unittest.main()
