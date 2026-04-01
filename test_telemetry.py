import unittest
import torch
import sqlite3
import os
from unittest.mock import MagicMock, patch
from run_benchmarks import get_gpu_memory, log_to_db, measure_generation_speed
from init_db import initialize_database

class TestTelemetry(unittest.TestCase):
    def test_gpu_memory_mock(self):
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.max_memory_allocated', return_value=1024 * 1024 * 500):
                mem = get_gpu_memory()
                self.assertEqual(mem, 500.0)

    def test_log_to_db(self):
        db_name = "test_log.db"
        if os.path.exists(db_name):
            os.remove(db_name)
        initialize_database(db_name)
        
        intel_metrics = {'passkey_acc': 1.0, 'ppl': 12.5}
        hw_metrics = {'vram': 4000.0, 'ttft': 50.0, 'tpot': 20.0, 'cache_retention': 0.8}
        
        log_to_db(db_name, "test-model", 0.05, intel_metrics, hw_metrics)
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT model_name, lambda_b, perplexity_score FROM Intelligence_Metrics")
        row = cursor.fetchone()
        self.assertEqual(row[0], "test-model")
        self.assertEqual(row[1], 0.05)
        self.assertEqual(row[2], 12.5)
        
        conn.close()
        os.remove(db_name)

    @patch('run_benchmarks.time.time', side_effect=[0, 0.1, 0.2, 0.5]) # Simulated times
    def test_measure_generation_speed(self, mock_time):
        model = MagicMock()
        tokenizer = MagicMock()
        model.device = torch.device("cpu")
        tokenizer.return_value = {"input_ids": torch.tensor([[0]])}
        
        ttft, tpot = measure_generation_speed(model, tokenizer)
        
        # TTFT: (0.1 - 0) * 1000 = 100
        # TPOT: ((0.5 - 0.2) * 1000) / 20 = 300 / 20 = 15
        self.assertEqual(ttft, 100.0)
        self.assertEqual(tpot, 15.0)

if __name__ == "__main__":
    unittest.main()
