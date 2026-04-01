import unittest
import sqlite3
import pandas as pd
import os
import csv
from analysis_utils import fetch_metrics, export_to_csv

class TestAnalysisUtils(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_telemetry.db"
        self.csv_path = "test_metrics.csv"
        
        # Initialize test database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE Intelligence_Metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            epoch INTEGER DEFAULT 1,
            model_name TEXT,
            lambda_b REAL,
            passkey_retrieval_acc REAL,
            perplexity_score REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE Hardware_Metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            epoch INTEGER DEFAULT 1,
            model_name TEXT,
            lambda_b REAL,
            peak_vram_mb REAL,
            ttft_ms REAL,
            tpot_ms REAL,
            cache_retention_pct REAL
        )
        ''')
        
        # Insert sample data
        cursor.execute("INSERT INTO Intelligence_Metrics (model_name, lambda_b, epoch, passkey_retrieval_acc, perplexity_score) VALUES (?, ?, ?, ?, ?)",
                       ("Qwen2-1.5B", 0.01, 1, 1.0, 12.5))
        cursor.execute("INSERT INTO Hardware_Metrics (model_name, lambda_b, epoch, peak_vram_mb, ttft_ms, tpot_ms, cache_retention_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       ("Qwen2-1.5B", 0.01, 1, 2000.0, 50.0, 20.0, 0.5))
        
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_fetch_metrics(self):
        # This should fail because fetch_metrics is not implemented
        df = fetch_metrics(self.db_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['model_name'], "Qwen2-1.5B")
        self.assertEqual(df.iloc[0]['lambda_b'], 0.01)
        self.assertEqual(df.iloc[0]['passkey_retrieval_acc'], 1.0)
        self.assertEqual(df.iloc[0]['peak_vram_mb'], 2000.0)

    def test_export_to_csv(self):
        # This should fail because export_to_csv is not implemented
        df = pd.DataFrame([{"model_name": "Qwen2-1.5B", "lambda_b": 0.01, "passkey_retrieval_acc": 1.0}])
        export_to_csv(df, self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path))
        
        with open(self.csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['model_name'], "Qwen2-1.5B")

    def test_calculate_discontinuity(self):
        from analysis_utils import calculate_discontinuity
        # Create a sample dataset where there is a clear drop in performance at lambda_b = 0.05
        data = [
            {"lambda_b": 0.01, "passkey_retrieval_acc": 1.0, "perplexity_score": 10.0},
            {"lambda_b": 0.02, "passkey_retrieval_acc": 1.0, "perplexity_score": 10.1},
            {"lambda_b": 0.03, "passkey_retrieval_acc": 1.0, "perplexity_score": 10.2},
            {"lambda_b": 0.04, "passkey_retrieval_acc": 0.98, "perplexity_score": 10.5},
            {"lambda_b": 0.05, "passkey_retrieval_acc": 0.80, "perplexity_score": 15.0}, # Discontinuity
            {"lambda_b": 0.06, "passkey_retrieval_acc": 0.40, "perplexity_score": 30.0},
        ]
        df = pd.DataFrame(data)
        
        # Test passkey accuracy discontinuity
        lb_acc = calculate_discontinuity(df, metric="passkey_retrieval_acc", method="elbow")
        self.assertEqual(lb_acc, 0.05)
        
        # Test fixed retention threshold (e.g., 90%)
        lb_fixed = calculate_discontinuity(df, metric="passkey_retrieval_acc", method="retention", threshold=0.90)
        self.assertEqual(lb_fixed, 0.04) # Last point above 90%

    def test_calculate_discontinuity_ppl(self):
        from analysis_utils import calculate_discontinuity
        # Test with perplexity (lower is better)
        data = [
            {"lambda_b": 0.01, "perplexity_score": 10.0},
            {"lambda_b": 0.02, "perplexity_score": 10.1},
            {"lambda_b": 0.03, "perplexity_score": 10.2},
            {"lambda_b": 0.04, "perplexity_score": 11.0},
            {"lambda_b": 0.05, "perplexity_score": 25.0}, # Discontinuity
            {"lambda_b": 0.06, "perplexity_score": 50.0},
        ]
        df = pd.DataFrame(data)
        
        # Test elbow with perplexity
        lb_elbow = calculate_discontinuity(df, metric="perplexity_score", method="elbow")
        self.assertEqual(lb_elbow, 0.04)
        
        # Test retention with perplexity (threshold is multiplier on baseline)
        # 80% retention = 1.25 * baseline
        lb_retention = calculate_discontinuity(df, metric="perplexity_score", method="retention", threshold=0.80)
        self.assertEqual(lb_retention, 0.04)

    def test_calculate_discontinuity_edge_cases(self):
        from analysis_utils import calculate_discontinuity
        # Empty df
        self.assertIsNone(calculate_discontinuity(pd.DataFrame()))
        
        # Less than 3 points for elbow
        df = pd.DataFrame([{"lambda_b": 0.01, "acc": 1.0}, {"lambda_b": 0.02, "acc": 0.5}])
        self.assertEqual(calculate_discontinuity(df, metric="acc", method="elbow"), 0.02)
        
        # No points above threshold for retention
        df = pd.DataFrame([{"lambda_b": 0.01, "acc": 0.1}])
        self.assertEqual(calculate_discontinuity(df, metric="acc", method="retention", threshold=0.9), 0.01)

    def test_generate_research_summary(self):
        from analysis_utils import generate_research_summary
        df = pd.DataFrame([
            {"model_name": "Qwen2-1.5B", "lambda_b": 0.01, "passkey_retrieval_acc": 1.0, "perplexity_score": 10.0, "peak_vram_mb": 2000.0},
            {"model_name": "Qwen2-1.5B", "lambda_b": 0.05, "passkey_retrieval_acc": 0.8, "perplexity_score": 15.0, "peak_vram_mb": 1500.0},
        ])
        summary = generate_research_summary(df)
        self.assertIn("# Research Summary", summary)
        self.assertIn("Qwen2-1.5B", summary)
        self.assertIn("model_name", summary) # Table header

if __name__ == '__main__':
    unittest.main()
