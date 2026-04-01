import unittest
import os
import subprocess
import sqlite3
import shutil

class TestRunMapping(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_telemetry_run.db"
        self.results_dir = "test_run_results"
        
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
        
        # Insert sample data (enough for curvature analysis)
        for lb in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            acc = 1.0 if lb < 0.05 else 0.5
            ppl = 10.0 if lb < 0.05 else 50.0
            vram = 2000.0 - (lb * 10000)
            
            cursor.execute("INSERT INTO Intelligence_Metrics (model_name, lambda_b, epoch, passkey_retrieval_acc, perplexity_score) VALUES (?, ?, ?, ?, ?)",
                           ("Qwen2-1.5B", lb, 1, acc, ppl))
            cursor.execute("INSERT INTO Hardware_Metrics (model_name, lambda_b, epoch, peak_vram_mb, ttft_ms, tpot_ms, cache_retention_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           ("Qwen2-1.5B", lb, 1, vram, 50.0, 20.0, 0.5))
        
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        if os.path.exists("results/latest"):
            if os.path.islink("results/latest"):
                os.unlink("results/latest")

    def test_run_mapping_integration(self):
        # Run the script
        cmd = ["venv/bin/python3", "run_mapping.py", "--db", self.db_path, "--output-dir", self.results_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"run_mapping.py failed with output: {result.stdout}\n{result.stderr}")
        self.assertIn("[SUCCESS] Discontinuity Mapping complete.", result.stdout)
        
        # Check if output directory exists
        subdirs = [d for d in os.listdir(self.results_dir) if d.startswith("results_")]
        self.assertEqual(len(subdirs), 1)
        results_path = os.path.join(self.results_dir, subdirs[0])
        
        # Check files
        self.assertTrue(os.path.exists(os.path.join(results_path, "metrics.csv")))
        self.assertTrue(os.path.exists(os.path.join(results_path, "pareto_frontier.png")))
        self.assertTrue(os.path.exists(os.path.join(results_path, "model_comparisons.png")))
        self.assertTrue(os.path.exists(os.path.join(results_path, "evolutionary_trends.png")))
        self.assertTrue(os.path.exists(os.path.join(results_path, "research_summary.md")))

if __name__ == '__main__':
    unittest.main()
