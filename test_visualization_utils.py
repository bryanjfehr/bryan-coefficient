import unittest
import os
import pandas as pd
import plotly.graph_objects as go
from visualization_utils import plot_pareto_frontier, plot_model_comparisons

class TestVisualizationUtils(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample data
        self.df = pd.DataFrame([
            {"model_name": "Qwen2-1.5B", "lambda_b": 0.01, "passkey_retrieval_acc": 1.0, "peak_vram_mb": 2000.0},
            {"model_name": "Qwen2-1.5B", "lambda_b": 0.05, "passkey_retrieval_acc": 0.8, "peak_vram_mb": 1500.0},
            {"model_name": "Phi-3-mini", "lambda_b": 0.01, "passkey_retrieval_acc": 0.99, "peak_vram_mb": 2500.0},
        ])

    def tearDown(self):
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)

    def test_plot_pareto_frontier(self):
        # This should fail because plot_pareto_frontier is not implemented
        output_path = os.path.join(self.output_dir, "pareto.png")
        fig = plot_pareto_frontier(self.df, output_path=output_path)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(os.path.exists(output_path))

    def test_plot_model_comparisons(self):
        # This should fail because plot_model_comparisons is not implemented
        output_path = os.path.join(self.output_dir, "comparisons.png")
        fig = plot_model_comparisons(self.df, output_path=output_path)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(os.path.exists(output_path))

    def test_plot_evolutionary_trends(self):
        from visualization_utils import plot_evolutionary_trends
        # Sample multi-epoch data
        df = pd.DataFrame([
            {"epoch": 1, "model_name": "Qwen2-1.5B", "passkey_retrieval_acc": 0.9, "peak_vram_mb": 2000.0},
            {"epoch": 2, "model_name": "Qwen2-1.5B", "passkey_retrieval_acc": 0.95, "peak_vram_mb": 1800.0},
            {"epoch": 3, "model_name": "Qwen2-1.5B", "passkey_retrieval_acc": 0.98, "peak_vram_mb": 1700.0},
        ])
        output_path = os.path.join(self.output_dir, "evolution.png")
        fig = plot_evolutionary_trends(df, output_path=output_path)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
