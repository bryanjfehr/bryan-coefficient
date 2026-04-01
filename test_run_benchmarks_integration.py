import unittest
from unittest.mock import MagicMock, patch
import torch
import os

class TestRunBenchmarksIntegration(unittest.TestCase):
    @patch('run_benchmarks.AutoModelForCausalLM.from_pretrained')
    @patch('run_benchmarks.AutoTokenizer.from_pretrained')
    @patch('run_benchmarks.persist_metrics')
    @patch('run_benchmarks.run_passkey_test')
    @patch('run_benchmarks.run_ppl_test')
    @patch('run_benchmarks.get_peak_vram')
    @patch('run_benchmarks.measure_generation_latency')
    def test_main_execution_flow(self, mock_latency, mock_vram, mock_ppl, mock_passkey, mock_persist, mock_tokenizer, mock_model):
        from run_benchmarks import main
        import sys
        
        # Setup mocks
        mock_passkey.return_value = (1.0, "mock_resp")
        mock_ppl.return_value = 15.0
        mock_vram.return_value = 4000.0
        mock_latency.return_value = (100.0, 20.0)
        
        # Mock CLI arguments
        test_args = ["run_benchmarks.py", "--model", "mock-model", "--lambda_b", "0.01"]
        with patch.object(sys, 'argv', test_args):
            with patch('run_benchmarks.apply_bryan_coefficient') as mock_surgery:
                main()
                
                # Verify surgery was applied
                mock_surgery.assert_called_once()
                
                # Verify all benchmarks were called
                mock_passkey.assert_called_once()
                mock_ppl.assert_called_once()
                mock_vram.assert_called_once()
                mock_latency.assert_called_once()
                
                # Verify persistence was called with aggregated metrics
                mock_persist.assert_called_once()
                args, _ = mock_persist.call_args
                metrics = args[1]
                self.assertEqual(metrics["model_name"], "mock-model")
                self.assertEqual(metrics["passkey_acc"], 1.0)
                self.assertEqual(metrics["ppl"], 15.0)
                self.assertEqual(metrics["peak_vram_mb"], 4000.0)

if __name__ == "__main__":
    unittest.main()
