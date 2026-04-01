import unittest
from unittest.mock import patch
import torch
from profiler_utils import get_peak_vram

class TestProfilerUtils(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.max_memory_allocated', return_value=1024 * 1024 * 100) # 100 MB
    def test_get_peak_vram_cuda(self, mock_max_mem, mock_cuda_avail):
        vram = get_peak_vram(unit="MB")
        self.assertEqual(vram, 100.0)
        mock_max_mem.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.max_memory_allocated', return_value=1024 * 1024 * 1024) # 1 GB
    def test_get_peak_vram_gb(self, mock_max_mem, mock_cuda_avail):
        vram = get_peak_vram(unit="GB")
        self.assertEqual(vram, 1.0)

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_peak_vram_no_cuda(self, mock_cuda_avail):
        vram = get_peak_vram()
        self.assertEqual(vram, 0.0)

    def test_measure_generation_latency_logic(self):
        from unittest.mock import MagicMock
        from profiler_utils import measure_generation_latency
        import torch
        
        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()
        
        # Mock tokenizer to return input_ids
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        # Two-pass generate:
        # 1. generate(max_new_tokens=1)
        # 2. generate(max_new_tokens=3)
        with patch('profiler_utils.time.time', side_effect=[
            0.0,  # start_ttft
            0.1,  # end_ttft (TTFT = 100ms)
            0.2,  # start_tpot
            0.5   # end_total (Total = 300ms, TPOT = (300-100)/2 = 100ms)
        ]):
            ttft, tpot = measure_generation_latency(model, tokenizer, "test prompt", max_new_tokens=3)
            
            self.assertAlmostEqual(ttft, 100.0)
            self.assertAlmostEqual(tpot, 100.0)

if __name__ == "__main__":
    unittest.main()
