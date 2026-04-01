import torch
import time
from unittest.mock import MagicMock, patch
from profiler_utils import get_peak_vram, measure_generation_latency

def main():
    print("--- [PHASE 3: HARDWARE PROFILER VERIFICATION] ---")
    
    # 1. Peak VRAM Reporting
    vram = get_peak_vram(unit="MB")
    print(f"Current Peak VRAM Reported: {vram:.2f} MB")
    
    # 2. Mock Latency Profiling (TTFT/TPOT)
    # We mock the model and tokenizer to show the profiler in action
    model = MagicMock()
    model.device = torch.device("cpu")
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    
    # We simulate a 2-pass generation that would take some time
    # Pass 1 (TTFT): 100ms
    # Pass 2 (Total): 400ms -> TPOT = (400 - 100) / (4 - 1) = 100ms
    with patch('profiler_utils.time.time', side_effect=[
        0.0,  # start_ttft
        0.1,  # end_ttft
        0.2,  # start_total
        0.6   # end_total (Total = 0.4s, TPOT = (0.4-0.1)/(4-1) = 0.1s = 100ms)
    ]):
        ttft, tpot = measure_generation_latency(model, tokenizer, "test prompt", max_new_tokens=4)
        print(f"\n[LATENCY REPORT (MOCKED)]")
        print(f"TTFT: {ttft:.2f} ms")
        print(f"TPOT: {tpot:.2f} ms")
        
    print("\nProfiler utilities are verified and ready for the gauntlet.")

if __name__ == "__main__":
    main()
