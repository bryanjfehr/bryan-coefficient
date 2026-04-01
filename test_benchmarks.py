import unittest
import torch
from unittest.mock import MagicMock
from run_benchmarks import evaluate_ppl, evaluate_passkey

class TestBenchmarks(unittest.TestCase):
    def test_ppl_calculation(self):
        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Ensure model.device is something torch.Tensor.to() can handle
        model.device = torch.device("cpu")
        
        # Mock model output for a single token
        # logits: (batch, seq, vocab)
        mock_logits = torch.tensor([[[10.0, 0.0, 0.0]]]) # High probability for index 0
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_output.loss = torch.tensor(0.5) # Add a loss value
        model.return_value = mock_output
        
        # Mock tokenizer
        tokenizer.return_value = {"input_ids": torch.tensor([[0]])}
        
        # If the model predicts the token with 100% confidence, loss is small, PPL is > 1.0
        ppl = evaluate_ppl(model, tokenizer, "test text")
        self.assertIsInstance(ppl, float)
        self.assertGreaterEqual(ppl, 1.0)
        
    def test_passkey_retrieval_logic(self):
        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()
        model.device = torch.device("cpu")
        
        # Case 1: Correct retrieval
        tokenizer.decode.return_value = "The passkey is 12345"
        acc = evaluate_passkey(model, tokenizer, passkey="12345")
        self.assertEqual(acc, 1.0)
        
        # Case 2: Incorrect retrieval
        tokenizer.decode.return_value = "The passkey is 00000"
        acc = evaluate_passkey(model, tokenizer, passkey="12345")
        self.assertEqual(acc, 0.0)

if __name__ == "__main__":
    unittest.main()
