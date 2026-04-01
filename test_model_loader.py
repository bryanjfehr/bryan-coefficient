import unittest
from unittest.mock import patch, MagicMock
import sys

# We need to be able to import from run_benchmarks
from run_benchmarks import load_model

class TestModelLoader(unittest.TestCase):
    @patch('run_benchmarks.AutoModelForCausalLM.from_pretrained')
    @patch('run_benchmarks.AutoTokenizer.from_pretrained')
    @patch('run_benchmarks.BitsAndBytesConfig')
    def test_load_model_config(self, mock_bnb_config, mock_tokenizer, mock_model):
        model_name = "test-model"
        
        # Setup mocks
        mock_bnb_instance = MagicMock()
        mock_bnb_config.return_value = mock_bnb_instance
        
        load_model(model_name)
        
        # Verify BitsAndBytesConfig was called with correct parameters
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype='float16', # Standard for BitsAndBytes NF4
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
        
        # Verify model was loaded with correct parameters
        mock_model.assert_called_once()
        args, kwargs = mock_model.call_args
        self.assertEqual(args[0], model_name)
        self.assertEqual(kwargs['quantization_config'], mock_bnb_instance)
        self.assertEqual(kwargs['device_map'], 'auto')
        self.assertEqual(kwargs['attn_implementation'], 'eager')
        
        # Verify tokenizer was loaded
        mock_tokenizer.assert_called_once_with(model_name)

if __name__ == "__main__":
    unittest.main()
