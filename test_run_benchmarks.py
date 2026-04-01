import unittest
import subprocess
import sys

class TestRunBenchmarks(unittest.TestCase):
    def test_argument_parsing(self):
        # Run the script with some arguments and check the output
        command = [sys.executable, "run_benchmarks.py", "--model", "test-model", "--lambda_b", "0.05"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Evaluating λB = 0.05 on test-model", result.stdout)
        
    def test_missing_arguments(self):
        # Run the script without arguments and expect an error
        command = [sys.executable, "run_benchmarks.py"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the following arguments are required: --model, --lambda_b", result.stderr)

if __name__ == "__main__":
    unittest.main()
