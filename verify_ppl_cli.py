import torch
from benchmark_utils import load_benchmark_dataset, calculate_ppl

def main():
    # 1. Load the dataset
    text = load_benchmark_dataset()
    print(f"Dataset loaded. Length: {len(text)} characters.")
    print(f"Sample: {text[:100]}...")

    # 2. Mock a PPL calculation
    # Vocab size 10, Seq length 5
    # Predict index 1 with high confidence for position 0 (which predicts for position 1)
    logits = torch.full((1, 5, 10), -10.0)
    logits[0, 0, 1] = 10.0 # High prob for target index 1 at pos 1
    
    labels = torch.tensor([[0, 1, 0, 0, 0]]) # Target at index 1 is indeed 1
    
    ppl = calculate_ppl(logits, labels)
    print(f"\nCalculated PPL (should be near 1.0 for perfect prediction): {ppl:.4f}")

    # 3. Random calculation
    random_logits = torch.randn(1, 5, 10)
    random_ppl = calculate_ppl(random_logits, labels)
    print(f"Calculated PPL for random logits: {random_ppl:.4f}")

if __name__ == "__main__":
    main()
