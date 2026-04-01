import sys
from benchmark_utils import inject_passkey, verify_passkey

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 verify_passkey_cli.py <passkey> <offset> [haystack_file]")
        print("Example: python3 verify_passkey_cli.py 84729 5000")
        sys.exit(1)

    passkey = sys.argv[1]
    offset = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'r') as f:
            haystack = f.read()
    else:
        # Generate a default haystack (~2000 words / 12k chars)
        haystack = "Lorem ipsum dolor sit amet. " * 2000

    modified = inject_passkey(haystack, passkey, offset)
    
    # Display context around the injection point
    print(f"\n--- Injection Context (Offset {offset}) ---")
    start = max(0, offset - 40)
    end = min(len(modified), offset + len(passkey) + 60)
    print(f"...{modified[start:end]}...")
    print("-" * 40)
    
    success = verify_passkey(modified, passkey)
    print(f"Verification Result: {'SUCCESS' if success else 'FAILURE'}")
    
    if success:
        print(f"Confirmed: Passkey '{passkey}' is present in the modified text.")

if __name__ == "__main__":
    main()
