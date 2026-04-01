import transformers.cache_utils as cu

print("Classes in cache_utils:")
for name in dir(cu):
    if "Cache" in name:
        obj = getattr(cu, name)
        print(f"--- {name} ---")
        if hasattr(obj, "__slots__"):
            print("Slots:", obj.__slots__)
        else:
            print("No slots.")
        
        print("Methods/Attrs:")
        for attr in dir(obj):
            if not attr.startswith("__"):
                print("  ", attr)
