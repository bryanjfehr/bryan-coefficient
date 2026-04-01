import outlines

print("dir outlines.generator:")
print(dir(outlines.generator))

import inspect
print("\nSignature of outlines.from_transformers:")
print(inspect.signature(outlines.from_transformers))

print("\nAttributes of a Transformers model:")
class DummyModel:
    pass
class DummyTokenizer:
    pass

model = outlines.models.Transformers(DummyModel(), DummyTokenizer())
print(dir(model))

try:
    print(inspect.signature(model.__call__))
except Exception as e:
    print(f"Error getting __call__ signature: {e}")
