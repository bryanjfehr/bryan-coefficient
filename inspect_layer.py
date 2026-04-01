import torch
import transformers.cache_utils as cu
dc = cu.DynamicCache()
dl = cu.DynamicLayer()
dc.layers.append(dl)
print("DynamicLayer dict:", dl.__dict__)
print("DynamicLayer slots:", hasattr(dl, "__slots__"))
