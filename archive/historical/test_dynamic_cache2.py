from transformers import DynamicCache
import torch

cache = DynamicCache()
key = torch.randn(1, 8, 12, 128)
value = torch.randn(1, 8, 12, 128)
cache.update(key, value, 0)
print('Dir:', [x for x in dir(cache) if not x.startswith('_')])