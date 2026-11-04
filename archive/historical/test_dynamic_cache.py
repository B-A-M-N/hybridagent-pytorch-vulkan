from transformers import DynamicCache
import torch

cache = DynamicCache()
key = torch.randn(1, 8, 12, 128)
value = torch.randn(1, 8, 12, 128)
cache.update(key, value, 0)
print('After first update:')
print('  len(key_cache):', len(cache.key_cache))
print('  key_cache[0].shape:', cache.key_cache[0].shape)

key2 = torch.randn(1, 8, 1, 128)
value2 = torch.randn(1, 8, 1, 128)
cache.update(key2, value2, 0)
print('After second update:')
print('  len(key_cache):', len(cache.key_cache))
print('  key_cache[0].shape:', cache.key_cache[0].shape)