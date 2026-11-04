from transformers import DynamicCache
import torch

cache = DynamicCache()
key = torch.randn(1, 8, 12, 128)
value = torch.randn(1, 8, 12, 128)
cache.update(key, value, 0)
layer = cache.layers[0]
print('keys shape:', layer.keys.shape)
print('values shape:', layer.values.shape)
print('keys dtype:', layer.keys.dtype)
print('values dtype:', layer.values.dtype)

key2 = torch.randn(1, 8, 1, 128)
value2 = torch.randn(1, 8, 1, 128)
cache.update(key2, value2, 0)
print('After second update:')
print('keys shape:', layer.keys.shape)
print('values shape:', layer.values.shape)