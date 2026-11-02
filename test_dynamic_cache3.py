from transformers import DynamicCache
import torch

cache = DynamicCache()
key = torch.randn(1, 8, 12, 128)
value = torch.randn(1, 8, 12, 128)
cache.update(key, value, 0)
print('layers:', cache.layers)
print('len(layers):', len(cache.layers))
print('layers[0]:', cache.layers[0])
print('layers[0] type:', type(cache.layers[0]))
print('Dir layers[0]:', [x for x in dir(cache.layers[0]) if not x.startswith('_')])

key2 = torch.randn(1, 8, 1, 128)
value2 = torch.randn(1, 8, 1, 128)
cache.update(key2, value2, 0)
print('After second update:')
print('layers[0]:', cache.layers[0])