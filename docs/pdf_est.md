# PDF Estimation

## Histogram Method


## Gotchas

### Search Sorted


**Numpy**

```python

```

**PyTorch**

```python
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    h_sorted = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    return h_sorted
```

This is an unofficial implementation. There is still some talks in the PyTorch community to implement this. See github issue [here](https://github.com/pytorch/pytorch/issues/1552). For now, we just use the implementation found in various [implementations](https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/spline_flows.py#L20).