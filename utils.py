import torch
from torch import nn
from torch.nn import functional as F


def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    ts = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + ts), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)


if __name__ == '__main__':
    x = torch.arange(10).view(1, 10, 1)
    r = look_around(x, backward = 2, forward = 0, pad_value = 0, dim = 2)
    print(r.shape)
    print(r)
    x = torch.randn(10, 101, 512)
    print(look_around(x, backward = 2, forward = 0, pad_value = 0, dim = 2).shape)