import numpy as np


def normc_init(tensor, std=1.0, **kwargs):
    tensor.data.normal_(0, 1)
    tensor.data *= std / np.sqrt(tensor.data.pow(2).sum(1, keepdim=True))


def fanin_init(tensor, **kwargs):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def truncated_norm_init(tensor, mean=0, std=None, **kwargs):
    size = tensor.shape
    std = std or 1.0/(2*np.sqrt(size[0]))
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

