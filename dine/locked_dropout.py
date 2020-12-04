import torch
import torch.nn as nn
import numpy as np
import model


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = m.div_(1 - dropout).detach()
        mask = mask.expand_as(x)
        return mask * x


class MyLockedDropout(nn.Module):
    def __init__(self):
        super(MyLockedDropout, self).__init__()

    def forward(self, layer, dropout=0.5):
        # 3/11/20 - dropout and div will occur at Train / MC eval
        # if there is no dropout OR we in normal eval:
        if dropout == 0 or not self.training:
            return layer
        else:
            # building a mask
            _seq_len = layer.data.shape[0]
            batch_size = layer.data.shape[1]
            neurons = layer.data.shape[2]
            probability = 1 - dropout
            # Tensor.data.new = duplicate type and device of the tensor
            mask = layer.data.new(np.random.binomial(1, p=probability, size=(1, batch_size, neurons)))
            # Multiplying each 1 in the mask in (1/(1-dropout)) to maintain probability space
            mask = torch.mul(mask, 1 / (1 - dropout)).detach()
            # Tensor.expand_as(x) = will generate many mask as the size of x first element
            mask = mask.expand_as(layer)
            return layer * mask  # dropping-out layer's neurons
