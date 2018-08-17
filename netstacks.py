"""
File: netstacks.py
By: Peter Caven, peter@sparseinference.com
Description:

Blocks and stacks of blocks for neural nets.

Blocks (neural net modules), and compositions of blocks (neural stacks),
are a higher level abstraction than layers.

The blocks defined here are variations on Residual Nets,
where the input layer and the output layer have the same dimension,
while the hidden layer(s) could be thinner or wider.


"""

import torch
import torch.nn as nn



class Block(nn.Module):
    """
    A ResNet module.
    """
    def __init__(self, iDim, hDim):
        super().__init__()
        #----
        self.W0 = nn.Linear(iDim, hDim)
        self.W1 = nn.Linear(hDim, iDim)
        #----
        def LS(w):
            return w.weight.numel() + w.bias.numel()
        self.parameterCount = LS(self.W0) + LS(self.W1)
        #----
    def forward(self, x):
        return self.W1(self.W0(x).abs()) + x
        # return self.W1(self.W0(x).clamp(min=0)) + x



class Stack(nn.Module):
    """
    A stack of blocks.
    """
    def __init__(self, block, stackDepth, iDim, hDim, *args, **kwargs):
        super().__init__()
        #----
        self.stack = nn.ModuleList([block(iDim, hDim, *args, **kwargs) for _ in range(stackDepth)])
        #----
        self.parameterCount = sum(nn.parameterCount for nn in self.stack)
        #----
    def forward(self, x):
        for nn in self.stack:
            x = nn(x).clamp(min=0.0)
        return x

