"""
File: test.py
By: Peter Caven, peter@sparseinference.com
Description:

Test the net.


"""

from argmaxnet import ArgMaxNet
import os

#=================================================================
modelPath = os.path.expanduser('~/models/argmaxnet/ArgMaxNet : StackDepth=3, Block(10,60),[111][0.02220].model')
#==================================================================================================

net = ArgMaxNet(3, 10, 60)
net.load(modelPath)
net.test(100, 500)

