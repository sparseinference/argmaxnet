"""
File: train.py
By: Peter Caven, peter@sparseinference.com
Description:

Train the net.


"""

from argmaxnet import ArgMaxNet
import os

#=================================================================
modelPath = os.path.expanduser('~/models/argmaxnet')
#==================================================================================================

ArgMaxNet(3, 10, 60).learn(modelPath, learningRate=0.0001, gamma=0.98)

