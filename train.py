"""
File: train.py
By: Peter Caven, peter@sparseinference.com
Description:

Train the net.


"""

from argmaxnet import ArgMaxNet


#=================================================================
modelPath = '/home/pi/models/argmaxnet'
#==================================================================================================

ArgMaxNet(10, 20, 4).learn(modelPath)

