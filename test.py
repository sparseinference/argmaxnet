"""
File: test.py
By: Peter Caven, peter@sparseinference.com
Description:

Test the net.


"""

from argmaxnet import ArgMaxNet


#=================================================================
modelPath = "/home/pi/models/argmaxnet/ArgMaxNet(10,20,StackDepth(4),gamma=0.9),SGD(0.0001,0.9,weight_decay=0),[81][0.03240].model"
#==================================================================================================

net = ArgMaxNet(10, 20, 4)
net.load(modelPath)
net.test(100, 500)

