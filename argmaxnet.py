"""
File: argmaxnet.py
By: Peter Caven, peter@sparseinference.com
Description:

Learn to compute the argmax function.


"""

import torch
import torch.nn as nn

from netstacks import Block,Stack

import time


class ArgMaxNet(nn.Module):  
    """
    A neural net that learns to compute the argmax function.
    """
    def __init__(self, stackDepth, iDim, hDim, *args):
        super().__init__()
        #----
        self.iDim = iDim
        self.hDim = hDim
        self.stackDepth = stackDepth
        #----
        self.Rn = Stack(Block, stackDepth, iDim, hDim)
        #----
        self.Wn = nn.Linear(iDim, 1)
        #----
        def LS(w):
            return w.weight.numel() + w.bias.numel()
        self.parameterCount = self.Rn.parameterCount + LS(self.Wn)
        #----
    #--------------------------------------
    def forward(self, scores):
        ln = self.Wn(self.Rn(scores)).abs()
        return ln.t()[0]
    #--------------------------------------
    def process(self):
        """
        Compute the forward and backward operations 
        with random data on every batch.
        """
        scores = torch.nn.functional.softmax(torch.randn(25, self.iDim), dim=1)
        labels = torch.tensor(torch.argmax(scores, dim=1), dtype=torch.float32)
        (self(scores) - labels).pow(2.0).sum().backward()
        # (self(scores) - labels).abs().sum().backward()
    #--------------------------------------
    def save(self, path, epochs, err):
         torch.save(self.state_dict(), path + f"/{self.description},[{epochs}][{err:.5f}].model")
    #--------------------------------------    
    def load(self, modelPath):
        self.load_state_dict(torch.load(modelPath))
    #--------------------------------------
    def stats(self, batches=10, batchSize=500):
        """
        Return error rate and mean loss.
        """
        testSize = batches * batchSize
        incorrect = 0
        loss = 0.0
        with torch.no_grad():
            for _ in range(batches):
                scores = torch.nn.functional.softmax(torch.randn(batchSize, self.iDim), dim=1)
                labels = torch.tensor(torch.argmax(scores, dim=1), dtype=torch.float32)
                outputs = self(scores)
                loss += (outputs - labels).pow(2.0).sum().item()
                # loss += (outputs - labels).abs().sum().item()
                incorrect += (outputs.round() != labels).sum().item()
        return incorrect/testSize,loss/testSize
    #--------------------------------------
    def learn(self, modelPath, learningRate = 0.001, momentum = 0.9, weightDecay=0, gamma=1.0):
        """
        Train the ArgMax neural net. 
        """
        self.description =  f"ArgMaxNet : StackDepth={self.stackDepth}, Block({self.iDim},{self.hDim})"
        optDescription = f"\n            opt: SGD(lr={learningRate} reduced by {gamma} each episode, momentum={momentum}, weight_decay={weightDecay})"
        print(self.description + optDescription)
        print(f"Parameter count = {self.parameterCount}")
        #------------------
        epochs = 0
        teErr = 1.0
        learningRate = learningRate/gamma
        startTime = time.time()
        try:
            while learningRate > 1.0e-6:  # or: teErr > 0.0:
                epochs += 1
                self.train(mode=True)
                #--------------------
                learningRate *= gamma
                optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)
                optimizer.zero_grad()
                #--------------------
                for _ in range(15000):
                    self.process()
                    optimizer.step()
                    optimizer.zero_grad()
                #--------------------
                self.train(mode=False)
                teErr,teLoss = self.stats()
                elapsedTime = (time.time() - startTime)/(60*60)
                print(f"[{epochs:5d}] Loss:{teLoss:>9.6f}   Err:{teErr:>9.6f}  elapsed: {elapsedTime:>9.6f} hours, lr={learningRate:>10.8f}")
                #--------------------
        except KeyboardInterrupt:
            pass
        finally:
            self.save(modelPath, epochs, teErr)
    #--------------------------------------
    def test(self, batches, batchSize):
        print(f"Testing {batches} batches of {batchSize} random scores ...")
        startTime = time.time()
        teErr,teLoss = self.stats(batches=batches, batchSize=batchSize)
        elapsedTime = (time.time() - startTime)
        msPerInstance = elapsedTime/(batches * batchSize) * 1000.0
        print(f"Loss:{teLoss:>9.6f}   Err:{teErr:>9.6f}  elapsed:{elapsedTime/(60*60):>9.6f} hours  perInstance:{msPerInstance:>9.6f} ms")
    #--------------------------------------
        

