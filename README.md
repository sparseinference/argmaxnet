# argmaxnet  


An experiment in training a fully connected residual net to learn the argmax function.  
This seems surprisingly difficult for such a simple problem.

<hr>

Experiment (*argmaxnet.py*):  

- Raspberry Pi 3B+
- Python 3.7
- PyTorch 0.5.0a0+f40ed54
- Stacked Residual Net, 4 residual modules deep.
- the ResNet module is wide in the hidden layer with the input and output dimensions equal  
(short, wide ResNets seem to converge more quickly)
- the activation function is <code>abs()</code> not <code>ReLU</code>    


<hr>
Training run:

<pre>
➜  argmaxnet git:(master) ✗ python3 train.py
ArgMaxNet(10,20,StackDepth(4),gamma=0.9),SGD(0.0001,0.9,weight_decay=0)
Parameter count = 1731
[    1] Loss: 0.936503   Err: 0.404400  elapsed:  0.029877 hours, lr=0.00010000
[    2] Loss: 0.775802   Err: 0.188200  elapsed:  0.056166 hours, lr=0.00009000
[    3] Loss: 0.466706   Err: 0.162600  elapsed:  0.082512 hours, lr=0.00008100
[    4] Loss: 0.649453   Err: 0.305800  elapsed:  0.108644 hours, lr=0.00007290
[    5] Loss: 0.489783   Err: 0.155000  elapsed:  0.134843 hours, lr=0.00006561
[    6] Loss: 0.506927   Err: 0.388000  elapsed:  0.161110 hours, lr=0.00005905
[    7] Loss: 0.417111   Err: 0.155000  elapsed:  0.187329 hours, lr=0.00005314
[    8] Loss: 0.548050   Err: 0.259000  elapsed:  0.213462 hours, lr=0.00004783
[    9] Loss: 0.554294   Err: 0.209400  elapsed:  0.239546 hours, lr=0.00004305
[   10] Loss: 0.454473   Err: 0.260800  elapsed:  0.265748 hours, lr=0.00003874
[   11] Loss: 0.368914   Err: 0.088400  elapsed:  0.292027 hours, lr=0.00003487
[   12] Loss: 0.298227   Err: 0.075000  elapsed:  0.318126 hours, lr=0.00003138
[   13] Loss: 0.358107   Err: 0.100600  elapsed:  0.344276 hours, lr=0.00002824
[   14] Loss: 0.339421   Err: 0.082000  elapsed:  0.369695 hours, lr=0.00002542
[   15] Loss: 0.382928   Err: 0.099000  elapsed:  0.394635 hours, lr=0.00002288
[   16] Loss: 0.234183   Err: 0.069600  elapsed:  0.419311 hours, lr=0.00002059
[   17] Loss: 0.237683   Err: 0.070800  elapsed:  0.444754 hours, lr=0.00001853
[   18] Loss: 0.273791   Err: 0.085000  elapsed:  0.470706 hours, lr=0.00001668
[   19] Loss: 0.407833   Err: 0.062600  elapsed:  0.496747 hours, lr=0.00001501
[   20] Loss: 0.319596   Err: 0.069800  elapsed:  0.522825 hours, lr=0.00001351
[   21] Loss: 0.271080   Err: 0.065200  elapsed:  0.548738 hours, lr=0.00001216
[   22] Loss: 0.241903   Err: 0.060600  elapsed:  0.574896 hours, lr=0.00001094
[   23] Loss: 0.190998   Err: 0.060200  elapsed:  0.600700 hours, lr=0.00000985
[   24] Loss: 0.174027   Err: 0.058800  elapsed:  0.625549 hours, lr=0.00000886
[   25] Loss: 0.238272   Err: 0.065200  elapsed:  0.650361 hours, lr=0.00000798
[   26] Loss: 0.152129   Err: 0.050800  elapsed:  0.675327 hours, lr=0.00000718
[   27] Loss: 0.146603   Err: 0.054400  elapsed:  0.700166 hours, lr=0.00000646
[   28] Loss: 0.227547   Err: 0.047800  elapsed:  0.725068 hours, lr=0.00000581
[   29] Loss: 0.182535   Err: 0.050400  elapsed:  0.749994 hours, lr=0.00000523
[   30] Loss: 0.140753   Err: 0.043400  elapsed:  0.774781 hours, lr=0.00000471
[   31] Loss: 0.159581   Err: 0.045000  elapsed:  0.799692 hours, lr=0.00000424
[   32] Loss: 0.133405   Err: 0.045200  elapsed:  0.824695 hours, lr=0.00000382
[   33] Loss: 0.152682   Err: 0.045600  elapsed:  0.849539 hours, lr=0.00000343
[   34] Loss: 0.125633   Err: 0.042600  elapsed:  0.874443 hours, lr=0.00000309
[   35] Loss: 0.128318   Err: 0.042800  elapsed:  0.899121 hours, lr=0.00000278
[   36] Loss: 0.162865   Err: 0.040800  elapsed:  0.923788 hours, lr=0.00000250
[   37] Loss: 0.103261   Err: 0.041200  elapsed:  0.948497 hours, lr=0.00000225
[   38] Loss: 0.106623   Err: 0.040800  elapsed:  0.973385 hours, lr=0.00000203
[   39] Loss: 0.176227   Err: 0.036200  elapsed:  0.998314 hours, lr=0.00000182
[   40] Loss: 0.094803   Err: 0.039200  elapsed:  1.024723 hours, lr=0.00000164
[   41] Loss: 0.123611   Err: 0.039600  elapsed:  1.050885 hours, lr=0.00000148
[   42] Loss: 0.122700   Err: 0.033800  elapsed:  1.076998 hours, lr=0.00000133
[   43] Loss: 0.119774   Err: 0.037600  elapsed:  1.103236 hours, lr=0.00000120
[   44] Loss: 0.133163   Err: 0.040400  elapsed:  1.129316 hours, lr=0.00000108
[   45] Loss: 0.160640   Err: 0.045200  elapsed:  1.155636 hours, lr=0.00000097
[   46] Loss: 0.114480   Err: 0.032400  elapsed:  1.181919 hours, lr=0.00000087
[   47] Loss: 0.095019   Err: 0.034400  elapsed:  1.207704 hours, lr=0.00000079
[   48] Loss: 0.111298   Err: 0.033200  elapsed:  1.233838 hours, lr=0.00000071
[   49] Loss: 0.106354   Err: 0.035200  elapsed:  1.260076 hours, lr=0.00000064
[   50] Loss: 0.114164   Err: 0.036200  elapsed:  1.286173 hours, lr=0.00000057
[   51] Loss: 0.070675   Err: 0.034400  elapsed:  1.312571 hours, lr=0.00000052
[   52] Loss: 0.095508   Err: 0.036000  elapsed:  1.338761 hours, lr=0.00000046
[   53] Loss: 0.113600   Err: 0.031400  elapsed:  1.364691 hours, lr=0.00000042
[   54] Loss: 0.083376   Err: 0.035200  elapsed:  1.390994 hours, lr=0.00000038
[   55] Loss: 0.085253   Err: 0.037600  elapsed:  1.416442 hours, lr=0.00000034
[   56] Loss: 0.059367   Err: 0.029800  elapsed:  1.442435 hours, lr=0.00000030
[   57] Loss: 0.097093   Err: 0.033000  elapsed:  1.468703 hours, lr=0.00000027
[   58] Loss: 0.119242   Err: 0.039400  elapsed:  1.495025 hours, lr=0.00000025
[   59] Loss: 0.085457   Err: 0.032800  elapsed:  1.521355 hours, lr=0.00000022
[   60] Loss: 0.095210   Err: 0.036200  elapsed:  1.547571 hours, lr=0.00000020
[   61] Loss: 0.080815   Err: 0.033000  elapsed:  1.573766 hours, lr=0.00000018
[   62] Loss: 0.084177   Err: 0.038600  elapsed:  1.600004 hours, lr=0.00000016
[   63] Loss: 0.087985   Err: 0.032600  elapsed:  1.625203 hours, lr=0.00000015
[   64] Loss: 0.105078   Err: 0.037200  elapsed:  1.650140 hours, lr=0.00000013
[   65] Loss: 0.067318   Err: 0.033600  elapsed:  1.675065 hours, lr=0.00000012
[   66] Loss: 0.057762   Err: 0.033000  elapsed:  1.699900 hours, lr=0.00000011
[   67] Loss: 0.068458   Err: 0.037800  elapsed:  1.724965 hours, lr=0.00000010
[   68] Loss: 0.086128   Err: 0.035600  elapsed:  1.750480 hours, lr=0.00000009
[   69] Loss: 0.091967   Err: 0.033400  elapsed:  1.776557 hours, lr=0.00000008
[   70] Loss: 0.090020   Err: 0.039400  elapsed:  1.802642 hours, lr=0.00000007
[   71] Loss: 0.102898   Err: 0.038200  elapsed:  1.828833 hours, lr=0.00000006
[   72] Loss: 0.090425   Err: 0.035800  elapsed:  1.854919 hours, lr=0.00000006
[   73] Loss: 0.092566   Err: 0.034400  elapsed:  1.881227 hours, lr=0.00000005
[   74] Loss: 0.091237   Err: 0.034000  elapsed:  1.907272 hours, lr=0.00000005
[   75] Loss: 0.071890   Err: 0.031600  elapsed:  1.933318 hours, lr=0.00000004
[   76] Loss: 0.090209   Err: 0.030800  elapsed:  1.959562 hours, lr=0.00000004
[   77] Loss: 0.069922   Err: 0.033200  elapsed:  1.985649 hours, lr=0.00000003
[   78] Loss: 0.089059   Err: 0.033800  elapsed:  2.012150 hours, lr=0.00000003
[   79] Loss: 0.116966   Err: 0.034000  elapsed:  2.038593 hours, lr=0.00000003
[   80] Loss: 0.096969   Err: 0.032400  elapsed:  2.064903 hours, lr=0.00000002
</pre>

<hr>
Test:

<pre>
➜  argmaxnet git:(master) ✗ python3 test.py
ArgMaxNet(10,20,StackDepth(4),gamma=0.9),SGD(0.0001,0.9,weight_decay=0)
Parameter count = 1731
Testing 100 batches of 500 random scores ...
Loss: 0.086108   Err: 0.033420  elapsed: 0.000195 hours  perInstance: 0.014018 ms
</pre>






