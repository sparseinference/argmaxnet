# argmaxnet  

An experiment in training a fully connected residual net to learn the argmax function.  
This seems surprisingly difficult for such a simple problem.

<hr>

Experiment:  

- Python 3.7
- PyTorch 0.5.0a0+f3ac619
- Stacked Residual Net with identical residual modules followed by a linear reduction to a single output
- The residual modules are wide in the hidden layer with the input and output dimensions equal

<hr>
Training:

<pre>
➜  argmaxnet git:(master) ✗ python3 train.py
ArgMaxNet : StackDepth=3, Block(10,60)
            opt: SGD(lr=0.0001 reduced by 0.98 each episode, momentum=0.9, weight_decay=0)
Parameter count = 3821
[    1] Loss: 0.841379   Err: 0.210000  elapsed:  0.009590 hours, lr=0.00010000
[    2] Loss: 1.009749   Err: 0.158800  elapsed:  0.019381 hours, lr=0.00009800
[    3] Loss: 0.673255   Err: 0.143200  elapsed:  0.029359 hours, lr=0.00009604
[    4] Loss: 0.400579   Err: 0.085800  elapsed:  0.039462 hours, lr=0.00009412
[    5] Loss: 0.522911   Err: 0.093000  elapsed:  0.049320 hours, lr=0.00009224
[    6] Loss: 0.390506   Err: 0.077000  elapsed:  0.059192 hours, lr=0.00009039
[    7] Loss: 0.352900   Err: 0.068200  elapsed:  0.069052 hours, lr=0.00008858
[    8] Loss: 0.293024   Err: 0.069600  elapsed:  0.078911 hours, lr=0.00008681
[    9] Loss: 0.308593   Err: 0.073800  elapsed:  0.088776 hours, lr=0.00008508
[   10] Loss: 0.485971   Err: 0.069800  elapsed:  0.098606 hours, lr=0.00008337
[   11] Loss: 0.503779   Err: 0.065200  elapsed:  0.108481 hours, lr=0.00008171
[   12] Loss: 0.277030   Err: 0.060000  elapsed:  0.118357 hours, lr=0.00008007
[   13] Loss: 0.223894   Err: 0.055800  elapsed:  0.128042 hours, lr=0.00007847
[   14] Loss: 0.266146   Err: 0.059600  elapsed:  0.137845 hours, lr=0.00007690
[   15] Loss: 0.316010   Err: 0.071800  elapsed:  0.147688 hours, lr=0.00007536
[   16] Loss: 0.285165   Err: 0.050400  elapsed:  0.157558 hours, lr=0.00007386
[   17] Loss: 0.392230   Err: 0.070600  elapsed:  0.167250 hours, lr=0.00007238
[   18] Loss: 0.246672   Err: 0.051400  elapsed:  0.176823 hours, lr=0.00007093
[   19] Loss: 0.209529   Err: 0.047200  elapsed:  0.186649 hours, lr=0.00006951
[   20] Loss: 0.183141   Err: 0.038200  elapsed:  0.196483 hours, lr=0.00006812
[   21] Loss: 0.212652   Err: 0.046800  elapsed:  0.206334 hours, lr=0.00006676
[   22] Loss: 0.276476   Err: 0.050800  elapsed:  0.216203 hours, lr=0.00006543
[   23] Loss: 0.289849   Err: 0.050800  elapsed:  0.225948 hours, lr=0.00006412
[   24] Loss: 0.275776   Err: 0.051600  elapsed:  0.235879 hours, lr=0.00006283
[   25] Loss: 0.200512   Err: 0.044600  elapsed:  0.245600 hours, lr=0.00006158
[   26] Loss: 0.194553   Err: 0.041800  elapsed:  0.255548 hours, lr=0.00006035
[   27] Loss: 0.169269   Err: 0.045200  elapsed:  0.265598 hours, lr=0.00005914
[   28] Loss: 0.273164   Err: 0.045200  elapsed:  0.275583 hours, lr=0.00005796
[   29] Loss: 0.272778   Err: 0.127400  elapsed:  0.285362 hours, lr=0.00005680
[   30] Loss: 0.174689   Err: 0.050200  elapsed:  0.295156 hours, lr=0.00005566
[   31] Loss: 0.285349   Err: 0.055200  elapsed:  0.304912 hours, lr=0.00005455
[   32] Loss: 0.300190   Err: 0.041400  elapsed:  0.314661 hours, lr=0.00005346
[   33] Loss: 0.202204   Err: 0.045000  elapsed:  0.324503 hours, lr=0.00005239
[   34] Loss: 0.198700   Err: 0.040200  elapsed:  0.334367 hours, lr=0.00005134
[   35] Loss: 0.202726   Err: 0.038800  elapsed:  0.344198 hours, lr=0.00005031
[   36] Loss: 0.227932   Err: 0.045600  elapsed:  0.354049 hours, lr=0.00004931
[   37] Loss: 0.168154   Err: 0.041200  elapsed:  0.363848 hours, lr=0.00004832
[   38] Loss: 0.244279   Err: 0.043800  elapsed:  0.373667 hours, lr=0.00004735
[   39] Loss: 0.192497   Err: 0.039800  elapsed:  0.383468 hours, lr=0.00004641
[   40] Loss: 0.175515   Err: 0.040000  elapsed:  0.393374 hours, lr=0.00004548
[   41] Loss: 0.132442   Err: 0.036400  elapsed:  0.403231 hours, lr=0.00004457
[   42] Loss: 0.131879   Err: 0.032400  elapsed:  0.413111 hours, lr=0.00004368
[   43] Loss: 0.239974   Err: 0.036400  elapsed:  0.422981 hours, lr=0.00004281
[   44] Loss: 0.146880   Err: 0.034800  elapsed:  0.432853 hours, lr=0.00004195
[   45] Loss: 0.116312   Err: 0.033400  elapsed:  0.442684 hours, lr=0.00004111
[   46] Loss: 0.214501   Err: 0.035400  elapsed:  0.452548 hours, lr=0.00004029
[   47] Loss: 0.277160   Err: 0.033400  elapsed:  0.462417 hours, lr=0.00003948
[   48] Loss: 0.178411   Err: 0.034000  elapsed:  0.472273 hours, lr=0.00003869
[   49] Loss: 0.246785   Err: 0.039400  elapsed:  0.482137 hours, lr=0.00003792
[   50] Loss: 0.231782   Err: 0.038800  elapsed:  0.492130 hours, lr=0.00003716
[   51] Loss: 0.157200   Err: 0.036600  elapsed:  0.502017 hours, lr=0.00003642
[   52] Loss: 0.259224   Err: 0.035000  elapsed:  0.511886 hours, lr=0.00003569
[   53] Loss: 0.262706   Err: 0.042200  elapsed:  0.521773 hours, lr=0.00003497
[   54] Loss: 0.169639   Err: 0.029600  elapsed:  0.531676 hours, lr=0.00003428
[   55] Loss: 0.280070   Err: 0.039000  elapsed:  0.541571 hours, lr=0.00003359
[   56] Loss: 0.151866   Err: 0.029200  elapsed:  0.551473 hours, lr=0.00003292
[   57] Loss: 0.201496   Err: 0.038000  elapsed:  0.561332 hours, lr=0.00003226
[   58] Loss: 0.125637   Err: 0.031600  elapsed:  0.571196 hours, lr=0.00003161
[   59] Loss: 0.218838   Err: 0.032000  elapsed:  0.581075 hours, lr=0.00003098
[   60] Loss: 0.119034   Err: 0.028800  elapsed:  0.590925 hours, lr=0.00003036
[   61] Loss: 0.087370   Err: 0.027000  elapsed:  0.600750 hours, lr=0.00002976
[   62] Loss: 0.105404   Err: 0.029600  elapsed:  0.610550 hours, lr=0.00002916
[   63] Loss: 0.202022   Err: 0.026000  elapsed:  0.620337 hours, lr=0.00002858
[   64] Loss: 0.114809   Err: 0.028200  elapsed:  0.630182 hours, lr=0.00002801
[   65] Loss: 0.156818   Err: 0.026400  elapsed:  0.639974 hours, lr=0.00002745
[   66] Loss: 0.117886   Err: 0.027400  elapsed:  0.649795 hours, lr=0.00002690
[   67] Loss: 0.148011   Err: 0.027200  elapsed:  0.659691 hours, lr=0.00002636
[   68] Loss: 0.185504   Err: 0.032400  elapsed:  0.669347 hours, lr=0.00002583
[   69] Loss: 0.158207   Err: 0.026200  elapsed:  0.678770 hours, lr=0.00002531
[   70] Loss: 0.211314   Err: 0.036000  elapsed:  0.688219 hours, lr=0.00002481
[   71] Loss: 0.123811   Err: 0.029000  elapsed:  0.697775 hours, lr=0.00002431
[   72] Loss: 0.121323   Err: 0.024200  elapsed:  0.707256 hours, lr=0.00002383
[   73] Loss: 0.092815   Err: 0.025200  elapsed:  0.716782 hours, lr=0.00002335
[   74] Loss: 0.077281   Err: 0.025600  elapsed:  0.726131 hours, lr=0.00002288
[   75] Loss: 0.127059   Err: 0.026400  elapsed:  0.735463 hours, lr=0.00002242
[   76] Loss: 0.135494   Err: 0.023200  elapsed:  0.744721 hours, lr=0.00002198
[   77] Loss: 0.095448   Err: 0.023400  elapsed:  0.753993 hours, lr=0.00002154
[   78] Loss: 0.137425   Err: 0.027600  elapsed:  0.763382 hours, lr=0.00002111
[   79] Loss: 0.105115   Err: 0.023200  elapsed:  0.772932 hours, lr=0.00002068
[   80] Loss: 0.135414   Err: 0.024000  elapsed:  0.782717 hours, lr=0.00002027
[   81] Loss: 0.146738   Err: 0.027800  elapsed:  0.792537 hours, lr=0.00001986
[   82] Loss: 0.109915   Err: 0.023200  elapsed:  0.802295 hours, lr=0.00001947
[   83] Loss: 0.100822   Err: 0.023200  elapsed:  0.812065 hours, lr=0.00001908
[   84] Loss: 0.129659   Err: 0.024000  elapsed:  0.821850 hours, lr=0.00001870
[   85] Loss: 0.117406   Err: 0.026800  elapsed:  0.831664 hours, lr=0.00001832
[   86] Loss: 0.097301   Err: 0.024600  elapsed:  0.841496 hours, lr=0.00001796
[   87] Loss: 0.088146   Err: 0.020000  elapsed:  0.851277 hours, lr=0.00001760
[   88] Loss: 0.108722   Err: 0.020400  elapsed:  0.861119 hours, lr=0.00001725
[   89] Loss: 0.105350   Err: 0.024800  elapsed:  0.870881 hours, lr=0.00001690
[   90] Loss: 0.140298   Err: 0.023400  elapsed:  0.880625 hours, lr=0.00001656
[   91] Loss: 0.081200   Err: 0.021400  elapsed:  0.890413 hours, lr=0.00001623
[   92] Loss: 0.114371   Err: 0.021400  elapsed:  0.900247 hours, lr=0.00001591
[   93] Loss: 0.206781   Err: 0.024200  elapsed:  0.910042 hours, lr=0.00001559
[   94] Loss: 0.171771   Err: 0.022200  elapsed:  0.919865 hours, lr=0.00001528
[   95] Loss: 0.120925   Err: 0.022200  elapsed:  0.929604 hours, lr=0.00001497
[   96] Loss: 0.087923   Err: 0.022400  elapsed:  0.939338 hours, lr=0.00001467
[   97] Loss: 0.081036   Err: 0.023000  elapsed:  0.949043 hours, lr=0.00001438
[   98] Loss: 0.105851   Err: 0.024000  elapsed:  0.958831 hours, lr=0.00001409
[   99] Loss: 0.116960   Err: 0.026200  elapsed:  0.968549 hours, lr=0.00001381
[  100] Loss: 0.065967   Err: 0.017200  elapsed:  0.978295 hours, lr=0.00001353
</pre>

Test:

<pre>
➜  argmaxnet git:(master) ✗ python3 test.py
Testing 100 batches of 500 random scores ...
Loss: 0.103573   Err: 0.018920  elapsed: 0.001009 hours  perInstance: 0.072660 ms
</pre>