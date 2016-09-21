# Graph-Regularization
In this repository you will find the essentials Matlab scripts to simulating the algorithm proposed in the paper:

“On-line Learning On Temporal Manifolds” (Marco Maggini, Alessandro Rossi , 2016) 

The main function is [TRgraph.m](https://github.com/alered87/Graph-Regularization/blob/master/TRgraph.m) which define the class

makeSystemMatrix.m : compute the matrices of the dynamical system

PlotImpulsiveResponse.m : plot the Impulsive Response of the dynamical system, given the roots of the characteristic polynomial

TRgraph.m : define a matlab object for the implementation of the graph and the dynamic system to train the model proposed in the paper, data must be provided as a matrix containing row-wise an input sample with its target.
Quick start commands - once you define your data in a matrix Data of dimension number_of_samples-by-(input_size+classes): 
G = TRgraph('classes',classes);G.train(data,maxEpochs);
to calculate performance on a given testSet: [Accuracy,MSE] = G.test(testSet);

euclidean.m : compute the Euclidean distance between the elements of two matrix

MNISTtrainingSequence*.mat : files containing portion of the training sequence generated from the MNIST dataset [1] 

MNISTtest.mat : test set from MNIST [1]

MNISTvideo.avi : video of a little portion of the training sequence (40sec)


Contacts: 
Alessandro Rossi : rossi111@unisi.it

[1] see: http://yann.lecun.com/exdb/mnist/
