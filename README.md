# Graph-Regularization
MATLAB scripts for computing “On-line Learning On Temporal Manifolds”
Authors: Marco Maggini, Alessandro Rossi (2016) 
Contacts: rossi111@unisi.it


makeSystemMatrix.m : compute the matrices of the dynamical system

PlotImpulsiveResponse.m : plot the Impulsive Response of the dynamical system, given the roots of the characteristic polynomial

TRgraph.m : define a matlab object for the implementation of the graph and the dynamic system to train the model proposed in the paper, data must be provided as a matrix containing row-wise an input sample with its target.
Quick start commands - once you define your data in a matrix Data of dimension number_of_samples-by-(input_size+classes): 
G = TRgraph('classes',classes);G.train(data,maxEpochs);
to calculate performance on a given testSet: [Accuracy,MSE] = G.test(testSet);

euclidean.m : compute the Euclidean distance between the elements of two matrix

MNISTtrainingSequence*.mat : files containing portion of the training sequence generated from the MNIST(1) dataset 
(1) see: http://yann.lecun.com/exdb/mnist/ dataset

MNISTtest.mat : test set from MNIST
