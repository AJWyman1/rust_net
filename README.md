# Multi-Threaded Neural Network
##### Written in Rust


### Alex Wyman
----

Final project for CIS-4710


Simple neural network created with custom data structures, which implement gradient descent and backpropagation. I also created single-threaded version to act as a benchmark to measure multi-threaded performance. Both networks can be customized to accept any number of inputs and produce any number of outputs as well as number of hidden layers and nodes in the hidden layers.

Both networks produce accuracies of 75-83% when trained on MNIST hand-drawn digit datasets with the multi-threaded version running 10% faster than the single-threaded version
