'Layer' class -  to define the contnts of a particular layer in my model.
- weight_init initiaalization: with an option for 'Xavier', which helps in stopping gradients from exploding or vanishing.
- 'activation' function: defines the implementation of different activation functions(ReLU, softmax, sigmoid, tanh)
'grad_activation' function: defines the the gradient of the different activation functions.
- 'forward' function: implements a simple feedforward across the layer
- 'backward': implements backpropogation across layer to return gradient of previous pre-activation function


Neural Network class - to define the contents of my neural network model
- 'forward': returns output from given inputs using the forward function of layer
- 'backward': calculates the gradient of pre-activation function of all layers and the dw and db of first layer
- optmimiser functions like minibatch_sgd, momentum_gd, adma_gd are used.
    - gradient of weights along with learning rate and weight decay are used to vary the weights and biases towards the minimum loss
- 'train' function used to train the data with the help of different optimiser functions
- 'test' function to test the the calculated weights on test data and then calulate the accuracy with respect to y_test and calulate the loss value using cross entropy function defined below
