# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:36:11 2017

@author: 14224
"""

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function = None):
    
    """ Weights and biases """
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([out_size]))          
    
    """ Multilayer perception """
    # Hidden layer, output layer: xw+b
    # Don't forget that order matters in matrix multiplication, so tf.matmul(a,b) is not the same as tf.matmul(b,a).
    layer = tf.add(tf.matmul(inputs, weights), biases)
    
    if activation_function is None:
        output = layer
    else:
        output = activation_function(layer)
    
    return output

"""Mini-batching is a technique for training on subsets of the dataset 
   instead of all the data at one time. This provides the ability to train a model, 
   even if a computer lacks the memory to store the entire dataset.
   Mini-batching is computationally inefficient, 
   since you can't calculate the loss simultaneously across all samples. 
   However, this is a small price to pay in order to be able to run the model at all.
   It's also quite useful combined with SGD. 
   The idea is to randomly shuffle the data at the start of each epoch, 
   then create the mini-batches. 
   For each mini-batch, you train the network weights with gradient descent. 
   Since these batches are random, you're performing SGD with each batch."""

""" Define Parameters (training_epochs, learning_rate, batch_size, display_step, num_neurons, etc.)"""
# Learning parameters
    # learning rate for GradientDescentOpitimizer
learning_rate = 0.001
    # one epoch = one forward pass and one backward pass of all the training examples
    # epoch: 學習周期，透過選擇一組訓練集，來稍微修改突觸的加權值
    # An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data.
training_epochs = 20 
    # batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
batch_size = 128 
display_step = 1 # ??

layer_1_n_input = 784 # e.g. MNIST data input (img shape: 28*28)
n_classes = 10 # e.g. MNIST total classes (0-9 digits)

# Hidden layer parameters
layer_1_n_hidden_layer = 256 # layer number of features, determines the sizes of the hidden layer in the neural network. This is also known as the width of a layer.
layer_2_n_hidden_layer = 256

""" Input """
# Input
#placeholder: 預留位，allows us to create our operations and build our computation graph, without needing the data，之後在session用feed_dict餵進去
x = tf.placeholder("float",[None,28,28,1])
y = tf.placeholder("float",[None,n_classes])

# Reshape
x_flat = tf.reshape(x, [-1, layer_1_n_input])

""" Create NN graph """
# Hidden layer
# in_size = n_input, out_size = n_hidden_layer
# logistic regression 只有一層，activation function: softmax (將輸入轉換為機率形式的輸出)
layer_1 = add_layer(x_flat, layer_1_n_input, layer_1_n_hidden_layer, activation_function = tf.nn.softmax)
layer_2 = add_layer(layer_1, layer_1_n_hidden_layer, layer_2_n_hidden_layer, activation_function = tf.nn.relu)

# Output layer
# in_size = n_hidden_layer, out_size = n_classes
prediction = add_layer(layer_2, layer_2_n_hidden_layer, n_classes, activation_function = None)


""" Define loss and optimizer """
#cost function: mean squared error, squared euclidean distance, cross-entropy, ...
# Build the loss rule
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
# cross entropy for cost function


# Choose the learning mechanism and minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)


""" ################## Start to train #################"""
""" Session """
#情節設定(plot setting)

# Initialize the variables
init = tf.global_variables_initializer()
    # or init = tf.initialize_all_variables()

# Build the sess and initialize it
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    #Training cycle
    for epoch in range(training_epoch):
        
    
    #最後還是要print出來
    #每一個epoch，印出相對的成本函數並視覺化
    #印出模型準確率



