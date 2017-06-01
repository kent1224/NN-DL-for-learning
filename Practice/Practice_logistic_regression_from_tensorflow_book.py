# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:08:16 2017

@author: 14224
"""

import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

""" read data """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" Define parameters """
training_epochs = 25
learning_rate = 0.01
batch_size = 100
display_step = 1

""" Inputs """
#輸入張量：28*28=784個像素點
x = tf.placeholder("float", [None, 784])
#輸出張量：10個可能類別與其機率
y = tf.placeholder("float", [None, 10])

""" Build model """
#activation function: softmax function to compute probabilities
#softmax函數會在兩個步驟中被使用
#1.計算出足以讓某一個圖像歸屬於特定類別的憑據(evidence)
#2.轉換憑據成為10個可能類別的歸屬機率(probabilities)

#權重輸入張量
w = tf.Variable(tf.zeros([784,10]))
#偏差張量
b = tf.Variable(tf.zeros([10]))
#evidence
evidence = tf.matmul(x,w) + b
#activation function
activation = tf.nn.softmax(evidence)

""" Define cost function and optimizer """
#cost function: mean squared error, squared euclidean distance, cross-entropy, ...
cross_entropy = y*tf.lg(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

""" Session """
#plot setting
avg_set = []
epoch_set = []
#initialize the variables
init = tf.initialize_all_variables()
#launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        #loop over all batched
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #在訓練時使用批次資料 (fit training using batch data)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            #compute average cost
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        
        #display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
        
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print ("Training phase finished.")
    
    #畫圖出來
    plt.plot(epoch_set, avg_set, 'o', label = 'Logistic Regression Training Phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #Test model
    #correct_prediction 的平均值將會提供給我們準確性
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    
    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Model Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))