# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:36:57 2017

@author: 14224
"""

import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

""" read data """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" Define parameters """
training_epochs = 20
learning_rate = 0.001
batch_size = 100
display_step = 1

""" Network Parameters """
#第一、二層的神經元數量、輸入尺寸、輸出類別
#定義隱藏層的數量以及每一層神經元的數量並沒有嚴格的標準，每次選擇都是基於「過往類似應用的經驗」
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

""" input """
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

""" create model """
#不同神經元的函數可以是不同的，但在實踐中，我們對於所有的神經元，採用的共同特徵通常是sigmoid類型的函數
h = tf.Variable(tf.random_normal(n_input, n_hidden_1))
bias_layer_1 = tf.Variable(tf.random_normal(n_hidden_1))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h), bias_layer_1))

w = tf.Variable(tf.random_normal(n_hidden_1, n_hidden_2))
bias_layer_2 = tf.Variable(tf.random_normal(n_hidden_2))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w), bias_layer_2))

output = tf.Variable(tf.random_normal(n_hidden_2, n_classes))
bias_output = tf.Variable(tf.random_normal(n_classes))
output_layer = tf.add(tf.matmul(layer_2, output), bias_output)

""" cost function and optimizer """
#tf.nn.softmax_cross_entropy_with_logits是計算softmax層的成本，它只會在訓練期間使用
#logits是輸出模型的非標準化對數機率
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

#tf.train.AdamOptimizer是用Kingma及Ba's Adam演算法來控制學習速率
#相對於簡單的tf.train.GradientDescentOptimizer，Adam使用更大而更有效率的step大小，並且演算法將會在沒有微調的情況下收斂到該step大小
#一個簡單的tf.train.GradientDescentOptimizer同樣也可以應用在你的MLP當中，但需要調整更多的超參數，才能夠快速收斂
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

""" session """
#情節設定(plot setting)
avg_set = []
epoch_set = []

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    #定義訓練週期
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        #對於所有的批次執行迴圈
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #fit training using batch data
            sess.run(optimizer, feed_dixt={x: batch_xs, y: batch_ys})
            
            #compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        
        #Display log per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost:", "{:,.9f}").format(avg_cost)
        
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print ("Training phase finished")
    
    #畫圖，輸出訓練階段
    plt.plot(epoch_set, avg_set, 'o', label = 'MLP training phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #test model
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y,1))
    
    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
            
            
            
        
        
    

