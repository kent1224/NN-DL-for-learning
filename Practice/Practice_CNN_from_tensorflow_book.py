# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:29:23 2017

@author: 14224
"""

import input_data
import tensorflow as tf

""" read data """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" parameters """
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

""" Network parameters """
n_input = 784
n_classes = 10

#為了減少過度配適(overfitting),應用Dropout(丟棄正則化技術)，指的是在神經網路中丟棄單元(隱藏、輸入和輸出)。
#決定消除哪些神經元是隨機的，一種方法是應用機率。
#probability to keep units
dropout = 0.75

""" input """
x = tf.placeholder(tf.float32, [None, n_input])
#28*28*1，-1代表不管輸入有幾個，Tensorflow 會自動調整的意思，假設輸入 50 個圖片第一個維度就是 50．
#這邊例子顏色是黑白灰階，因此值為1，若是RGB，則是3
_x = tf.reshape(x, shape = [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, n_classes])

""" Dropout (keep probability) """
#用於保持神經元輸出在丟棄期間的機率
keep_prob = tf.placeholder(tf.float32)

""" create model """
#tf.nn.conv2d從輸入tensor和共享權重計算出2D卷積，然後將該操作的結果加入到偏置矩陣bc1。
#relu(修正的線性單元)是深度神經網路隱藏層中常用的activation function
#填充值'SAME'用以指示輸出張量的輸出將與輸入張量的大小一致
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides = [1, 1, 1, 1], padding = 'SAME'), b))

#tf.nn.max_pool對輸入執行最大池化
def max_pool(img, k):
    return tf.nn.max_pool(img, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

""" store layers weight and bias (共享權重與共享偏差)"""
#第一層卷積層：32個特徵，5*5*1的filter
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))
#第二層卷積層：64個特徵，5*5*32的filter
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))
#Fully connected layer(全連結層)：inputs: 7*7*64, outputs: 1024 -用來處理整個圖像
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))
#output layer (class prediction)
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))

""" construct model """
#第一卷積層(first convolution layer)
conv1 = conv2d(_x, wc1, bc1)
#第一層卷積層的池化，簡化之前建立的卷積層的輸出資訊(down sampling)
conv1 = max_pool(conv1, k=2)
#第一層dropout
conv1 = tf.nn.dropout(conv1, keep_prob)

#第二卷積層(second convolution layer)
conv2 = conv2d(conv1, wc2, bc2)
#第二層卷積層的池化，簡化之前建立的卷積層的輸出資訊(down sampling)
conv2 = max_pool(conv2, k=2)
#第二層dropout
conv2 = tf.nn.dropout(conv2, keep_prob)

#全連結層(密集連接層，Fully connected layer)：計算和一般的NN一樣 ("flattening")
#Reshape conv2 output to fit dense layer input(assume 1024 outputs)
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
#計算 and activation function(assume relu)
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
#fully connected layer needs dropout as well
dense1 = tf.nn.dropout(dense1, keep_prob)

#output layer
pred = tf.add(tf.matmul(dense1, wout), bout)

""" Define loss and optimizer """
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

""" Evaluate model """
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred), tf.float32)

""" Session """
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    #keep training until reach max iteration
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        #fit training using training data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        
        if step % display_step == 0:
            #calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            
            #calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            
            print ("Iter " + str(step*batch_size) + ", Minibatch loss= " + "{:.6f}".format(loss) + ", Training accuracy= " + "{:.5f}".format(acc))
        
        step += 1
    
    print ("Optimization finished!")
    
    #calculate accuracy for 256 mnist test images
    print ("Testing accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
    