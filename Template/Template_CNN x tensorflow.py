# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:37:38 2017

@author: 14224
"""


import input_data
import tensorflow as tf

""" read data """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" parameters """
learning_rate = 0.001
training_epoch = 10
batch_size = 128
display_step = 10

""" Number of samples to calculate validation and accuracy """
test_valid_size = 256 # Decrease this if you're running out of memory to calculate accuracy

""" Network parameters (include dropout)"""
n_input = 784
n_classes = 10

""" input """
x = tf.placeholder(tf.float32, [None, n_input])
#28*28*1，-1代表不管輸入有幾個，Tensorflow 會自動調整的意思，假設輸入 50 個圖片第一個維度就是 50．
#這邊例子顏色是黑白灰階，因此值為1，若是RGB，則是3
_x = tf.reshape(x, shape = [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, n_classes])

""" Dropout (keep probability) """
#為了減少過度配適(overfitting),應用Dropout(丟棄正則化技術)，指的是在神經網路中丟棄單元(隱藏、輸入和輸出)。
#決定消除哪些神經元是隨機的，一種方法是應用機率。
#probability to keep units
dropout = 0.75
#用於保持神經元輸出在丟棄期間的機率
keep_prob = tf.placeholder(tf.float32)

""" create model: convolutions and max_pooling """
#tf.nn.conv2d從輸入tensor和共享權重計算出2D卷積，然後將該操作的結果加入到偏置矩陣bc1。
#relu(修正的線性單元)是深度神經網路隱藏層中常用的activation function
#填充值'SAME'用以指示輸出張量的輸出將與輸入張量的大小一致
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides = [1, 1, 1, 1], padding = 'SAME'), b))

#tf.nn.max_pool對輸入執行最大池化: for decreasing the size of the output, and preventing overfitting
"""
Conceptually, the benefit of the max pooling operation is to reduce the size of the input, 
and allow the neural network to focus on only the most important elements. 
Max pooling does this by only retaining the maximum value for each filtered area, 
and removing the remaining values.
"""
def maxpool2d(img, k):
    return tf.nn.max_pool(img, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

""" store layers weight and bias (共享權重與共享偏差)"""
#第一層卷積層：32個特徵，5*5*1的filter
#第二層卷積層：64個特徵，5*5*32的filter
#Fully connected layer(全連結層)：inputs: 7*7*64, outputs: 1024 -用來處理整個圖像
#output layer (class prediction)
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wout': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}

""" construct model """
def conv_net(x, weights, biases, keep_prob):
    #第一卷積層(first convolution layer): 28*28*1 to 28*28*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #第一層卷積層的池化，簡化之前建立的卷積層的輸出資訊(down sampling): 28*28*32 to 14*14*32
    conv1 = maxpool2d(conv1, k=2)
    #第一層dropout
    conv1 = tf.nn.dropout(conv1, keep_prob)

    #第二卷積層(first convolution layer): 14*14*32 to 14*14*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #第二層卷積層的池化，簡化之前建立的卷積層的輸出資訊(down sampling): 14*14*64 to 7*7*64
    conv2 = maxpool2d(conv2, k=2)
    #第二層dropout
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Fully connected layer - 7*7*64 to 1024
    #全連結層(密集連接層，Fully connected layer)：計算和一般的NN一樣
    #Reshape conv2 output to fit dense layer input(assume 1024 outputs)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #計算 and activation function(assume relu)
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']))
    #fully connected layer needs dropout as well
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output Layer - class prediction - 1024 to 10
    pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return pred

""" model """
pred = conv_net(x, weights, biases, keep_prob)

""" Define cost function and optimizer """
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

""" Evaluate model """
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred), tf.float32)

""" Session """
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init) 
    
    #Training cycle
    for epoch in range(training_epoch):
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # loop over all batches
        for batch in range(total_batch):
            # split training data into x and y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            # run optimizer (backprop): 在訓練時使用批次資料 (fit training using batch data)        
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images[:test_valid_size], y: mnist.validation.labels[:test_valid_size], keep_prob: 1.})
            
            #display logs per epoch step per batch
            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
    

""" example on tensorflow book
    
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
""" 