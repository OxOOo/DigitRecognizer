# encoding: utf-8

# 一层权链接+softmax

import os, sys
sys.path.append(os.path.curdir)
import data
import tensorflow as tf

x = tf.placeholder("float", [None, 28*28])
W = tf.Variable(tf.random_normal([28*28, 10], mean=0.0))
b = tf.Variable(tf.random_normal([10], mean=0.0))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y + 1e-10))

train = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print 'inited'
train_data = data.train_data()

print 'started'
for i in range(1000):
    sess.run(train, feed_dict={x: train_data['train_imgs'], y_: train_data['train_labels']})
    print 'B', sess.run(accuracy, feed_dict={x: train_data['test_imgs'], y_: train_data['test_labels']})