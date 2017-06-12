# encoding: utf-8

# CNN+softmax

import os, sys
sys.path.append(os.path.curdir)
import data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1]) # reshape

W_conv1 = weight_variable([5, 5, 1, 32]) # 第一层
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64]) # 第二层
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 4 * 64, 1024]) # 全连接1
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float") # dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10]) # softmax
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))

#train = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

print 'inited'
train_data = data.train_data()

BLOCK_SIZE = 400
BLOCK_NUM = len(train_data['train_imgs'])/BLOCK_SIZE

print 'started'
try:
    for index in range(50000):
        bindex = index%BLOCK_NUM
        imgs = train_data['train_imgs'][bindex*BLOCK_SIZE: bindex*BLOCK_SIZE+BLOCK_SIZE]
        labels = train_data['train_labels'][bindex*BLOCK_SIZE: bindex*BLOCK_SIZE+BLOCK_SIZE]
        sess.run(train, feed_dict={x: imgs, y_: labels, keep_prob: 0.5})
        if index % 100 == 0:
            print 'A', index, sess.run(accuracy, feed_dict={x: train_data['test_imgs'], y_: train_data['test_labels'], keep_prob: 1.0})
except KeyboardInterrupt:
    pass

print '\npredict'
test_data = data.test_data()
with open('cnn.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for i in range(0, len(test_data['imgs']), BLOCK_SIZE):
        print 'predict', i
        imgs = test_data['imgs'][i:i+BLOCK_SIZE]
        labels = sess.run(tf.argmax(y_conv, 1), feed_dict={x: imgs, keep_prob: 1.0})
        for j in range(len(imgs)):
            f.write('%d,%d\n' % (i+j+1, labels[j]))
