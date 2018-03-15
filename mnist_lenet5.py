import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])

x_image = tf.reshape(X, [-1, 28, 28, 1])

#卷积层1
with tf.name_scope('conv1'):
    with tf.name_scope('con1_filter'):
        #shape[filter_height, filter_width, in_channels, out_channels]
        con1_filter = tf.get_variable(name='filter', shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=1))
        tf.summary.histogram('con1_filter', con1_filter)
    with tf.name_scope('con1_bias'):
        con1_bias = tf.get_variable(name='bias', shape=[32], initializer=tf.zeros_initializer(0.0))
        tf.summary.histogram('con1_bias', con1_bias)
    with tf.name_scope('con1_2d'):
        con1_2d = tf.nn.conv2d(x_image, con1_filter, strides=[1, 1, 1, 1], padding='SAME')
        con1_relu = tf.nn.relu(tf.nn.bias_add(con1_2d, con1_bias))

#池化层1
with tf.name_scope('pooling1'):
    #ksize=[batch, height, width, channels]
    pool1 = tf.nn.max_pool(con1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#卷积层2
with tf.name_scope('conv2'):
    with tf.name_scope('con2_filter'):
        con2_filter = tf.get_variable(name='bias', shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=1))
        tf.summary.histogram('con2_filter', con2_filter)
    with tf.name_scope('con2_bias'):
        con2_b = tf.get_variable(name='bias', shape=[64], initializer=tf.zeros_initializer(0.0))
        tf.summary.histogram('con2_b', con2_b)
    with tf.name_scope('con2'):
        con2_2d = tf.nn.conv2d(pool1, con2_filter, strides=[1, 1, 1, 1], padding='SAME')
        con2_relu = tf.nn.relu(tf.nn.bias_add(con2_2d, con2_b))

#池化层2
with tf.name_scope('pooling2'):
    pool2 = tf.nn.max_pool(con2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#全连接层1
with tf.name_scope('dense'):
    den_w = tf.get_variable(name='weight', shape=[7*7*64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram('den1_weight', den_w)
    den_b = tf.get_variable(name='bias', shape=[1024], initializer=tf.constant_initializer(0.1))
    tf.summary.histogram('den1_bias', den_b)
    pool2 = tf.reshape(pool2, [-1, 7*7*64])
    den1 = tf.nn.relu(tf.add(tf.matmul(den_w, pool2), den_b))


