import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1];

# x_data = [[1, 1, 1, 1, 1],
#           [1., 0., 3., 0., 5.],
#           [0., 2., 0., 4., 0.]]
# y_data = [1., 2., 3., 4., 5.]

# W = tf.Variable( tf.random_uniform([1,len(x_data)], -5.0, 5.0) )
W = tf.Variable( tf.random_uniform([1,len(x_data)], -1.0, 1.0) )

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W, X)
# hypothesis = tf.matmul(W, X) + b
# hypothesis = W1 * X1 + W2 * X2 + b

cost = tf.reduce_mean( tf.square(hypothesis - Y) )

a = tf.Variable( 0.1 )
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize( cost )

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run( init )

for step in xrange(2001):
    sess.run( train, feed_dict={X:x_data, Y:y_data} )
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
# Learns best fit is W:[1,1], b:[0]
        