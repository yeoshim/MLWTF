import tensorflow as tf
import numpy as np

xy = np.loadtxt('train3.txt', unpack=True, dtype='float32')
x_data = np.transpose( xy[0:3] )
y_data = np.transpose( xy[3:] )

X = tf.placeholder( "float", [None, 3] )
Y = tf.placeholder( "float", [None, 3] )
W = tf.Variable( tf.zeros([3, 3]) )

hypothesis = tf.nn.softmax( tf.matmul(X, W) )
# hypothesis = tf.nn.softmax( tf.matmul(W, X) )    # error because of transpose
# Hypothesis based on sigmoid, H(X) = 1 / 1 + e^(-WX)
# z = tf.matmul(W, X)
# hypothesis = tf.div( 1., 1. + tf.exp(-z) )

# Cost func based on Cross-entropy
cost = tf.reduce_mean( -tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1) )
# -1/m sigma( ylog(H(x)) + (1-y)log(1-H(x) )
# cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis) )

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer( learning_rate ).minimize( cost )
# a = tf.Variable( 0.1 )
# optimizer = tf.train.GradientDescentOptimizer(a)
# train = optimizer.minimize( cost )

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run( init )

    for step in xrange(2001):
        sess.run( optimizer, feed_dict={X:x_data, Y:y_data} )
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

    print '-------------------------------------------'
    # bias, study_hour, attendance
    a = sess.run( hypothesis, feed_dict={X:[[1, 11, 7]]} )
    print a, sess.run( tf.arg_max(a, 1) )
    b = sess.run( hypothesis, feed_dict={X:[[1, 3, 4]]} )
    print b, sess.run( tf.arg_max(b, 1) )
    c = sess.run( hypothesis, feed_dict={X:[[1, 1, 0]]} )
    print c, sess.run( tf.arg_max(c, 1) )
    
    all = sess.run( hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]} )
    print all, sess.run( tf.arg_max(all, 1) )