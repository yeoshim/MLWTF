import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))
# x_data = xy[0:-1]
# y_data = xy[-1];

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Deep NN
W1 = tf.Variable( tf.random_uniform([2,5], -1.0, 1.0) )
W2 = tf.Variable( tf.random_uniform([5,4], -1.0, 1.0) )
W3 = tf.Variable( tf.random_uniform([4,1], -1.0, 1.0) )
# W1 = tf.Variable( tf.random_uniform([2,2], -1.0, 1.0) )
# W2 = tf.Variable( tf.random_uniform([2,1], -1.0, 1.0) )

b1 = tf.Variable( tf.zeros([5]), name="Bias1" )
b2 = tf.Variable( tf.zeros([4]), name="Bias1" )
b3 = tf.Variable( tf.zeros([1]), name="Bias3" )

L2 = tf.sigmoid( tf.matmul(X, W1) + b1 )
L3 = tf.sigmoid( tf.matmul(L2, W2) + b2 )
hypothesis = tf.sigmoid( tf.matmul(L3, W3) + b3 )

# L2 = tf.sigmoid( tf.matmul(X, W1) + b1 )
# hypothesis = tf.sigmoid( tf.matmul(L2, W2) + b2 )

# Hypothesis based on sigmoid
# H(X) = 1 / 1 + e^(-WX)
# z = tf.matmul(W, X)
# hypothesis = tf.div( 1., 1. + tf.exp(-z) )

# Cost func based on Cross-entropy
# -1/m sigma( ylog(H(x)) + (1-y)log(1-H(x) )
cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis) )

a = tf.Variable( 0.1 )
# a = tf.Variable( 0.01 )
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize( cost )

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run( init )

#     for step in xrange(10000):    # accu: 1.0 => 1.0 => 1.0
#     for step in xrange(5000):    # accu: 0.75 => 1.0 => 0.75
#     for step in xrange(3000):    # accu: _ => 1.0 => 0.75
    for step in xrange(1000):    # accu: 0.5 => 0.75 => 0.5
        sess.run( train, feed_dict={X:x_data, Y:y_data} )
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)
    
    print '-------------------------------------------'
    
    # Test model
    correct_prediction = tf.equal( tf.floor(hypothesis+0.5), Y )    # 0/1
    
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, "float") )
    print sess.run( [hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data} )
    print "Accuracy: ", accuracy.eval( {X:x_data, Y:y_data} )
        