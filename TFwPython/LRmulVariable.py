import tensorflow as tf

x1_data = [1., 0., 3., 0., 5.]
x2_data = [0., 2., 0., 4., 0.]
y_data = [1., 2., 3., 4., 5.]

W1 = tf.Variable( tf.random_uniform([1], -1.0, 1.0) )
W2 = tf.Variable( tf.random_uniform([1], -1.0, 1.0) )
b = tf.Variable( tf.random_uniform([1], -1.0, 1.0) )

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W1 * X1 + W2 * X2 + b
#hypothesis = W * x_data + b

cost = tf.reduce_mean( tf.square(hypothesis - y_data) )

a = tf.Variable( 0.1 )
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize( cost )

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run( init )

for step in xrange(2001):
    sess.run( train, feed_dict={X1:x1_data, X2:x2_data, Y:y_data} )
#    sess.run( train )
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X1:x1_data, X2:x2_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(b)
#         print step, sess.run(cost), sess.run(W), sess.run(b)
        