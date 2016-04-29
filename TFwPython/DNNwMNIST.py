import tensorflow as tf
from numpy.random.mtrand import randint

import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
# learning_rate = 0.01
# training_epochs = 25
batch_size = 100
display_step = 1

x = tf.placeholder( "float", [None, 784] )
y = tf.placeholder( "float", [None, 10] )
# x = tf.placeholder( "float", [None, 784] )
# y = tf.placeholder( "float", [None, 10] )

W1 = tf.Variable( tf.random_normal([784, 256]) )
W2 = tf.Variable( tf.random_normal([256, 256]) )
W3 = tf.Variable( tf.random_normal([256, 10]) )
# W = tf.Variable( tf.zeros([784, 10]) )

b1 = tf.Variable( tf.random_normal([256]) )
b2 = tf.Variable( tf.random_normal([256]) )
b3 = tf.Variable( tf.random_normal([10]) )
# b = tf.Variable( tf.zeros([10]) )

# Hypothesis
L1 = tf.nn.relu( tf.add(tf.matmul(x, W1), b1) )
L2 = tf.nn.relu( tf.add(tf.matmul(L1, W2), b2) )
activation = tf.add( tf.matmul(L2, W3), b3 )    # w/o softmax
# activation = tf.nn.softmax( tf.matmul(x, W) + b )

# Cost func based on Cross-entropy
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(activation, y) ) # apply softmax
# cost = tf.reduce_mean( -tf.reduce_sum(y*tf.log(activation), reduction_indices=1) )

optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize( cost )
# optimizer = tf.train.GradientDescentOptimizer( learning_rate ).minimize( cost )

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range( training_epochs ):
        avg_cost = 0.
        total_batch = int( mnist.train.num_examples/batch_size)
        
        for i in range( total_batch ):
            batch_xs, batch_ys = mnist.train.next_batch( batch_size )
            sess.run( optimizer, feed_dict={x: batch_xs, y:batch_ys} )
            avg_cost += sess.run( cost, feed_dict={x: batch_xs, y:batch_ys} ) / total_batch
            
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
    print "Optimization Finished!"
    
    # Get one and predict
    r = randint( 0, mnist.test.num_examples -1 )
    print "Label: ", sess.run( tf.arg_max(mnist.test.labels[r:r+1], 1) )
    print "Prediction: ", sess.run( tf.arg_max(activation, 1), {x: mnist.test.images[r:r+1]} )
    
    # Test model
    correct_prediction = tf.equal( tf.arg_max(activation, 1), tf.arg_max(y, 1) )
    # Calculate accuracy
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, "float") )
    print "Accuracy:", accuracy.eval( {x: mnist.test.images, y: mnist.test.labels} )
    
    #Show the img
#     plt.imshow( mnist.test.images[r:r+1], reshape(28, 28), cmap="Greys", interpolation='nearest' )
#     plt.show()
