import tensorflow as tf

# Define 'x1' tensor as a placeholder
x1 = tf.placeholder(tf.float32,shape=[2,2],name='x1')

# Define 'x2' tensor as a placeholder
x2 = tf.placeholder(tf.float32,shape=[2,2],name='x2')

# Define 'subtract' tensor to subtract x1, x2
subtract = tf.subtract(x1,x2)

# Define 'pow' tensor to element-wise power x1, x2
pow = tf.pow(x1,x2)

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Run session
with tf.Session(config=config) as sess:
    # Run session and feed x1:[[2,2],[3,3]] , x2:[[3,3], [2,2]]
     substract_sess=sess.run(substract , feed_dict={x1:[[2,2],[3,3]] , x2:[[3,3], [2,2]]}
    print('subtract = ' + str(subtract_sess))
    print('')

    # Run another session and feed x1:[[2,2],[3,3]] , x2:[[1,3], [2,4]]
    pow_sess=sess.run(pow , feed_dict={x1:[[2,2],[3,3]] , x2:[[1,3], [2,4]]}
    print('pow = ' + str(pow_sess))
