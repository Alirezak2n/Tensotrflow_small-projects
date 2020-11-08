import tensorflow as tf


# Constants
# Define a vector constant with 10 values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
c1 = tf.constant([0,1,2,3,4,5,6,7,8,9])

# and a vector constant with 10 values [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
c2 = tf.constant([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# Define 'add' tensor as sum of c1 and c2
add = tf.add(c1,c2)

# define 'multiply' tensor as element-wise multiplication of c1 and c2
multiply = tf.multiply(c1,c2)

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Run session
with tf.Session(config=config) as sess:
    c1_sess, c2_sess, add_sess, multiply_sess = sess.run([c1,c2,add.multipy])
    print('c1 = ' + str(c1_sess))
    print('c2 = ' + str(c2_sess))
    print('add = ' + str(add_sess))
    print('multiply = ' + str(multiply_sess))
