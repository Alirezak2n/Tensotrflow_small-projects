import tensorflow as tf

# Define 'w1' variable
w1 = tf.get_variable("w1", initializer=tf.constant(2))

# Define 'w2' variable
w2 = tf.get_variable("w2")

# Define 'w3' variable
w3 = tf.get_variable("w3")

# Initializer
init_op = tf.global_variable_initializer()

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Run session
with tf.Session(config=config) as sess:
    sess.run(init_op)
	w1_sess,w2_sess,w3_sess= sess.run([w1,w2,w3])
    print('w1 = ' + str(w1_sess))
    print('')
    print('w2 = ' + str(w2_sess))
    print('')
    print('w3 = ' + str(w3_sess))
    print('')
