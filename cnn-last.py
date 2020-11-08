from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import MNIST data
mnist = input_data.read_data_sets("../1-dnn/", one_hot=True)

# Training Parameters
num_steps = 2000
batch_size = 128
display_step = 10
strides = 1
k = 2

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
learning_rate = 0.001

# tf Graph input
X = tf.placeholder(tf.float32, name='samples', shape=[None, 784])
Y = tf.placeholder(tf.int32, name='labels', shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#
#  Store layers weight & bias
# The first three convolutional layer
w_c_1 = tf.Variable(tf.random_normal([5, 5, 1, 512]))
w_c_2 = tf.Variable(tf.random_normal([5, 5, 512, 512]))
w_c_3 = tf.Variable(tf.random_normal([5, 5, 512, 512]))
b_c_1 = tf.Variable(tf.random_normal([512]))
b_c_2 = tf.Variable(tf.random_normal([512]))
b_c_3 = tf.Variable(tf.random_normal([512]))

# The second three convolutional layer weights
w_c_4 = tf.Variable(tf.random_normal([5, 5, 512, 512]))
w_c_5 = tf.Variable(tf.random_normal([5, 5, 512, 512]))
w_c_6 = tf.Variable(tf.random_normal([5, 5, 512, 512]))
b_c_4 = tf.Variable(tf.random_normal([512]))
b_c_5 = tf.Variable(tf.random_normal([512]))
b_c_6 = tf.Variable(tf.random_normal([512]))

# Fully connected weight
w_f_1 = tf.get_variable(name='wf1', shape=[7 * 7 * 512, 2048], initializer=tf.random_normal_initializer())
w_f_2 = tf.get_variable(name='wf2', shape=[2048, 4096], initializer=tf.random_normal_initializer())
w_f_3 = tf.get_variable(name='wf3', shape=[4096, 4096], initializer=tf.random_normal_initializer())
b_f_1 = tf.get_variable(name='bf1', shape=[1, 2048], initializer=tf.zeros_initializer())
b_f_2 = tf.get_variable(name='bf2', shape=[1, 4096], initializer=tf.zeros_initializer())
b_f_3 = tf.get_variable(name='bf3', shape=[1, 4096], initializer=tf.zeros_initializer())

# output layer weight
w_out = tf.get_variable(name='w_out', shape=[4096, 10])
b_out = tf.get_variable(name='b_out', shape=[1, 10])

#
# Define model
x = tf.reshape(X, [-1, 28, 28, 1])
# first layer convolution
conv1 = tf.nn.conv2d(x, w_c_1, strides=[1,1,1,1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, b_c_1)
conv1 = tf.nn.relu(conv1)

# second layer convolution
conv2 = tf.nn.conv2d(conv1, w_c_2, strides=[1,1,1,1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_c_2)
conv2 = tf.nn.relu(conv2)

# third layer convolution
conv3 = tf.nn.conv2d(conv2, w_c_3, strides=[1,1,1,1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, b_c_3)
conv3 = tf.nn.relu(conv3)

# first Max Pooling (down-sampling)
pool_1 = tf.nn.max_pool(conv3, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME')

# fourth layer convolution
conv4 = tf.nn.conv2d(pool_1, w_c_4, strides=[1,1,1,1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, b_c_4)
conv4 = tf.nn.relu(conv4)

# fifth layer convolution
conv5 = tf.nn.conv2d(conv4, w_c_5, strides=[1,1,1,1], padding='SAME')
conv5 = tf.nn.bias_add(conv5, b_c_5)
conv5 = tf.nn.relu(conv5)


# sixth layer convolution
conv6 = tf.nn.conv2d(conv5, w_c_6, strides=[1,1,1,1], padding='SAME')
conv6 = tf.nn.bias_add(conv6, b_c_6)
conv6 = tf.nn.relu(conv6)

# second Max Pooling (down-sampling)
pool_2 = tf.nn.max_pool(conv6, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME')

# first Fully connected layer
# Reshape pool_2 output to fit fully connected layer input
fc1 = tf.reshape(pool_2, [-1, w_f_1.get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, w_f_1), b_f_1)
fc1 = tf.nn.relu(fc1)
# Apply Dropout
fc1 = tf.nn.dropout(fc1, dropout)

# second Fully connected layer
fc2 = tf.add(tf.matmul(fc1, w_f_2), b_f_2)
fc2 = tf.nn.relu(fc2)

# Third Fully connected layer
fc3 = tf.add(tf.matmul(fc2, w_f_3), b_f_3)
fc3 = tf.nn.relu(fc3)

# Output, class prediction
logits = tf.add(tf.matmul(fc3, w_out), b_out)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Start training
with tf.Session(config=config) as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict= {X:batch_x, Y:batch_y, keep_prob:1})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
