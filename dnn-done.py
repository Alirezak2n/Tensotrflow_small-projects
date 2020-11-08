from _sha3 import sha3_224

from tensorflow.contrib import labeled_tensor
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import MNIST data
mnist = input_data.read_data_sets(".", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 150
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_hidden_3 = 256  # 3nd layer number of neurons
n_input = 784  # MNIST data input_features (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input_features
X = tf.placeholder(tf.float32, shape=[None, 784], name='samples')
Y = tf.placeholder(tf.int32, shape=[None, 10], name='labels')

# Store layers weight & bias
weights = {
    'h1': tf.get_variable(name='W1', shape=[784, 256], initializer=tf.random_normal_initializer()),
    'h2': tf.get_variable(name='W2', shape=[256, 256], initializer=tf.random_normal_initializer()),
    'h3': tf.get_variable(name='W3', shape=[256, 256], initializer=tf.random_normal_initializer()),
    'out': tf.get_variable(name='W_out', shape=[256, 10], initializer=tf.random_normal_initializer())
}
biases = {
    'b1': tf.get_variable(name='b1', shape=[1, 256], initializer=tf.zeros_initializer()),
    'b2': tf.get_variable(name='b2', shape=[1, 256], initializer=tf.zeros_initializer()),
    'b3': tf.get_variable(name='b3', shape=[1, 256], initializer=tf.zeros_initializer()),
    'out': tf.get_variable(name='b_out', shape=[1, 10], initializer=tf.zeros_initializer())
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

with tf.Session(config=config) as sess:
    # Initializing the session
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X:batch_x, Y:batch_y})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)    # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print('')
    print("Test Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
