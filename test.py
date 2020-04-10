import time
import uuid
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from bn_lstm_layer import LSTM
from tensorflow.examples.tutorials.mnist import input_data

def multi_cell(cells,
               state_is_tuple=True):
    return tf.nn.rnn_cell.MultiRNNCell(cells,
                                       state_is_tuple)

batch_size = 100
hidden_size = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

x_inp = tf.expand_dims(x, -1)
training = tf.placeholder(tf.bool)

lstm = LSTM(hidden_size, batch_size, 784, 1, apply_bn = True, is_training=training)

last_state = lstm(x_inp)
print('last_state: {}'.format(last_state))

W = tf.get_variable('W', [hidden_size, 10], initializer=tf.keras.initializers.Orthogonal())
b = tf.get_variable('b', [10])

y = tf.nn.softmax(tf.matmul(last_state, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.5)
gvs = optimizer.compute_gradients(cross_entropy)
capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.apply_gradients(capped_gvs)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Summaries
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("xe_loss", cross_entropy)
merge_summary = tf.summary.merge_all()

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

logdir = 'logs/10_test'

os.makedirs(logdir)
print('logging to ' + logdir)
train_bn_true_writer = tf.summary.FileWriter(logdir + '/train-bn-true', sess.graph)
train_bn_false_writer = tf.summary.FileWriter(logdir+'/train-bn-false')
test_writer = tf.summary.FileWriter(logdir + '/test')

current_time = time.time()
print("Using population statistics (training: False) at test time gives worse results than batch statistics")

train_steps = 100000

for i in range(train_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_loss, summary_bn_true,accuracy_train, _ = sess.run([cross_entropy, merge_summary, accuracy, train_step], feed_dict={x: batch_xs, y_: batch_ys, training: True})
    train_bn_true_writer.add_summary(summary_bn_true, i)


    summary_bn_false = sess.run(merge_summary, feed_dict={x: batch_xs, y_:batch_ys, training: False})
    train_bn_false_writer.add_summary(summary_bn_false, i)

    batch_test_x, batch_test_y = mnist.test.next_batch(batch_size) # 5_use_ilampard_lstm_layer_bn_apply_false

    summary_test = sess.run(merge_summary, feed_dict = {x: batch_test_x, y_: batch_test_y, training: False})
    test_writer.add_summary(summary_test, i)

    print('training step:{}, train_loss: {}, train loss: {}'.format(i, accuracy_train, train_loss))
