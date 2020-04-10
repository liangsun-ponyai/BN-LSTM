from tensorflow.keras import layers
import tensorflow as tf
class LSTM(layers.Layer):
    def __init__(self,
                 hidden_dim,
                 batch_size,
                 num_steps,
                 feature_size,
                 apply_bn=False,
                 is_training=False,
                 decay=0.9):
        """
            Initialize the LSTM layer
            Build the network given the inputs_shape passed
            Vanilla LSTM architecture -  (Hochreiter & Schmidhuber, 1997)
            Batch norm architecture - https://arxiv.org/abs/1603.09025
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.apply_bn = apply_bn
        self.is_training = is_training
        self.decay = decay

        self.feature_size = feature_size

        self.W_XH = tf.Variable(name='W_XH', \
            initial_value=tf.initializers.orthogonal()(shape=[self.feature_size, 4 * self.hidden_dim]), trainable=True)
        self.W_HH = tf.Variable(name='W_HH', \
            initial_value=tf.initializers.orthogonal()(shape=[self.hidden_dim, 4 * self.hidden_dim]), trainable=True)
        self.bias = tf.Variable(name='bias', \
            initial_value=tf.initializers.ones()(shape=[4 * self.hidden_dim]), trainable=True)

    def batch_norm(self, inputs, idx_step, scope, offset=0, scale=0.1, variance_epsilon=1e-5):
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            # Initialize the population stats for all time steps

            self.scale = tf.get_variable('scale', [self.num_steps, input_dim], initializer=tf.constant_initializer(0.1))
            self.offset = tf.get_variable('offset', [self.num_steps, input_dim], initializer=tf.zeros_initializer())

            current_step_scale = self.scale[idx_step]
            current_step_offset = self.offset[idx_step]

            self.pop_mean = tf.get_variable(name='pop_mean',
                                            shape=[self.num_steps, input_dim],
                                            initializer=tf.zeros_initializer())

            self.pop_var = tf.get_variable(name='pop_var',
                                           shape=[self.num_steps, input_dim],
                                           initializer=tf.ones_initializer())
            pop_mean = self.pop_mean[idx_step]
            pop_var = self.pop_var[idx_step]
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

            def batch_statistics():
                pop_mean_new = pop_mean * self.decay + batch_mean * (1 - self.decay)
                pop_var_new = pop_var * self.decay + batch_var * (1 - self.decay)
                with tf.control_dependencies([pop_mean.assign(pop_mean_new),
                                              pop_var.assign(pop_var_new)]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean,
                                                     batch_var,
                                                     current_step_offset,
                                                     current_step_scale,
                                                     variance_epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(inputs,
                                                 pop_mean,
                                                 pop_var,
                                                 current_step_offset,
                                                 current_step_scale,
                                                 variance_epsilon)

            return tf.cond(self.is_training, batch_statistics, population_statistics)

    def call(self, inputs, **kwargs):
        """ Return the hidden states for all time steps """
        self.init_hidden_state = tf.zeros([self.batch_size, self.hidden_dim])
        self.init_cell_state = tf.zeros([self.batch_size, self.hidden_dim])

        # (num_steps, batch_size, input_dim)
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple = tf.scan(self.one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state,
                                           self.init_cell_state,
                                           0))

        # (batch_size, num_steps, hidden_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state[:, -1, :]

    def one_step(self, prev_state_tuple, current_input):
        """ Move along the time axis by one step  """
        hidden_state, cell_state, idx_step = prev_state_tuple

        feature_size = current_input.get_shape().as_list()[1]

        if self.apply_bn:
            hidden = self.batch_norm(tf.matmul(current_input, self.W_XH), idx_step, 'batch_norm_w_xh')+ \
                     self.batch_norm(tf.matmul(hidden_state, self.W_HH), idx_step, 'batch_norm_w_hh') + \
                     self.bias
        else:
            hidden = tf.matmul(current_input, self.W_XH) + \
                     tf.matmul(hidden_state, self.W_HH) + \
                     self.bias

        forget_, input_, output_, cell_bar = tf.split(hidden,
                                                      axis=1,
                                                      num_or_size_splits=4)

        # (batch_size, hidden_dim)
        cell_state = tf.nn.sigmoid(forget_ + 1.) * cell_state + tf.nn.sigmoid(input_) * tf.nn.tanh(cell_bar)
        cell_state_normal = self.batch_norm(cell_state, idx_step, 'lstm_cell_state')
        hidden_state = tf.nn.sigmoid(output_) * tf.nn.tanh(cell_state_normal)

        return (hidden_state, cell_state, idx_step + 1)