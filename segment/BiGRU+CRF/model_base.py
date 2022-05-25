import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell
from tensorflow.python.ops import rnn_cell_impl as rnn_cell


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    '''
    Computes a 2-D convolution on all input channels
    '''
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def cnn_text(input_, kernels, kernel_features, active=tf.nn.relu, scope='TDNN'):
    '''
    A convolutional + max-pooling layer for text classification
    :input:           input float tensor of shape [batch_size x max_sequence_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    :return
        A 2D Tensor with shape [batch_size x total_kernel_features]
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_sequence_length = input_.get_shape()[1]
    # embed_size = input_.get_shape()[-1]

    # input_: [batch_size, 1, max_sequence_length, embed_size]
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_sequence_length - kernel_size + 1

            # [batch_size x 1 x reduced_length x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(active(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(1, layers)
        else:
            output = layers[0]
    # [batch_size x sum(kernel_feature_size)]
    return output


def linear(input_, output_dim, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_dim: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_dim] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("matrix", [input_size, output_dim], dtype=input_.dtype)
        bias_term = tf.get_variable("bias", [output_dim], dtype=input_.dtype)

    # return tf.matmul(input_, matrix) + bias_term
    return tf.nn.xw_plus_b(input_, matrix, bias_term)


def highway(input_, output_dim, num_layers=1, bias=-2.0, active=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    Args:
        input_: A 2D Tensor with shape [batch x input_size]
    Returns:
        A 2D Tensor with shape [batch x output_dim]
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = active(linear(input_, output_dim, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, output_dim, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def attention(inputs, attention_size, context=None, input_length=None, scope="Attention"):
    """General attention mechanism model.
    It return the weighted average of inputs and the weights.

    Args:
        inputs: A 3D shaped Tensor [batch_size x time x input_size].
        attention_size: Size of attention unit.
        context: A 2D shaped Tensor [batch_size x context_size].
        input_length: An int32 tensor of shape `[batch_size]` defining the sequence
        length of the inputs values.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A 2D shaped Tensor [batch_size x input_size].
            states: A 2D shaped Tensor [batch_size x time].
    """

    with tf.variable_scope(scope):
        shape = inputs.get_shape()
        time = shape[1].value
        input_size = shape[2].value

        if context is not None:
            context_size = context.get_shape()[-1].value

        with tf.variable_scope('weights'):
            # First level
            weight_1 = tf.Variable(tf.truncated_normal([input_size, attention_size], stddev=0.01))
            # Second level
            weight_2 = tf.Variable(tf.truncated_normal([attention_size], stddev=0.01))
            if context is not None:
                weight_3 = tf.Variable(tf.truncated_normal([context_size, attention_size], stddev=0.01))

        with tf.variable_scope('projections'):
            # apply first level attention, output shape will be(batch_size, time, attention_size)
            first_level = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, input_size]), weight_1),
                                     [-1, time, attention_size])
            if context is not None:
                first_level = first_level + tf.expand_dims(tf.matmul(context, weight_3), 1)

            # shape: (batch_size, time)
            scores = tf.reduce_sum(weight_2 * tf.tanh(first_level), axis=2)

        with tf.variable_scope('outputs'):
            if input_length is not None:
                scores_mask = tf.sequence_mask(
                    lengths=tf.to_int32(input_length),
                    maxlen=tf.to_int32(time),
                    dtype=tf.float32)
                scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
            # normalize outputs
            scores_normalized = tf.nn.softmax(scores)
            out = tf.reduce_sum(inputs * tf.expand_dims(scores_normalized, 2), axis=1)

    return out, scores_normalized


def attention_batch_window(inputs, attention_size, window= 9, input_length=None, scope="AttentionBatchWindow"):
    with tf.variable_scope(scope):
        shape = inputs.get_shape()
        time = shape[1].value
        input_size = shape[2].value
        if window>time:
            window = time
        halfw = int(window/2)
        outs = []
        splits = tf.split(inputs, time, 1)
        contents = [tf.squeeze(c) for c in splits]
        for i in range(0, time):
            start = i - halfw
            end = i + halfw
            if start < 0:
                start = 0
            elif end >= time:
                start = time - window
            # batch, window, input_size
            inputsi = tf.slice(inputs, [0, start, 0], [-1, window, -1])

            outi, score = attention(inputsi, attention_size, context=contents[i], input_length=None, scope="AttentionWindow" + str(i))
            outs.append(tf.expand_dims(outi, 1))
        out = tf.concat(outs,1)

    return out, score


def bidirectional_lstm(input_, hidden_dim, num_layers=1, dropout_rate=None, sequence_length=None, scope="Bidirectional_lstm"):
    '''
    bidirectional recurrent neural network
    :param input_: [batch_size, max_sequence_length, input_size]
    :param hidden_dim: int, The number of units in the LSTM cell
    :param num_layers: the stack size of lstm network
    :param dropout_rate:
    :param sequence_length: An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    :param scope:
    :return:
    A tuple (outputs, output_states_fw, output_states_bw), see rnn.stack_bidirectional_dynamic_rnn
    '''
    with tf.variable_scope(scope):
        # lstm cell
        lstm_cell_fw = rnn_cell.LSTMCell(hidden_dim, use_peepholes=True)
        lstm_cell_bw = rnn_cell.LSTMCell(hidden_dim, use_peepholes=True)

        # dropout
        if dropout_rate is not None and 0.0 < dropout_rate < 1.0:
            lstm_cell_fw = rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lstm_cell_bw = rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - dropout_rate))

        # forward and backward
        return rnn.stack_bidirectional_dynamic_rnn(
            [lstm_cell_fw] * num_layers,
            [lstm_cell_bw] * num_layers,
            input_,
            sequence_length=sequence_length,
            dtype=tf.float32
        )
