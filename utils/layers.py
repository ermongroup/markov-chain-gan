import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x, updates_collections=None), alpha=alpha)


def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x, updates_collections=None))


def batch_norm(x):
    return tcl.batch_norm(x, updates_collections=None)


def minibatch_discrimination(x, output_size, num_kernels, dim_per_kernel, scope='mbd'):
    # input_size, num_kernels, output_size and dim_per_kernel corresponds to A, n, B, C in the Improve GAN paper.
    # their implementation seems to
    with tf.variable_scope(scope) as vs:
        batch_size = tf.shape(x)[0]
        mini_batches = batch_size / num_kernels
        input_size = x.get_shape().as_list()[1]
        t = tf.get_variable('tensor', shape=[input_size, output_size * dim_per_kernel])
        m = tf.matmul(x, t) # m = (batch_size, output_size * dim_per_kernel)
        m = tf.reshape(m, [mini_batches, num_kernels, output_size, dim_per_kernel])

        # [mini_batches, num_kernels, output_size, dim_per_kernel, num_kernels]
        mi = tf.expand_dims(m, axis=-1)
        mj = tf.transpose(mi, perm=[0, 4, 2, 3, 1])
        w = tf.reduce_sum(tf.exp(-tf.reduce_sum(tf.abs(mi - mj), axis=3)), axis=3)

        # w should have dimension [mini_batches, num_kernels, output_size]
        w = tf.reshape(w, [batch_size, output_size])
        y = tf.concat(1, [x, w])
        return y


def nn_conv2d_transpose(input, num_outputs, kernel_size, stride,
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                        activation_fn=tf.identity
                        ):
    size = input.get_shape().as_list()[1:3]
    new_size = [size[0] * stride[0], size[1] * stride[1]]
    input = tf.image.resize_nearest_neighbor(input, new_size)
    return tc.layers.convolution2d_transpose(
        input, num_outputs, kernel_size, [1, 1],
        weights_initializer=weights_initializer,
        weights_regularizer=weights_regularizer,
        activation_fn=activation_fn)


def bi_conv2d_transpose(input, num_outputs, kernel_size, stride,
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                        activation_fn=tf.identity
                        ):
    size = input.get_shape().as_list()[1:3]
    new_size = [size[0] * stride[0], size[1] * stride[1]]
    input = tf.image.resize_bilinear(input, new_size)
    return tc.layers.convolution2d_transpose(
        input, num_outputs, kernel_size, [1, 1],
        weights_initializer=weights_initializer,
        weights_regularizer=weights_regularizer,
        activation_fn=activation_fn)


def conv2d(input, num_outputs, kernel_size, stride,
           weights_initializer=tf.random_normal_initializer(stddev=0.02),
           weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
           normalizer_fn=None,
           normalizer_params=None,
           activation_fn=tf.identity
           ):
    return tc.layers.conv2d(input, num_outputs, kernel_size, stride,
                            weights_initializer=weights_initializer,
                            weights_regularizer=weights_regularizer,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            activation_fn=activation_fn)


def conv2d_transpose(input, num_outputs, kernel_size, stride,
                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                     weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                     normalizer_fn=None,
                     normalizer_params=None,
                     activation_fn=tf.identity
                     ):
    return tc.layers.convolution2d_transpose(
        input, num_outputs, kernel_size, stride,
        weights_initializer=weights_initializer,
        weights_regularizer=weights_regularizer,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        activation_fn=activation_fn)


def dense(input, num_outputs, activation_fn=tf.identity, normalizer_fn=None, normalizer_params=None):
    return tcl.fully_connected(input, num_outputs, activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)


lrelu = leaky_relu
lrelu_bn = leaky_relu_batch_norm
relu_bn = relu_batch_norm
