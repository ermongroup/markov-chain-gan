import tensorflow as tf
from scipy.misc import imsave
from utils.base import TransitionBase, DiscriminatorBase
from utils.visualize import figs, images_chain
from utils.layers import leaky_relu, conv2d, dense, relu_bn, batch_norm


class TransitionFunction(TransitionBase):
    def __init__(self, name='mnist/t'):
        TransitionBase.__init__(self, 784, name)

    def __call__(self, x):
        fc1 = dense(x, 600, activation_fn=leaky_relu)
        fc2 = dense(fc1, 100, activation_fn=tf.identity)
        z = batch_norm(fc2)
        z = z + 0.05 * tf.random_normal(tf.shape(fc2), mean=0.0, stddev=1)
        fc1_ = dense(z, 600, activation_fn=relu_bn)
        x_ = dense(fc1_, 784, activation_fn=tf.sigmoid)
        return x_


class Discriminator(DiscriminatorBase):
    def __init__(self, name='mnist/d'):
        DiscriminatorBase.__init__(self, 784, 2, name)

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = conv2d(x, 64, [4, 4], [2, 2], activation_fn=leaky_relu)
            conv2 = conv2d(conv1, 128, [4, 4], [2, 2], activation_fn=leaky_relu)
            conv2 = tf.contrib.layers.flatten(conv2)
            fc1 = dense(conv2, 1024, activation_fn=leaky_relu)
            fc2 = dense(fc1, 2, activation_fn=tf.identity)
            return fc2, tf.sigmoid(fc2)

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


def visualizer(model, name):
    fig = figs['mnist.mlp.chain']
    x = images_chain(model, fig, 50, size=[28, 28])
    imsave('{}.png'.format(name), x)


epoch_size = 1000
logging_freq = 100
