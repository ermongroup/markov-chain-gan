from utils import *
from __init__ import *


class TransitionFunction(TransitionBase):
    def __init__(self, name='celeba/t'):
        TransitionBase.__init__(self, 32 * 32 * 3, name)

    def __call__(self, x):
        bs = tf.shape(x)[0]
        x = tf.reshape(x, [bs, 32, 32, 3])
        c1 = conv2d(x, 64, [4, 4], [2, 2], activation_fn=lrelu)
        c1 = tcl.flatten(c1)
        z = dense(c1, 200, activation_fn=tcl.batch_norm) + 0.2 * tf.random_normal(tf.stack([bs, 200]), mean=0.0, stddev=1)
        c1 = dense(z, 16 * 16 * 64)
        c1 = relu_bn(tf.reshape(c1, tf.stack([bs, 16, 16, 64])))
        x_ = conv2d_transpose(c1, 3, [4, 4], [2, 2], activation_fn=tf.nn.tanh)
        x_ = tf.reshape(x_, tf.stack([bs, 32 * 32 * 3]))
        return x_


class Discriminator(DiscriminatorBase):
    def __init__(self, name='celeba/d'):
        DiscriminatorBase.__init__(self, 32 * 32 * 3, 2, name)

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 32, 32, 3])
            conv1 = conv2d(x, 64, [4, 4], [2, 2], activation_fn=lrelu)
            conv2 = conv2d(conv1, 128, [4, 4], [2, 2], activation_fn=lrelu)
            conv3 = conv2d(conv2, 256, [4, 4], [2, 2], activation_fn=lrelu)
            conv3 = tcl.flatten(conv3)
            y = dense(conv3, 2, activation_fn=tf.identity)
            return y, tf.sigmoid(y)

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


def visualizer(model, name):
    fig = figs['celeba']
    x = images_chain(model, fig, 50, size=[32, 32, 3], data_to_image=data_to_image)
    imsave('{}.png'.format(name), x)


epoch_size = 1000
logging_freq = 100