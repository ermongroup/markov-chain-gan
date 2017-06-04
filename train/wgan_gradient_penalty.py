import time
import tensorflow as tf
from utils.base import TransitionCell


class Trainer(object):
    def __init__(self, transition_fn, discriminator,
                 data_sampler, noise_sampler,
                 b, m, scale=10.0):
        self.name = 'wgan_gradient_penalty'
        self.transition_cell = TransitionCell(transition_fn)

        self.input_size = transition_fn.input_size
        self.state_size = transition_fn.state_size

        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1), [])) + 1
        self.m = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1
        self.steps = tf.placeholder(tf.int32, [], name='steps')

        # Bypass the use of placeholders in order to train faster.
        x, z = data_sampler, noise_sampler

        # For interpolation between samples and data.
        xl = tf.concat([x, x, x], 0)

        # Shorthand for batch size.
        xbs, zbs = tf.shape(x)[0], tf.shape(z)[0]

        # Inference Network
        with tf.variable_scope(self.name):
            with tf.variable_scope('rnn'):
                self.zs_, _ = tf.nn.dynamic_rnn(
                    self.transition_cell,
                    tf.zeros(tf.stack([zbs, self.steps, self.input_size])),
                    initial_state=z
                )

        # Training Network
        with tf.variable_scope(self.name):
            # From random noise z, after b steps, obtain samples z1.
            with tf.variable_scope('rnn', reuse=True):
                _, self.z1 = tf.nn.dynamic_rnn(
                    self.transition_cell,
                    tf.zeros(tf.stack([zbs, self.b, self.input_size])),
                    initial_state=z
                )
                self.z1_ng = tf.stop_gradient(self.z1)
                self.z1 = tf.reshape(self.z1, [zbs, 1, self.input_size])

            # From data x, after m steps, obtain samples z2
            with tf.variable_scope('rnn', reuse=True):
                _, self.z2 = tf.nn.dynamic_rnn(
                    self.transition_cell,
                    tf.zeros(tf.stack([xbs, self.m, self.input_size])),
                    initial_state=x
                )
                self.z2 = tf.reshape(self.z2, [xbs, 1, self.input_size])

            # From samples z1, after m steps, obtain samples z3.
            # For improving (empirical) training speed only.
            # The optimum for the MGAN objective is the same with or without this term.
            with tf.variable_scope('rnn', reuse=True):
                _, self.z3 = tf.nn.dynamic_rnn(
                    self.transition_cell,
                    tf.zeros(tf.stack([zbs, self.m, self.input_size])),
                    initial_state=self.z1_ng
                )
                self.z3 = tf.reshape(self.z3, [zbs, 1, self.input_size])

            # Merge to obtain samples.
            x_ = tf.reshape(tf.concat([self.z1, self.z2, self.z3], 1), [-1, self.input_size])
            self.d, _ = discriminator(x, reuse=False)
            self.d_, _ = discriminator(x_)

            # Wasserstein GAN objectives.
            self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)
            self.g_loss = tf.reduce_mean(self.d_)

            # Gradient Penalty.
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = xl * epsilon + x_ * (1 - epsilon)
            d_hat, _ = discriminator(x_hat)

            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
            self.d_loss = self.d_loss + ddx

            # Training ops.
            self.g_vars = [var for var in tf.global_variables() if self.transition_cell.name in var.name]
            self.d_vars = [var for var in tf.global_variables() if discriminator.name in var.name]

            self.d_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.d_vars)
            self.g_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=self.g_vars)

            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.saver = tf.train.Saver(var_list=[var for var in tf.global_variables() if self.name in var.name])

    def train(self, num_batches, visualizer, path, d_iters=5, epoch_size=1000, logging_freq=100):
        with self.sess as sess:
            sess.run(self.init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                start_time = time.time()
                for t in range(0, num_batches):
                    for _ in range(0, d_iters):
                        sess.run(self.d_train)

                    sess.run(self.g_train)

                    if t % logging_freq == 0:
                        d_loss = sess.run(self.d_loss)
                        g_loss = sess.run(self.g_loss)
                        print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                                    (t, time.time() - start_time, d_loss, g_loss))

                    if t % epoch_size == 0 and t > 0:
                        visualizer(self, '{}/{}'.format(path, t/epoch_size))
                        self.saver.save(sess, '{}/model.ckpt'.format(path))
            except KeyboardInterrupt:
                pass
            finally:
                coord.request_stop()
                coord.join(threads)

    def evaluate(self, steps):
        output = self.sess.run(self.zs_, feed_dict={self.steps: steps})
        return output

    def load(self, path):
        print('Loading from checkpoint: {}/model.ckpt'.format(path))
        self.saver.restore(self.sess, '{}/model.ckpt'.format(path))
        print('Session restored.')
