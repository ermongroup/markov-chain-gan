import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from os.path import expanduser
from tensorflow.examples.tutorials.mnist import input_data
from utils.visualize import grid_transform

def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=None, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([784], tf.float32)
        })
    label = features['label']
    image = features['image']
    return image, label


def read_and_decode(path):
    filenames = ['{}_{}.tfrecords'.format(path, i) for i in range(0, 30)]
    return read_and_decode_single_example(filenames)

data_sets = input_data.read_data_sets('/data/mnist')
mnist_mean = np.mean(data_sets.train.images, axis=0)
mnist_std = np.std(data_sets.train.images, axis=0)

image, label = read_and_decode(expanduser('~') + '/data/mnist_tfrecords/mnist')
batch_size = 64

data_sampler = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=30000,
                                      min_after_dequeue=10000, num_threads=4)
noise_sampler = mnist_mean + mnist_std * tf.random_normal([batch_size, 784])

if __name__ == '__main__':
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start = time.time()

        for i in range(0, 1000):
            img = sess.run(data_sampler)
            img = grid_transform(img, size=[28, 28])
            plt.imshow(img, cmap='gray')
            plt.show()

        end = time.time()
        print(end - start)
        coord.request_stop()
        coord.join(threads)