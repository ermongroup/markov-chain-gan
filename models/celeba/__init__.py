import tensorflow as tf
import numpy as np
from os.path import expanduser


def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.image.resize_images(image, [32, 32])
    image = tf.reshape(image, [32 * 32 * 3])
    return byte_to_data(image)


def read_and_decode(path):
    filenames = ['{}_{}.tfrecords'.format(path, i) for i in range(0, 100)]
    return read_and_decode_single_example(filenames)


def byte_to_data(image):
    return tf.divide(tf.to_float(image), 127.5) - 1.0


def data_to_image(image):
    rescaled = np.divide(image + 1.0, 2.0)
    return np.clip(rescaled, 0.0, 1.0)


image = read_and_decode(expanduser('~') + '/data/celeba_tfrecords/celeba')
batch_size = 32
data_sampler = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=3000,
                                      min_after_dequeue=1000, num_threads=4)
noise_sampler = tf.random_normal(tf.stack([batch_size, 32 * 32 * 3]))