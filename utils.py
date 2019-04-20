import numpy as np
import tensorflow as tf
from tensorflow.python import keras


def expand(x, dim, N):
    ndim = tf.shape(tf.shape(x))[0]
    expand_idx = tf.concat([tf.ones((tf.maximum(0, dim),), dtype=tf.int32), tf.reshape(N, (-1,)),
                            tf.ones((tf.minimum(ndim - dim, ndim),), dtype=tf.int32)], axis=0)
    return tf.tile(tf.expand_dims(x, dim), expand_idx)


def learned_init(units):
    return tf.Variable(initial_value=keras.initializers.glorot_uniform()(shape=(units,)))


def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return keras.initializers.truncated_normal(stddev=stddev, dtype=dtype)
