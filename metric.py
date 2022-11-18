import tensorflow
import tensorflow as tf


def rounded_mae(label, prediction):
    rounded_prediction = tf.math.round(prediction)
    error = tensorflow.reduce_mean(tf.abs(label-rounded_prediction))
    return error
