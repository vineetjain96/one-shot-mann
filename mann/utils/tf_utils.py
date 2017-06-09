import numpy as np
import tensorflow as tf

def variable_float32(x, name=''):
	return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), dtype=tf.float32), name=name)

def variable_one_hot(shape, name=''):
	initial = np.zeros(shape, dtype=np.float32)
	initial[...,0] = 1
	return tf.Variable(tf.cast(initial, dtype=tf.float32), name=name)