import numpy as np
import tensorflow as tf

def variable_float32(x, name=''):
	return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), dtype=tf.float32), name=name)

def variable_one_hot(shape, name=''):
	initial = np.zeros(shape, dtype=np.float32)
	initial[...,0] = 1
	return tf.Variable(tf.cast(initial, dtype=tf.float32), name=name)

def cosine_similarity(x, y, eps=1e-6):
	z = tf.matmul(x, tf.transpose(y, perm=[0,2,1])) 
	z /= tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(x), 2, keep_dims=True), tf.reduce_sum(tf.square(x), 2, keep_dims=True) + eps))
	return z