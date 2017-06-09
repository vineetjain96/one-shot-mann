import tensorflow as tf

def cosine_similarity(x, y, eps=1e-6):
	z = tf.matmul(x, tf.transpose(y, perm=[0,2,1])) 
	z /= tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(x), 2, keep_dims=True), tf.reduce_sum(tf.square(x), 2, keep_dims=True) + eps))
	return z