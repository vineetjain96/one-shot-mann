import numpy as np
import tensorflow as tf

from model import mann
from utils.generators import OmniglotGenerator

def omniglot():

	batch_size = 16
	nb_classes = 5
	nb_samples = 10*5
	input_size = 20*20

	nb_reads = 4
	controller_size = 200
	memory_size = (128,40)

	input_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_samples, input_size))
	target_var = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_samples))

	generator = OmniglotGenerator(data_folder='data/omniglot', batch_size=batch_size, nb_classes=nb_classes, \
			nb_samples=nb_samples, max_rotation=0., max_shift=0, img_size=(20, 20))

	net = mann(input_size=input_size, memory_size=memory_size, controller_size=controller_size, \
			nb_reads=nb_reads, nb_classes=nb_classes, batch_size=batch_size)
	output_var, params = net.compute_output(input_var, target_var)

	with tf.variable_scope('weights', reuse=True):
		W_key = tf.get_variable('W_key', shape=(nb_reads, controller_size, memory_size[1]))
		b_key = tf.get_variable('b_key', shape=(nb_reads, memory_size[1]))

		W_sigma = tf.get_variable('W_sigma', shape=(nb_reads, controller_size, 1))
		b_sigma = tf.get_variable('b_sigma', shape=(nb_reads, 1))

		W_xh = tf.get_variable('W_xh', shape=(input_size + nb_classes, 4*controller_size))
		W_hh = tf.get_variable('W_hh', shape=(controller_size, 4*controller_size))
		b_h = tf.get_variable('b_h', shape=(4*controller_size))

		W_o = tf.get_variable('W_o', shape=(controller_size + nb_reads * memory_size[1], nb_classes))
		b_o = tf.get_variable('b_o', shape=(nb_classes))

		gamma = 0.95

	params = [W_key, b_key, W_sigma, b_sigma, W_xh, W_hh, b_h, W_o, b_o]

	target_one_hot = tf.one_hot(target_var, nb_classes, axis=-1)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_var, labels=target_one_hot), name="cost")
	opt = tf.train.AdamOptimizer(learning_rate=1e-3)
	train_step = opt.minimize(cost, var_list=params)

	max_iter = 1000

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	with sess.as_default():
		try:
			for i in range(max_iter):
				episode_input, episode_output = generator.episode()
				feed_dict = {input_var: episode_input, target_var: episode_output}
				train_step.run(feed_dict)
				score = cost.eval(feed_dict)
				print 'Step ' + str(i) + ': cost = ' + str(score)
		except KeyboardInterrupt:
			print 'Final result --> Step ' + str(i) + ': cost = ' + score
			pass

if __name__ == '__main__':
    omniglot()