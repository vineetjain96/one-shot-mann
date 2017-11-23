import numpy as np
import tensorflow as tf

from mann.model import mann
from mann.utils.generators import OmniglotGenerator

from matplotlib import pyplot as plt
from argparse import ArgumentParser

BATCH_SIZE = 16
NB_CLASSES = 5
NB_SAMPLES = 10*5
INPUT_HEIGHT = 20
INPUT_WIDTH = 20

NB_READS = 4
CONTROLLER_SIZE = 200
MEMORY_LOCATIONS = 128
MEMORY_WORD_SIZE = 40

LEARNING_RATE = 1e-4
ITERATIONS = 100000

def build_argparser():
	parser = ArgumentParser()

	parser.add_argument('--batch-size',
			dest='_batch_size',	help='Batch size (default: %(default)s)',
			type=int, default=BATCH_SIZE)
	parser.add_argument('--num-classes',
			dest='_nb_classes', help='Number of classes in each episode (default: %(default)s)',
			type=int, default=NB_CLASSES)
	parser.add_argument('--num-samples',
			dest='_nb_samples', help='Number of total samples in each episode (default: %(default)s)',
			type=int, default=NB_SAMPLES)
	parser.add_argument('--input-height',
			dest='_input_height', help='Input image height (default: %(default)s)',
			type=int, default=INPUT_HEIGHT)
	parser.add_argument('--input-width',
			dest='_input_width', help='Input image width (default: %(default)s)',
			type=int, default=INPUT_WIDTH)
	parser.add_argument('--num-reads',
			dest='_nb_reads', help='Number of read heads (default: %(default)s)',
			type=int, default=NB_READS)
	parser.add_argument('--controller-size',
			dest='_controller_size', help='Number of hidden units in controller (default: %(default)s)',
			type=int, default=CONTROLLER_SIZE)
	parser.add_argument('--memory-locations',
			dest='_memory_locations', help='Number of locations in the memory (default: %(default)s)',
			type=int, default=MEMORY_LOCATIONS)
	parser.add_argument('--memory-word-size',
			dest='_memory_word_size', help='Size of each word in memory (default: %(default)s)',
			type=int, default=MEMORY_WORD_SIZE)
	parser.add_argument('--learning-rate',
			dest='_learning_rate', help='Learning Rate (default: %(default)s)',
			type=float, default=LEARNING_RATE)
	parser.add_argument('--iterations',
			dest='_iterations', help='Number of iterations for training (default: %(default)s)',
			type=int, default=ITERATIONS)

	return parser


def omniglot():

	parser = build_argparser()
	args = parser.parse_args()

	batch_size = args._batch_size
	nb_classes = args._nb_classes
	nb_samples = args._nb_samples
	img_size = (args._input_height, args._input_width)
	input_size = args._input_height * args._input_width

	nb_reads = args._nb_reads
	controller_size = args._controller_size
	memory_size = (args._memory_locations, args._memory_word_size)
	
	learning_rate = args._learning_rate
	max_iter = args._iterations

	input_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_samples, input_size))
	target_var = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_samples))

	generator = OmniglotGenerator(data_folder='data/omniglot', batch_size=batch_size, nb_classes=nb_classes, \
			nb_samples=nb_samples, max_rotation=0., max_shift=0, img_size=img_size)

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
	acc = tf.reduce_mean(tf.cast(tf.equal(target_var, tf.cast(tf.argmax(output_var, axis=2), dtype=tf.int32)), dtype=tf.float32))

	opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, momentum=0.9)
	train_step = opt.minimize(cost, var_list=params)


	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	iters = []
	accuracies = []
	costs = []

	with sess.as_default():
		try:
			for i in range(max_iter):
				episode_input, episode_output = generator.episode()
				feed_dict = {input_var: episode_input, target_var: episode_output}
				train_step.run(feed_dict)
				if i % (max_iter*1e-3) == 0:
					cost_val = sess.run(cost, feed_dict=feed_dict)
					acc_val = sess.run(acc, feed_dict=feed_dict)

					iters.append(i)
					costs.append(cost_val)
					accuracies.append(acc_val)
					
					print 'Target Labels:'
					print sess.run(target_var[0], feed_dict=feed_dict)
					print 'Model Output:'
					print sess.run(tf.argmax(output_var[0], axis=1), feed_dict=feed_dict)
					print 'Episode ' + str(i) + ': Cost = ' + str(cost_val) + '\t Accuracy = ' + str(acc_val)
					print ''

					with open('omniglot-cost', 'wb') as fp:
						pickle.dump(costs, fp)

					with open('omniglot-acc', 'wb') as fp:
						pickle.dump(accuracies, fp)

					with open('omniglot-iters', 'wb') as fp:
						pickle.dump(iters, fp)

		except KeyboardInterrupt:
			print '\nInterrupted at Episode ' + str(i)
			print 'Cost = ' + str(cost_val)
			print 'Accuracy = ' + str(acc_val)
			pass

	
	fig = plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	plt.plot(iters, costs, 'b', label='Training Error', linewidth=2, alpha=0.8)
	plt.xlabel('Episodes', fontsize=22)
	plt.ylabel('Cross Entropy Loss', fontsize=22)
	plt.title('Training Error', fontsize=26)

	plt.subplot(1,2,2)
	plt.plot(iters, accuracies, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
	plt.xlabel('Episodes', fontsize=22)
	plt.ylabel('Accuracy', fontsize=22)
	plt.title('Training Accuracy', fontsize=26)
	plt.show()


if __name__ == '__main__':
    omniglot()