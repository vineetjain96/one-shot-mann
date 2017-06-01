import numpy as np
import tensorflow as tf

from utils.tf_utils import variable_float32, variable_one_hot
from utils.similarities import cosine_similarity

class mann(object):

	def __init__(self, input_size=20*20, memory_size=(128, 40), \
		controller_size=200, nb_reads=4, nb_class=5, batch_size=16):
		self.input_size = input_size
		self.memory_size = memory_size
		self.controller_size = controller_size
		self.nb_reads = nb_reads
		self.nb_class = nb_class
		self.batch_size = batch_size

	def initialize(self):
		M_0 = variable_float32(1e-6 * np.ones((self.batch_size,) + self.memory_size), name='memory')
		c_0 = variable_float32(np.zeros((self.batch_size, self.controller_size)), name='controller_cell_state')
		h_0 = variable_float32(np.zeros((self.batch_size, self.controller_size)), name='controller_hidden_state')
		r_0 = variable_float32(np.zeros((self.batch_size, self.nb_reads * self.memory_size[1])), name='read_vector')
		wr_0 = variable_one_hot((self.batch_size, self.nb_reads, self.memory_size[0]), name='wr')
		wu_0 = variable_one_hot((self.batch_size, self.memory_size[0]), name='wu')

		return [M_0, c_0, h_0, r_0, wr_0, wu_0]

	def step(self, (M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, wu_tm1), (x_t)):
		with tf.variable_scope('weights', reuse=True):
			W_key = tf.get_variable('W_key', shape=(self.nb_reads, self.controller_size, self.memory_size[1]))
			b_key = tf.get_variable('b_key', shape=(self.nb_reads, self.memory_size[1]))

			W_sigma = tf.get_variable('W_sigma', shape=(self.nb_reads, self.controller_size, 1))
			b_sigma = tf.get_variable('b_sigma', shape=(self.nb_reads, 1))

			W_xh = tf.get_variable('W_xh', shape=(self.input_size + self.nb_class, 4*self.controller_size))
			W_hh = tf.get_variable('W_hh', shape=(self.controller_size, 4*self.controller_size))
			b_h = tf.get_variable('b_h', shape=(4*self.controller_size))

			W_o = tf.get_variable('W_o', shape=(self.controller_size + self.nb_reads * self.memory_size[1], self.nb_class))
			b_o = tf.get_variable('b_o', shape=(self.nb_class))

			gamma = 0.95

		def lstm_step(size, x_t, c_tm1, h_tm1, W_xh, W_hh, b_h):

			preactivations = tf.matmul(x_t, W_xh) + tf.matmul(h_tm1, W_hh) + b_h

			gf = tf.sigmoid(preactivations[:, 0:size])
			gi = tf.sigmoid(preactivations[:, size:2*size])
			go = tf.sigmoid(preactivations[:, 2*size:3*size])
			u = tf.tanh(preactivations[:, 3*size:4*size])

			c_t = gf*c_tm1 + gi*u
			h_t = go*tf.tanh(c_t)

			return [c_t, h_t]

		[c_t, h_t] = lstm_step(self.controller_size, x_t, c_tm1, h_tm1, W_xh, W_hh, b_h)

		shape_key = (self.batch_size, self.nb_reads, self.memory_size[1])
		shape_sigma = (self.batch_size, self.nb_reads, 1)

		_W_key = tf.reshape(W_key, shape=(self.controller_size, -1))
		_W_sigma = tf.reshape(W_sigma, shape=(self.controller_size, -1))

		k_t = tf.tanh(tf.reshape(tf.matmul(h_t, _W_key), shape=shape_key) + b_key)
		sigma_t = tf.sigmoid(tf.reshape(tf.matmul(h_t, _W_sigma), shape=shape_sigma) + b_sigma)


		_, indices = tf.nn.top_k(wu_tm1, k=self.memory_size[0])
		wlu_tm1 = tf.slice(indices, [0,self.memory_size[0] - self.nb_reads], [self.batch_size,self.nb_reads])
		wlu_tm1 = tf.cast(wlu_tm1, dtype=tf.int32)

		row_idx = tf.reshape(tf.tile(tf.reshape(wlu_tm1[:,0], shape=(-1, 1)), (1, self.memory_size[1])), [-1])
		row_idx += self.memory_size[0] * tf.reshape(tf.tile(tf.reshape(range(self.batch_size), shape=(-1, 1)), (1, self.memory_size[1])), [-1])
		col_idx = tf.tile(range(self.memory_size[1]), [self.batch_size])
		coords = tf.transpose(tf.stack([row_idx, col_idx]))
		binary_mask = tf.cast(tf.sparse_to_dense(coords, (self.batch_size*self.memory_size[0], self.memory_size[1]), 1), tf.bool)
		
		M_t = tf.where(binary_mask, tf.constant(0., shape=(self.batch_size*self.memory_size[0], self.memory_size[1])), tf.reshape(M_tm1, shape=(self.batch_size*self.memory_size[0], self.memory_size[1])))
		M_t = tf.reshape(M_t, shape=(self.batch_size, self.memory_size[0], self.memory_size[1]))

		wlu_tm1 = tf.one_hot(wlu_tm1, self.memory_size[0], axis=-1)
		ww_t = tf.multiply(sigma_t, wr_tm1) + tf.multiply(1.-sigma_t, wlu_tm1)

		K_t = cosine_similarity(k_t, M_t)
		wr_t = tf.nn.softmax(K_t)

		wu_t = gamma*wu_tm1 + tf.reduce_sum(wr_t, axis=1)+ tf.reduce_sum(ww_t, axis=1)
		r_t = tf.reshape(tf.matmul(wr_t, M_t), shape=(self.batch_size,-1))

		return [M_t, c_t, h_t, r_t, wr_t, wu_t]

	def compute_output(self, input_var, target_var):
		[M_0, c_0, h_0, r_0, wr_0, wu_0] = self.initialize()

		with tf.variable_scope('weights'):
			W_key = tf.get_variable('W_key', shape=(self.nb_reads, self.controller_size, self.memory_size[1]))
			b_key = tf.get_variable('b_key', shape=(self.nb_reads, self.memory_size[1]))

			W_sigma = tf.get_variable('W_sigma', shape=(self.nb_reads, self.controller_size, 1))
			b_sigma = tf.get_variable('b_sigma', shape=(self.nb_reads, 1))

			W_xh = tf.get_variable('W_xh', shape=(self.input_size + self.nb_class, 4*self.controller_size))
			W_hh = tf.get_variable('W_hh', shape=(self.controller_size, 4*self.controller_size))
			b_h = tf.get_variable('b_h', shape=(4*self.controller_size))

			W_o = tf.get_variable('W_o', shape=(self.controller_size + self.nb_reads * self.memory_size[1], self.nb_class))
			b_o = tf.get_variable('b_o', shape=(self.nb_class))

			gamma = 0.95

		sequence_length = input_var.get_shape().as_list()[1]

		one_hot_target = tf.one_hot(target_var, self.nb_class, axis=-1)
		offset_target_var = tf.concat([tf.zeros_like(tf.expand_dims(one_hot_target[:,0], 1)), one_hot_target[:,:-1]], axis=1)
		ntm_input = tf.concat([input_var, offset_target_var], axis=2)

		ntm_var = tf.scan(self.step, elems=tf.transpose(ntm_input, perm=[1,0,2]), initializer=[M_0, c_0, h_0, r_0, wr_0, wu_0])
		ntm_output = tf.transpose(tf.concat(ntm_var[2:4], axis=2), perm=[1,0,2])

		output_var = tf.matmul(tf.reshape(ntm_output, shape=(self.batch_size*sequence_length, -1)), W_o) + b_o
		output_var = tf.reshape(output_var, shape=(self.batch_size, sequence_length, -1))
		output_var = tf.nn.softmax(output_var)

		params = [W_key, b_key, W_sigma, b_sigma, W_xh, W_hh, b_h, W_o, b_o]

		return output_var, params