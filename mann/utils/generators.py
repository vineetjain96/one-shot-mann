import os
import numpy as np

from images import get_sampled_data, transform_image

class OmniglotGenerator(object):

	def __init__(self, data_folder, batch_size=1, nb_classes=5, nb_samples=10*5, max_rotation=np.pi/6, \
			max_shift=10, img_size=(20, 20)):
		self.data_folder = data_folder
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.nb_samples = nb_samples
		self.max_rotation = max_rotation
		self.max_shift = max_shift
		self.img_size = img_size
		self.character_folders = [os.path.join(data_folder, alphabet, character) for alphabet in os.listdir(data_folder) \
         						if os.path.isdir(os.path.join(data_folder, alphabet)) \
         						for character in os.listdir(os.path.join(data_folder, alphabet))]
	
	def episode(self):
		episode_input = np.zeros((self.batch_size, self.nb_samples, np.prod(self.img_size)), dtype=np.float32)
		episode_output = np.zeros((self.batch_size, self.nb_samples), dtype=np.int32)

		for i in range(self.batch_size):
			sampled_data = get_sampled_data(self.character_folders, nb_classes=self.nb_classes, nb_samples=self.nb_samples)
			sequence_length = len(sampled_data)
			labels, image_files = zip(*sampled_data)

			angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
			shifts = np.random.randint(-self.max_shift, self.max_shift + 1, size=(sequence_length, 2))

			episode_input[i] = np.asarray([transform_image(filename, angle=angle, s=shift, size=self.img_size).flatten() \
	        	 				for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
			episode_output[i] = np.asarray(labels, dtype=np.int32)

		return episode_input, episode_output