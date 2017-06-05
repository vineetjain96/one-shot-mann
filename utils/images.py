import os
import random
import numpy as np

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize

def get_sampled_data(character_folders, nb_classes=5, nb_samples=10*5):
	sampled_characters = random.sample(character_folders, nb_classes)
	labels_and_images = [(label, os.path.join(character, image_path)) \
				for label, character in zip(np.arange(nb_classes), sampled_characters) \
				for image_path in os.listdir(character)]
	sampled_data = random.sample(labels_and_images, nb_samples)
	return sampled_data


def transform_image(image_path, angle=0., s=(0,0), size=(20,20)):
    original = imread(image_path, flatten=True)
    rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
    shifted = shift(rotated, shift=s)
    resized = np.asarray(imresize(rotated, size=size), dtype=np.float32)/255
    inverted = 1. - resized
    max_value = np.max(inverted)
    if max_value > 0:
        inverted /= max_value
    return inverted