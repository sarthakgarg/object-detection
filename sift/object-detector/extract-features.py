# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from skimage.transform import resize
# To read file names
import argparse as ap
import glob
import os, sys, cv2, imutils
from config import *
import numpy as np
import cPickle as pickle
from rootsift import RootSIFT

if __name__ == "__main__":
	dirs = os.listdir('../data/train')

	type_feature = sys.argv[1]

	fds = []
	labels = []
	sift = cv2.xfeatures2d.SIFT_create()

	for path in dirs:
		filenames = os.listdir('../data/train/' + path)
		for filename in filenames:
			im_path = '../data/train/' + path + '/' + filename
			im = cv2.imread(im_path)
			gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

			if type_feature == 'hog':
				im = resize(gray, (min_wdw_sz[0],min_wdw_sz[1]))
				fd = hog(gray,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
				fds.append(fd)
				labels.append(path)
			else:
				kp, fd = sift.detectAndCompute(gray, None)

				if len(kp) == 0:
					continue

				fd /= (fd.sum(axis=1, keepdims=True) + 0.0000001)
				fd = np.sqrt(fd)
				fds.append(fd)
				# print fd.shape
				# print fds
				labels.append(path)

		print 'Extracted features for ' + path

	# print fds.shape
	fds = np.array(fds)
	print fds.shape
	labels = np.array(labels)

	pickle.dump(fds, open('../data/features/features_' +  type_feature + '.p', 'wb'))
	pickle.dump(labels, open('../data/features/labels_' + type_feature + '.p', 'wb'))

	print "Completed calculating features from training images"
