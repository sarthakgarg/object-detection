# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
from config import *

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image_directory", help="Path to images",
                        required=True)
    parser.add_argument('-d', "--Data_Folder", help="Path to Data Folder",
                        required=True)
    parser.add_argument('-de', "--descriptor", help="Descriptor to be used -- HOG",
                        default="HOG")
    args = vars(parser.parse_args())

    pos_im_path1 = args["image_directory"] + 'Car'
    pos_im_path2 = args["image_directory"] + 'Person'
    pos_im_path3 = args["image_directory"] + 'Motorcycle'
    pos_im_path4 = args["image_directory"] + 'Rickshaw'
    pos_im_path5 = args["image_directory"] + 'Bicycle'
    pos_im_path6 = args["image_directory"] + 'Autorickshaw'


    pos_feat_ph1 = args["Data_Folder"] + 'Car'
    pos_feat_ph2 = args["Data_Folder"] + 'Person'
    pos_feat_ph3 = args["Data_Folder"] + 'Motorcycle'
    pos_feat_ph4 = args["Data_Folder"] + 'Rickshaw'
    pos_feat_ph5 = args["Data_Folder"] + 'Bicycle'
    pos_feat_ph6 = args["Data_Folder"] + 'Autorickshaw'


    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph1):
        os.makedirs(pos_feat_ph1)
    if not os.path.isdir(pos_feat_ph2):
        os.makedirs(pos_feat_ph2)
    if not os.path.isdir(pos_feat_ph3):
        os.makedirs(pos_feat_ph3)
    if not os.path.isdir(pos_feat_ph4):
        os.makedirs(pos_feat_ph4)
    if not os.path.isdir(pos_feat_ph5):
        os.makedirs(pos_feat_ph5)
    if not os.path.isdir(pos_feat_ph6):
        os.makedirs(pos_feat_ph6)

    WIDTH = 100
    HEIGHT = WIDTH

    for im_path in glob.glob(os.path.join(pos_im_path1, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph1, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)


    for im_path in glob.glob(os.path.join(pos_im_path2, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph2, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)
            print "Calculating the descriptors for the positive samples and saving them"

    for im_path in glob.glob(os.path.join(pos_im_path3, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph3, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)
            print "Calculating the descriptors for the positive samples and saving them"

    for im_path in glob.glob(os.path.join(pos_im_path4, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph4, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)
            print "Calculating the descriptors for the positive samples and saving them"

    for im_path in glob.glob(os.path.join(pos_im_path5, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph5, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)
            print "Calculating the descriptors for the positive samples and saving them"

    for im_path in glob.glob(os.path.join(pos_im_path6, "*")):
        im = imread(im_path, as_grey=True)
        im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph6, fd_name)
            joblib.dump(fd, fd_path)
            print "Positive features saved in {}".format(pos_feat_ph)


    print "Completed calculating features from training images"
