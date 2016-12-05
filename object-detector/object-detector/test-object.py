# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score
import cv2
import argparse as ap
from nms import nms
from config import *
import glob, os
# def sliding_window(image, window_size, step_size):
#     '''
#     This function returns a patch of the input image `image` of size equal
#     to `window_size`. The first image returned top-left co-ordinates (0, 0)
#     and are increment in both x and y directions by the `step_size` supplied.
#     So, the input parameters are -
#     * `image` - Input Image
#     * `window_size` - Size of Sliding Window
#     * `step_size` - Incremented Size of Window

#     The function returns a tuple -
#     (x, y, im_window)
#     where
#     * x is the top-left x co-ordinate
#     * y is the top-left y co-ordinate
#     * im_window is the sliding window image
#     '''
#     for y in xrange(0, image.shape[0], step_size[1]):
#         for x in xrange(0, image.shape[1], step_size[0]):
#             yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--Data_Folder", help="Path to the data features directory", required=True)
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    args = vars(parser.parse_args())

    pos_feat_path1 = args["Data_Folder"] + 'Car'
    pos_feat_path2 = args["Data_Folder"] + 'Person'
    pos_feat_path3 = args["Data_Folder"] + 'Motorcycle'
    pos_feat_path4 = args["Data_Folder"] + 'Rickshaw'
    pos_feat_path5 = args["Data_Folder"] + 'Bicycle'
    pos_feat_path6 = args["Data_Folder"] + 'Autorickshaw'

    # Classifiers supported
    clf_type = args['classifier']

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path1,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    for feat_path in glob.glob(os.path.join(pos_feat_path2,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(2)

    for feat_path in glob.glob(os.path.join(pos_feat_path3,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(3)

    for feat_path in glob.glob(os.path.join(pos_feat_path4,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(4)

    for feat_path in glob.glob(os.path.join(pos_feat_path5,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(5)

    for feat_path in glob.glob(os.path.join(pos_feat_path6,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(6)

    clf = joblib.load(model_path)
    pred = clf.predict(fds)
    # print pred
    # pred = [a[0] for a in pred]
    print classification_report(labels, pred),"Accuracy = ", accuracy_score(labels, pred)

    # Parse the command line arguments
    # parser = ap.ArgumentParser()
    # parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    # parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
    #         type=int)
    # parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
    #         action="store_true")
    # args = vars(parser.parse_args())

    # Read the image
    # im = imread(args["image"], as_grey=True)
    # min_wdw_sz = (100, 40)
    # step_size = (10, 10)
    # downscale = args['downscale']
    # visualize_det = args['visualize']
    #
    # # Load the classifier
    # clf = joblib.load(model_path)
    #
    # # List to store the detections
    # detections = []
    # # The current scale of the image
    # scale = 0
    # # Downscale the image and iterate
    #
    #     # Calculate the HOG features
    # fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    # pred = clf.predict(fd)
    # print  "Detection:: Location -> ({}, {})".format(x, y)
    # print "Scale ->  {} | Confidence Score {} \n Pred = {}".format(scale,clf.decision_function(fd),pred)

    # for feat_path in glob.glob(os.path.join(pos_feat_path1,"*.feat")):
    #     fd = joblib.load(feat_path)
    #     fds.append(fd)
    #     labels.append(1)
