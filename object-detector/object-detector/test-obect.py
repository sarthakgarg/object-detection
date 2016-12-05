# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from config import *

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
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=int)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    # Read the image
    im = imread(args["image"], as_grey=True)
    min_wdw_sz = (100, 40)
    step_size = (10, 10)
    downscale = args['downscale']
    visualize_det = args['visualize']

    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate

        # Calculate the HOG features
    fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    pred = clf.predict(fd)
    print  "Detection:: Location -> ({}, {})".format(x, y)
    print "Scale ->  {} | Confidence Score {} \n Pred = {}".format(scale,clf.decision_function(fd),pred)
