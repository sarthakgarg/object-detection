# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse as ap
import glob
import os
from config import *
from sklearn.cluster import MiniBatchKMeans
from sklearn import cross_validation, metrics
import numpy as np
import time
import random



def kfold(data, labels):
    classifier = LinearSVC(random_state= 0)
    predictions = cross_validation.cross_val_predict(classifier, data, labels, cv=5, n_jobs = -1)
    print(   "Accuracy = ", metrics.accuracy_score(labels, predictions),"Precision = ", metrics.precision_score(labels, predictions, pos_label = None, average = 'macro'),"Recall = " ,metrics.recall_score(labels, predictions, pos_label = None, average = 'macro'),"F1 score = " ,metrics.f1_score(labels, predictions, pos_label = None, average = 'macro'))


if __name__ == "__main__":
    # Parse the command line arguments
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

    data = zip(fds, labels)
    random.shuffle(data)
    fds = [x for (x,z) in data]
    labels = [y for (w,y) in data]
    kfold(fds, labels)



    # if clf_type is "LIN_SVM":
    #     clf = LinearSVC(random_state = 0)
    #     classifier = OneVsRestClassifier(clf, n_jobs = -1)

    #     print "Training a Linear SVM Classifier"
    #     # fds = np.asarray(fds)
    #     # print(fds.shape)
    #     classifier.fit(fds, labels)
    #     # If feature directories don't exist, create them
    #     if not os.path.isdir(os.path.split(model_path)[0]):
    #         os.makedirs(os.path.split(model_path)[0])
    #     joblib.dump(classifier, model_path)
    #     print "Classifier saved to {}".format(model_path)
