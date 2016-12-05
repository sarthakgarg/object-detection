import cv2
import pickle
import os
from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import argparse as ap
import glob
import os
from config import *
from sklearn.externals import joblib

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

    for f in fds :
        print(len(f))

    # if clf_type is "LIN_SVM":
    #     clf = LinearSVC(random_state = 0)
    #     classifier = OneVsRestClassifier(clf, n_jobs = 1)

    #     print "Training a Linear SVM Classifier"
    #     fds = np.asarray(fds)
    #     print(fds.shape)
    #     classifier.fit(fds, labels)
    #     # If feature directories don't exist, create them
    #     if not os.path.isdir(os.path.split(model_path)[0]):
    #         os.makedirs(os.path.split(model_path)[0])
    #     joblib.dump(clf, model_path)
    #     print "Classifier saved to {}".format(model_path)



x = fds 
y = labels

#print(x.shape)
no_sift_list = [500, 400, 300, 200, 100, 50]
X = []
for k in x:
    X.extend(k)

X = np.array(X)

num_clusters_list = [500, 600, 700, 800, 900, 1000]
for num_clusters in num_clusters_list:
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size = 100, n_init = 5)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    print (num_clusters, t_mini_batch)
    mbk_means_labels = mbk.labels_
    print(mbk_means_labels.shape)
    print(len(y))
    feat_vector = [[0 for i in range(num_clusters)] for j in range(len(y))]
    pos=0
    for i in range(len(y)):
        for j in range(len(x[i])):
            feat_vector[i][mbk_means_labels[pos]] += 1
            pos +=1
    pickle.dump(np.array(feat_vector),open("clustering/"+ str(num_clusters) +".p","wb"))
    print("over")
