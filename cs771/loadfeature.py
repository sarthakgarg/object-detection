import cv2
import pickle
import os
from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np

x = pickle.load(open("sift_features_datasample1.p","rb"))
y = pickle.load(open("sift_label_datasample1.p","rb"))

print(x.shape)
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

