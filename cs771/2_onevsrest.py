import os, string, nltk
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from prettytable import PrettyTable
from multiprocessing import Pool
from sklearn.svm import LinearSVC
from sklearn import cross_validation,metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
import pickle
m={"Autorickshaw":0,"Bicycle":1,"Car":2,"Motorcycle":3,"Number-plate":4,"Person":5,"Rickshaw":6}
def read_data(cluster):
    i = 0
    data=[]
    labels=[]
    fdata = pickle.load(open('./clustering/' + str(cluster)+'.p',"rb"))
    flabel = pickle.load(open('sift_label_datasample1.p',"rb"))
    for line in flabel:
        labels.append(m[line])

    #print data, labels
    return fdata, labels

def kfold(data, labels, cluster):

    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    predictions = cross_validation.cross_val_predict(classifier, data, labels, cv=5, n_jobs = -1)
    print("for cluster ",cluster)
    #print predictions
    #print labels
    print(   "Accuracy = ", metrics.accuracy_score(labels, predictions),"Precision = ", metrics.precision_score(labels, predictions, pos_label = None, average = 'macro'),"Recall = " ,metrics.recall_score(labels, predictions, pos_label = None, average = 'macro'),"F1 score = " ,metrics.f1_score(labels, predictions, pos_label = None, average = 'macro'))
for i in [500,600,700,800,900,1000]:
    data,labels=read_data(i)
    kfold(data, labels,i)
