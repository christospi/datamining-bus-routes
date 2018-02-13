"""
BigDataMining - Part 2.BC
(B) - Features extract for classification
(C) - Classification
"""

import ast
import csv
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from math import sqrt
import config
import numpy as np
import preprocess as prep

import pandas as pd
from gmplot import gmplot

########################################################################################################################
#                                  Function to find min long/lat to use in grid                                        #
########################################################################################################################
def min_borders():
    df = pd.read_csv('tripsClean.csv')
    lat = []
    lon = []
    clean_list = []
    for i, row in df.iterrows():
        trajectories = ast.literal_eval(row[2])
        for j in range(0, len(trajectories)):
            lon.append(float(trajectories[j][1]))
            lat.append(float(trajectories[j][2]))
            clean_list.append([float(trajectories[j][1]), float(trajectories[j][2])])
    print "Minimum lat. is: ", min(lat)
    print "Minimum long. is: ", min(lon)
    return min(lat), min(lon)

########################################################################################################################
#                                  Functions to extract features from data files                                       #
########################################################################################################################
def features_extract():
    filename = 'C_tripsClean.csv'
    with open(filename, 'wb') as csvfile:
        fieldnames = ['tripId', 'journeyPatternId', 'cells']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        df = pd.read_csv('tripsClean.csv')
        minlat, minlon = min_borders()
        for i, row in df.iterrows():
            c = ""
            trajectories = ast.literal_eval(row[2])

            for j in range(0, len(trajectories)):
                dx, dy = cell_pick(minlon, minlat, trajectories[j][1], trajectories[j][2], 0.3)
                c += str("C" + str(int(dx)) + "," + str(int(dy)) + ";")
            writer.writerow({'tripId': row[0], 'journeyPatternId': row[1], 'cells': c})

########################################################################################################################
def testset_features():
    minlat, minlon = min_borders()
    filename = 'C_test.csv'
    with open(filename, 'wb') as csvfile:
        fieldnames = ['tripId', 'cells']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        df = pd.read_csv(config.testsetPath, delimiter=';')
        for i, row in df.iterrows():
            c = ""
            trajectories = ast.literal_eval(row[1])

            for j in range(0, len(trajectories)):
                dx, dy = cell_pick(minlon, minlat, trajectories[j][1], trajectories[j][2], 0.3)
                c += str("C" + str(int(dx)) + "," + str(int(dy)) + ";")
            writer.writerow({'tripId': row[0], 'cells': c})

########################################################################################################################
def cell_pick(minlon, minlat, lon, lat, csize):
    distx = prep.haversine_dist(float(minlon), float(lat), float(minlon), float(minlat))
    disty = prep.haversine_dist(float(lon), float(minlat), float(minlon), float(minlat))
    return distx // csize, disty // csize

########################################################################################################################
#                                  Functions for Classification and Prediction                                         #
########################################################################################################################
def my_tokenizer(s):
    return s.split(';')

########################################################################################################################
def classify():
    df = pd.read_csv('C_tripsClean.csv')
    # df = pd.read_csv('C_tripsClean_v5.csv')
    # df = pd.read_csv('C_tripsClean_v6.csv')
    label = preprocessing.LabelEncoder()
    target_cat=[]
    label.fit(df['journeyPatternId'])
    x_train = df['cells']
    y_train = label.transform(df['journeyPatternId'])
    for jid in df['journeyPatternId']:
        target_cat.append(jid)
    countVect = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(2, 2))
    hashVect = HashingVectorizer(tokenizer=my_tokenizer, ngram_range=(2, 2))

    # Count Vectorizer + Tfidf + KNN
    cl_knn1 = Pipeline([('vect', countVect),
                        ('tfidf', TfidfTransformer()),
                        ('clf', KNeighborsClassifier())])
    # Hashing Vectorizer + Tfidf + KNN
    cl_knn2 = Pipeline([('vect', hashVect),
                        ('tfidf', TfidfTransformer()),
                        ('clf', KNeighborsClassifier())])
    # Count Vectorizer + Logistic Regression
    cl_lgr1 = Pipeline([('vect', countVect),
                        ('clf', LogisticRegression())])
    # Hashing Vectorizer + Logistic Regression
    cl_lgr2 = Pipeline([('vect', hashVect),
                        ('clf', LogisticRegression())])
    # Count Vectorizer + Tfidf + Logistic Regression
    cl_lgr1tf = Pipeline([('vect', countVect),
                          ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression())])
    # Hashing Vectorizer + Tfidf + Logistic Regression
    cl_lgr2tf = Pipeline([('vect', hashVect),
                          ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression())])
    # Count Vectorizer + Random Forest
    cl_rf1 = Pipeline([('vect', countVect),
                       ('clf', RandomForestClassifier())])
    # Hashing Vectorizer + Random Forest
    cl_rf2 = Pipeline([('vect', hashVect),
                       ('clf', RandomForestClassifier())])
    # Count Vectorizer + Tfidf + Random Forest
    cl_rf1tf = Pipeline([('vect', countVect),
                         ('tfidf', TfidfTransformer()),
                       ('clf', RandomForestClassifier())])
    # Hashing Vectorizer + Tfidf + Random Forest
    cl_rf2tf = Pipeline([('vect', hashVect),
                         ('tfidf', TfidfTransformer()),
                       ('clf', RandomForestClassifier())])


    scores = cross_val_score(cl_knn1, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("KNN1: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_knn2, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("KNN2: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_rf1, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("rf1: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_rf2, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("rf2: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_rf2tf, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("rf2tf: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_rf1tf, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("rf1tf: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_lgr1tf, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("lgr1tf: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_lgr2tf, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("lgr2tf: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_lgr1, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("lgr1: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(cl_lgr2, x_train, y_train, cv=10)
    accrf = scores.mean()
    print("lgr2: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # predict(cl_knn2, x_train, target_cat,1)
    # predict(cl_rf1, x_train,  target_cat, 2)
    # predict(cl_rf2, x_train,  target_cat,3)
    # predict(cl_knn1, x_train, target_cat, 4)
########################################################################################################################
def predict(clf, x_train,  target_cat,i):
    testdata = pd.read_csv('C_test.csv')
    # target_cat = np.array(target_cat)
    test_tripid = []
    for id in testdata['tripId']:
        test_tripid.append(id)
    clf.fit(x_train,  target_cat)
    test = testdata['cells']
    predictions = clf.predict(test)
    with open('testSet_JourneyPatternIDs'+str(i)+'.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter='\t')
        fieldnames=['Test_Trip_ID',  'Predicted_JourneyPatternID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        i=0
        for row in predictions:
            writer.writerow({'Test_Trip_ID': test_tripid[i], 'Predicted_JourneyPatternID': row})
            i+=1

########################################################################################################################

# testset_features()
# features_extract()
# classify()
