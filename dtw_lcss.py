import threading

import dtw as dtw
import pandas as pd
import numpy as np
import gmplot as gmplot
import time

import config
import ast
import preprocess as prep
import fastdtw as fdtw


########################################################################################################################
#                                    DTW Function + Thread-Work Function                                               #
########################################################################################################################
def dtw_compute():

    df = pd.read_csv('tripsClean.csv')
    df_test = pd.read_csv(config.testseta1Path, delimiter='/')
    threads = []

    # Iterate over test_a1 rows
    for i, t_row in df_test.iterrows():
        t = threading.Thread(target=dtw_worker, args=(i, t_row,df,df_test))
        threads.append(t)
        t.start()

def dtw_worker( i, t_row,df,df_test):
    print 'Worker:',i
    test_list = []
    neighbours = []
    tlat= []
    tlong = []
    t_trajectories = ast.literal_eval(t_row[0])


    # Collect in list all the trajectories for this trip
    for j in range(0, len(t_trajectories)):
        test_list.append( [float(t_trajectories[j][1]), float(t_trajectories[j][2])] )
        tlong.append(float(t_trajectories[j][1]))
        tlat.append(float(t_trajectories[j][2]))

    # Iterate over all trips in tripsClean
    start = time.time()
    for k, row in df.iterrows():
        clean_list = []
        trajectories = ast.literal_eval(row[2])

        for l in range(0, len(trajectories)):
            clean_list.append([float(trajectories[l][1]), float(trajectories[l][2])])

        # Compute DTW for these trips using Haversine as distance metric
        dist, cost, acc, path = dtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))
        # dist, path = fdtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))


        neighbours.append([int(row[0]), dist])
    end = time.time()


    neighbours = np.asarray(neighbours)
    neighbours = neighbours[neighbours[:, 1].argsort()][:5]
    print neighbours
    gmap = gmplot.GoogleMapPlotter(tlat[0], tlong[0], 10, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
    gmap.plot(tlat, tlong, 'green', edge_width=5)
    gmap.draw('Maps/dtwMaps/testTrip' + str(i+1) + '/test-' + str(i+1) + '.html')
    print "Test Trip ", i,"\n"
    filename='Maps/dtwMaps/testTrip' + str(i + 1) + '/data' + str(i + 1) + '.txt'
    open(filename, 'w').close()
    f = open(filename, "a+")
    for n in range(0,5):
        for g, grow in df[df['tripId'] == neighbours[n][0]].iterrows():
            gtrajectory = ast.literal_eval(grow[2])
            longlist = []
            latlist = []

            for j in range(0, len(gtrajectory)):
                longlist.append(float(gtrajectory[j][1]))
                latlist.append(float(gtrajectory[j][2]))

            gmap = gmplot.GoogleMapPlotter(latlist[0], longlist[0], 10, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
            gmap.plot(latlist, longlist, 'green', edge_width=5)
            gmap.draw('Maps/dtwMaps/testTrip' + str(i+1) + '/neighbour' + str(n+1) + '-' + str(grow[1]) + '.html')

            f.write ("Neighbor %d \nJP_ID: %s \nDTW: %8.5f\n" % (n ,grow[1],float(neighbours[n][1])))
            f.write( "dt: %8.5f\n\n" %float(end-start))
    f.close()
########################################################################################################################
def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dist = prep.haversine_dist(float(X[i-1][0]), float(X[i-1][1]), float(Y[j-1][0]), float(Y[j-1][1]))

            if dist < 0.2:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C

def backTrack(C, X, Y, i, j):
    dist = prep.haversine_dist(float(X[i - 1][0]), float(X[i - 1][1]), float(Y[j - 1][0]), float(Y[j - 1][1]))
    if i == 0 or j == 0:
        return []
    elif dist<0.2:
        return backTrack(C, X, Y, i-1, j-1) + (X[i-1])
    else:
        if C[i][j-1] > C[i-1][j]:
            return backTrack(C, X, Y, i, j-1)
        else:
            return backTrack(C, X, Y, i-1, j)





def lcss_compute():
    df = pd.read_csv('tripsClean.csv')
    df_test = pd.read_csv(config.testseta2Path, delimiter='/')
    threads = []

    # Iterate over test_a1 rows
    for i, t_row in df_test.iterrows():
        t = threading.Thread(target=lcs_worker, args=(i, t_row,df,df_test))
        threads.append(t)
        t.start()

def lcs_worker(i, t_row, df, df_test):
    print "Worker",i
    test_list = []
    tlat = []
    tlong = []
    t_trajectories = ast.literal_eval(t_row[0])
    subseqs = []
    a=1

    # Collect in list all the trajectories for this trip
    for j in range(0, len(t_trajectories)):
        test_list.append([float(t_trajectories[j][1]), float(t_trajectories[j][2])])
        tlong.append(float(t_trajectories[j][1]))
        tlat.append(float(t_trajectories[j][2]))
    testrun = 0
    # Iterate over all trips in tripsClean
    start = time.time()
    for k, row in df.iterrows():
        clean_list = []
        trajectories = ast.literal_eval(row[2])

        for l in range(0, len(trajectories)):
            clean_list.append([float(trajectories[l][1]), float(trajectories[l][2])])
        m = len(clean_list)
        n = len(test_list)

        lt =LCS(clean_list,test_list)
        if (lt[m][n]==0): continue
        # print "lt[m][n]", lt[m][n]
        sub = backTrack(lt, clean_list, test_list, m, n)
        subseqs.append((lt[m][n],sub, row[1], row[0]))
        # print "subseqs " ,subseqs
        testrun += 1
        if testrun ==10:break
    print "subseqs2",subseqs
    subseqs = sorted(subseqs,reverse=True)
    print subseqs[:5]
    end = time.time()
    gmap = gmplot.GoogleMapPlotter(tlat[0], tlong[0], 10, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
    gmap.plot(tlat, tlong, 'green', edge_width=5)
    gmap.draw('Maps/lcsMaps/testTrip' + str(i + 1) + '/test-' + str(i + 1) + '.html')
    print "Test Trip ", i, "\n"
    filename = 'Maps/lcsMaps/testTrip' + str(i + 1) + '/data' + str(i + 1) + '.txt'
    open(filename, 'w').close()
    f = open(filename, "a+")
    for n in range(0,5):
        for g, grow in df[df['tripId'] == subseqs[n][3]].iterrows():
            gtrajectory = ast.literal_eval(grow[2])
            longlist = []
            latlist = []

            for j in range(0, len(gtrajectory)):
                longlist.append(float(gtrajectory[j][1]))
                latlist.append(float(gtrajectory[j][2]))

            gmap = gmplot.GoogleMapPlotter(latlist[0], longlist[0], 10, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
            gmap.plot(latlist, longlist, 'green', edge_width=5)
            gmap.plot(tlat, tlong, 'red', edge_width=5)
            gmap.draw('Maps/lcsMaps/testTrip' + str(i+1) + '/neighbour' + str(n+1) + '-' + str(grow[1]) + '.html')

            f.write ("Neighbor %d \nJP_ID: %s \n#Matching Points: %d\n" % (n ,grow[1],float(subseqs[n][0])))
            f.write( "dt: %8.5f\n\n" %float(end-start))
    f.close()
# find . -name "*.html" -type f -delete
lcss_compute()
# dtw_compute()
