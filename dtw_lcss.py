import threading

import dtw as dtw
import pandas as pd
import numpy as np
import gmplot as gmplot

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

    testrun = 0

    # Collect in list all the trajectories for this trip
    for j in range(0, len(t_trajectories)):
        test_list.append( [float(t_trajectories[j][1]), float(t_trajectories[j][2])] )
        tlong.append(float(t_trajectories[j][1]))
        tlat.append(float(t_trajectories[j][2]))

    # Iterate over all trips in tripsClean
    for k, row in df.iterrows():
        clean_list = []
        trajectories = ast.literal_eval(row[2])

        for l in range(0, len(trajectories)):
            clean_list.append([float(trajectories[l][1]), float(trajectories[l][2])])

        # Compute DTW for these trips using Haversine as distance metric
        dist, cost, acc, path = dtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))
        # dist, path = fdtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))
        neighbours.append([int(row[0]), dist])

        testrun += 1
        if testrun==10:break

    neighbours = np.asarray(neighbours)
    neighbours = neighbours[neighbours[:, 1].argsort()][:5]
    print neighbours
    gmap = gmplot.GoogleMapPlotter(tlat[0], tlong[0], 10, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
    gmap.plot(tlat, tlong, 'green', edge_width=5)
    gmap.draw('dtwMaps/testTrip' + str(i+1) + '/test-' + str(i+1) + '.html')
    print "Test Trip ", i,"\n"
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
            gmap.draw('dtwMaps/testTrip' + str(i+1) + '/neighbour' + str(n+1) + '-' + str(grow[1]) + '.html')
            print "Neighbor ",n," \nJP_ID: ",grow[1]," \nDTW:",neighbours[n][1],"\n"

########################################################################################################################
# find . -name "*.html" -type f -delete

dtw_compute()