import dtw as dtw
import pandas as pd
import numpy as np
import config
import ast
import preprocess as prep
import fastdtw as fdtw


########################################################################################################################
#                                                   DTW Function                                                       #
########################################################################################################################
def dtw_compute():

    df = pd.read_csv('tripsClean.csv')
    df_test = pd.read_csv(config.testseta1Path, delimiter='/')

    # Iterate over test_a1 rows
    for i, t_row in df_test.iterrows():
        test_list = []
        neighbours = []
        sfa = 0

        t_trajectories = ast.literal_eval(t_row[0])

        # Collect in list all the trajectories for this trip
        for j in range(0, len(t_trajectories)):
            test_list.append( [float(t_trajectories[j][1]), float(t_trajectories[j][2])] )

        # Iterate over all trips in tripsClean
        for k, row in df.iterrows():
            clean_list = []
            trajectories = ast.literal_eval(row[2])
            for l in range(0, len(trajectories)):
                clean_list.append([float(trajectories[l][1]), float(trajectories[l][2])])

            # Compute DTW for these trips using Haversine as distance metric
            dist, cost, acc, path = dtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))
            # dist, path = fdtw.fastdtw(test_list, clean_list, dist=lambda c1, c2: prep.haversine_dist(c1[0], c1[1], c2[0], c2[1]))
            neighbours.append([row[0], dist])
            sfa+=1
            if sfa == 10:break

        neighbours = np.asarray(neighbours)
        neighbours = neighbours[neighbours[:, 1].argsort()]
        # print neighbours
        # print neighbours





########################################################################################################################

dtw_compute()