import ast
import csv

import config
import pandas as pd
import numpy as np

def preprocessing():

    df = pd.read_csv(config.trainsetPath)
    df = df[pd.notnull(df['journeyPatternId'])]
    tripid=0

    with open('trips.csv', 'wb') as csvfile:
        fieldnames = ['tripId','journeyPatternId','trajectories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        list=[]
        for veh_id in df.vehicleID.unique():
            flag=1
            for i, row in df[df['vehicleID']==veh_id].iterrows():
                if flag:
                    flag=0
                    jid=row['journeyPatternId']

                curjid = row['journeyPatternId']

                if jid!=curjid:
                    writer.writerow({'tripId': tripid , 'journeyPatternId': jid , 'trajectories': list})
                    list=[]
                    list.append([row['timestamp'], row['longitude'], row['latitude']])
                    tripid+=1
                    jid=curjid
                else:
                    list.append([row['timestamp'], row['longitude'], row['latitude']])

            writer.writerow({'tripId': tripid, 'journeyPatternId': jid, 'trajectories': list})
            list=[]
            tripid+=1

def cleandata():
    df = pd.read_csv('trips.csv')
    # for i,row in df.iterrows():
        # print row['timestamp']
        # a=np.array(row['timestamp'])
        # l = re.split('\[\[, \]\*\n',a)

    with open('tripsClean.csv', 'wb') as csvfile:

        fieldnames = ['tripId','journeyPatternId','trajectories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in df.iterrows():
            maxdist = 0
            totaldist = 0
            trajectories = ast.literal_eval(row[2])
            for j in range(1, len(trajectories)):
                harvdist = haversine_dist(float(trajectories[j-1][1]), float(trajectories[j-1][2]), float(trajectories[j][1]), float(trajectories[j][2]))
                totaldist += harvdist
                if harvdist > maxdist:
                    maxdist = harvdist

            if( maxdist <=2 and totaldist >= 2):
                writer.writerow({'tripId': row[0], 'journeyPatternId': row[1], 'trajectories': row[2]})


cleandata()



