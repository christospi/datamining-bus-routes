import ast
import csv
import re
from math import radians, sin, cos, asin, sqrt

import config
import pandas as pd


def haversine_dist(long1, lat1, long2, lat2):
    long1,long2,lat1,lat2 = map(radians, [long1, lat1, long2, lat2])  #decimal to radians

    difflong = long2-long1
    difflat = lat2-lat1
    a = sin(difflat/2)**2 + cos(lat1) * cos(lat2) * sin(difflong/2)**2
    c = 2 * asin(sqrt(a))
    rad = 6371 #earth's radius in km
    return c * rad

def preprocessing():
    df = pd.read_csv(config.trainsetPath)
    df = df[pd.notnull(df['journeyPatternId'])]

    tripid=0
    with open('trips.csv', 'wb') as csvfile:
        fieldnames = ['tripId','journeyPatternId','timestamp']
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

                curjid = row['journeyPatternId']
                if jid!=curjid:
                    writer.writerow({'tripId': tripid , 'journeyPatternId': jid , 'timestamp': list})
                    list=[]
                    list.append([row['timestamp'], row['longitude'], row['latitude']])
                    tripid+=1
                    jid=curjid

                else:
                    list.append([row['timestamp'], row['longitude'], row['latitude']])
            writer.writerow({'tripId': tripid, 'journeyPatternId': jid, 'timestamp': list})
            list=[]
            tripid+=1

def cleandata():
    df = pd.read_csv('trips.csv')
    # for i,row in df.iterrows():
        # print row['timestamp']
        # a=np.array(row['timestamp'])
        # l = re.split('\[\[, \]\*\n',a)

    for i, row in df.iterrows():

        trajectories = ast.literal_eval(row[2])
        for j in range(len(trajectories) - 1):
            print("lat: %25.20f" % float(trajectories[j][1]))
            print("lon: %25.20f" % float(trajectories[j][2]))


cleandata()



