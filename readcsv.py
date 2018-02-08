import ast
import csv
from math import radians, sin, cos, asin, sqrt

import config
import pandas as pd


def haversine_dist(long1, lat1, long2, lat2):
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    difflong = long2 - long1
    difflat = lat2 - lat1
    a = sin(difflat/2)**2 + cos(lat1) * cos(lat2) * sin(difflong/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Earth's radius in km

    return c * r

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

    with open('tripsClean.csv', 'wb') as csvfile:

        fieldnames = ['tripId','journeyPatternId','trajectories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for  i,row in df.iterrows():
            maxdist = 0
            totaldist = 0
            trajectories = ast.literal_eval(row[2])
            for j in range(1, len(trajectories)):
                harvdist = haversine_dist(float(trajectories[j-1][1]), float(trajectories[j-1][2]), float(trajectories[j][1]), float(trajectories[j][2]))
                print harvdist
                totaldist += harvdist
                if harvdist > maxdist:
                    maxdist = harvdist
            print maxdist
            if( maxdist <=2 and totaldist >= 2):
                writer.writerow({'tripId': row[0], 'journeyPatternId': row[1], 'trajectories': row[2]})


cleandata()

