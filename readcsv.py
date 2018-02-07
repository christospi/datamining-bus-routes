import csv
from time import sleep

import config
import pandas as pd
import numpy as np

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
                list.append([[row['timestamp'], row['longitude'], row['latitude']]])
                tripid+=1
                jid=curjid

            else:
                list.append([[row['timestamp'], row['longitude'], row['latitude']]])
        writer.writerow({'tripId': tripid, 'journeyPatternId': jid, 'timestamp': list})




    # if i['vehicleID']==veh_id :
       #     print i
# with open('trips.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['TripId','JourneyPatternId','Timestamp'])


