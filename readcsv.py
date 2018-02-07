import csv

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\kwnst\\Desktop\gouno\\train_set.csv')
df = df[pd.notnull(df['journeyPatternId'])]
tripid=0
with open('trips.csv', 'w') as csvfile:
    fieldnames = ['tripId','journeyPatternId','timestamp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    jid=0
    for veh_id in df['vehicleID']:
        if tripid>5: break
        for i, row in df.iterrows():
            list=[]
            list.append([[row['timestamp'], row['longitude'], row['latitude']]])
            if row['vehicleID']==veh_id:
                curjid = row['journeyPatternId']
                print curjid
                if jid!=curjid:
                    writer.writerow({'tripId': tripid , 'journeyPatternId': curjid , 'timestamp': list})
                    list=[]
                    tripid+=1
                    print tripid
                    jid=curjid
                    break
                else:
                    list.append([[row['timestamp'], row['longitude'], row['latitude']]])



    # if i['vehicleID']==veh_id :
       #     print i
# with open('trips.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['TripId','JourneyPatternId','Timestamp'])


