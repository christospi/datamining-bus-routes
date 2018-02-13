"""
BigDataMining - Part 1
(A) - Data preprocessing
(B) - Data clean-up
(C) - Data visualization through gmplot
"""

from math import radians, sin, cos, asin, sqrt
import gmplot as gmplot
import config
import pandas as pd
import ast
import csv

########################################################################################################################
#                                       Harvesine Distance Function                                                    #
########################################################################################################################
def haversine_dist(long1, lat1, long2, lat2):
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    difflong = long2 - long1
    difflat = lat2 - lat1
    a = sin(difflat/2)**2 + cos(lat1) * cos(lat2) * sin(difflong/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Earth's radius in km
    return c * r

########################################################################################################################
#                                           Preprocessing Function                                                     #
########################################################################################################################
def preprocessing():
    df = pd.read_csv(config.trainsetPath)
    df = df[pd.notnull(df['journeyPatternId'])] # Removes rows with null values

    tripid=0
    with open('trips.csv', 'wb') as csvfile:
        fieldnames = ['tripId','journeyPatternId','trajectories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        jlist=[]
        for veh_id in df.vehicleID.unique():
            flag=1
            for i, row in df[df['vehicleID']==veh_id].iterrows():
                if flag:
                    flag=0
                    jid=row['journeyPatternId']

                curjid = row['journeyPatternId']
                if jid!=curjid:
                    writer.writerow({'tripId': tripid , 'journeyPatternId': jid , 'trajectories': jlist})
                    jlist=[]
                    jlist.append([row['timestamp'], row['longitude'], row['latitude']])
                    tripid+=1
                    jid=curjid

                else:
                    jlist.append([row['timestamp'], row['longitude'], row['latitude']])
            writer.writerow({'tripId': tripid, 'journeyPatternId': jid, 'trajectories': jlist})
            jlist=[]
            tripid+=1

########################################################################################################################
#                                           Cleaning Data Function                                                     #
########################################################################################################################
def cleandata():
    maxfails = 0    # Fails due to max distance
    totalfails = 0  # Fails due to total distance

    df = pd.read_csv('trips.csv')

    with open('tripsClean.csv', 'wb') as csvfile:

        fieldnames = ['tripId','journeyPatternId','trajectories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i,row in df.iterrows():
            maxdist = 0
            totaldist = 0
            trajectories = ast.literal_eval(row[2])
            for j in range(1, len(trajectories)):

                # Compute distance between the two points
                harvdist = haversine_dist(float(trajectories[j-1][1]), float(trajectories[j-1][2]), float(trajectories[j][1]), float(trajectories[j][2]))

                totaldist += harvdist
                if harvdist > maxdist:
                    maxdist = harvdist

            if maxdist > 2: maxfails+=1
            if totaldist < 2: totalfails+=1

            if( maxdist <=2 and totaldist >= 2):
                writer.writerow({'tripId': row[0], 'journeyPatternId': row[1], 'trajectories': row[2]})

        print ("Total Fails: %d (MaxDistance) | %d (TotalDistance)" % (maxfails,totalfails))

########################################################################################################################
#                                               Plot Data Function                                                     #
########################################################################################################################
def plot_data():
    df = pd.read_csv('tripsClean.csv')
    plotcount = 0

    for jid in df.journeyPatternId.unique()[::66]:
        for i, row in df[df['journeyPatternId'] == jid].iterrows():
            trajectory = ast.literal_eval(row[2])
            longlist = []
            latlist = []

            for j in range(0, len(trajectory)):
                longlist.append(float(trajectory[j][1]))
                latlist.append(float(trajectory[j][2]))

            gmap = gmplot.GoogleMapPlotter(latlist[0], longlist[0], 12, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
            gmap.plot(latlist, longlist, 'green', edge_width=5)
            gmap.draw('Maps/gmplotMaps/map-tripID' + str(i) + '.html')
            print jid
            break

        plotcount += 1
        if plotcount == 5: break

########################################################################################################################

# preprocessing()
# cleandata()
# plot_data()
