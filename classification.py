import ast
import csv

import preprocess as prep

import pandas as pd
from gmplot import gmplot


def min_borders():

    df = pd.read_csv('tripsClean.csv')
    lat = []
    lon = []
    clean_list=[]
    for i, row in df.iterrows():
        trajectories = ast.literal_eval(row[2])
        for j in range(0, len(trajectories)):
            lon.append(float(trajectories[j][1]))
            lat.append(float(trajectories[j][2]))
            clean_list.append([float(trajectories[j][1]), float(trajectories[j][2])])
    print "to min lat einai: ", min(lat)
    print "to min lon einai: ", min(lon)
    return min(lat),min(lon)

def features_extract():
    filename = 'C_tripsClean.csv'
    with open(filename, 'wb') as csvfile:
        fieldnames = ['tripId','journeyPatternId','cells']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        df = pd.read_csv('tripsClean.csv')
        minlat, minlon = min_borders()
        for i, row in df.iterrows():
            c=[]
            trajectories = ast.literal_eval(row[2])

            for j in range(0, len(trajectories)):
                dx,dy=cell_pick(minlon,minlat,trajectories[j][1],trajectories[j][2],0.3)
                c.append("C"+str(int(dx))+","+str(int(dy))+";")
            writer.writerow({'tripId': row[0], 'journeyPatternId': row[1], 'cells':c})


def cell_pick( minlon, minlat, lon, lat, csize):

    distx = prep.haversine_dist(float(minlon), float(lat), float(minlon),float(minlat))
    disty = prep.haversine_dist(float(lon), float(minlat), float(minlon),float(minlat))
    return distx//csize, disty//csize





features_extract()