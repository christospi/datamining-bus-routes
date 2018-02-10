import ast

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
    print "max lon", max
    # lat=[min(lat),max(lat)]
    # lon=[min(lon),max(lon)]
    # gmap = gmplot.GoogleMapPlotter(lat[0], lon[0], 11, 'AIzaSyDf6Dk2_fg0p8XaEhQdFVCXg-AMlm54dAs')
    # gmap.plot(lat, lon, '#0284ff', edge_width=5)
    #
    # gmap.draw('minmax.html')

min_borders()