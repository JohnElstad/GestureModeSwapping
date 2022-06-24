## Heatmap creation for use with data collected by heatmap_collection.py
## John Elstad 6/1/2022

## Generates a heat map that shows accuracy of gesture predictions. The heatmap is the area infront of the camera in meters
#The number of boxes in x and z direction can be changed below. The path to the heatmap data file is important to specify as well.
#also make sure to specify which gesture is the correct one being tested.


from zlib import Z_BEST_COMPRESSION
import numpy as np
import csv
from numpy import loadtxt
from numpy import *
import pandas as pd
from scipy import NaN
import seaborn as sns
import matplotlib.pyplot as plt


correctGesture = 4 #change to the correct gesture. ex: back is 2, no gesture is 5, 3 is right,4 = cross arm, 0 = forward, left = 1
nx = 9 #number of bins created by heatmap in x dirction (distance to the side of camera)
nz = 17 #number of bins created by heatmap in z dirction (distance from camera)
path = '/home/user/john_ws/src/gesture_ctrl_john/src/Data/Heatmap3DswapV1Out.csv' #Data import path for data created by heatmap_collection.py
title = 'Outdoor Cross-Arm Heatmap' #title of plot created




sns.set()
dataset = loadtxt(path, delimiter=',')
# print(dataset[1])
print(np.shape(dataset[1]))

df = pd.read_csv(path,header=None).T
print(df.to_string())
df.columns = ['X', 'Z', 'Prediction']

# print(df.head()) 


#read in data from file and separate into bins. Average each bin to get prediction accuracy
binsx = np.linspace(-2,2,nx)
print('binx ' + str(binsx))

binsz = np.linspace(0,8,nz)
print('binz ' + str(binsz))

df['binsx'] = pd.cut(df['X'] , bins=binsx, include_lowest=True,ordered=True)
df['binsz'] = pd.cut(df['Z'] , bins=binsz, include_lowest=True,ordered=True)

df.loc[df.Prediction == correctGesture, 'CorrectPrediction'] = 1
df.loc[df.CorrectPrediction.isnull(), 'CorrectPrediction'] = 0

print(df)
accuracy = df["CorrectPrediction"].mean()
print(f"Mean Accuracy: {accuracy}")

binAvg = df.groupby(['binsx', 'binsz'])['CorrectPrediction'].mean().unstack()
print(binAvg.to_string())
dataFinished = []

#create plot and setup axis/labels
res = sns.heatmap(binAvg, cmap ='RdYlGn', linewidths = 0.30, annot = True)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 8, rotation = 30)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 8)
res.set_xlabel("Z Distance from Camera (m)")
res.set_ylabel("X Distance from Camera (m)")
plt.title(title, fontsize = 18)

plt.show()

