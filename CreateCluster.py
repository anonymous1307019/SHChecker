import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import re
import os
import csv
from sklearn import preprocessing
import time


tic = time.perf_counter()


# loading dataset
dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(columns="Unnamed: 0")
number_of_features=dataset.shape[1]-1


X= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

count_values = y.value_counts()

## parameters
eps=0.5
minPts=5

states = []
for state in np.unique(y):
    if y.value_counts()[state] >= 20:
        states.append(state)


y= preprocessing.LabelEncoder().fit_transform( y )

os.mkdir("Clusters")
for state in range(len(states)):
    os.mkdir("Clusters/"+ str(states[state]))

count = 0

for state in range(len(states)):
    for i in range(number_of_features-1):
        for j in range(i+1,number_of_features):
            X=dataset.iloc[y==state,np.r_[i,j]].values
            remove_duplicate = [] 
            [remove_duplicate.append(x) for x in X.tolist() if x not in remove_duplicate]
            X=remove_duplicate
            dir=os.mkdir("Clusters/"+ str(states[state])+"/"+str(i) + "_" +str(j)+'/')
            db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
            labels = db.labels_
            
            # Number of clusters in labels, ignoring noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            with open('info.csv', mode='a', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(states[state]),str(i), str(j), str(eps), str(minPts), str(n_clusters), str(n_noise)])
            
            for cluster in range(n_clusters):
                points = []
                for k in range(len(labels)):
                    if labels[k]==cluster:
                        points.append(X[k])
                points=np.array(points)
                
                if (len(points>0)):
                    f = open("Clusters/"+ str(states[state])+"/"+str(i) + "_" +str(j)+'/'+str(cluster)+".txt", "w")
                    for data in points:
                        count += 1
                        f.write(str(data[0])+' '+str(data[1])+'\n')
                    f.close()
    print("State "+str(state)+" done")


toc = time.perf_counter()
print(f"Clustering took {toc - tic:0.4f} seconds")
