# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:03:40 2019

@author: Nur Imtiazul Haque
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import os
from z3 import *
from sklearn import preprocessing

class CreateConstraints:
    def __init__(self, datasetFile)
        # loading dataset
        dataset = pd.read_csv(datasetFile)
        
        # feature count
        number_of_features = dataset.shape[1]-1
        
        # feature
        x= dataset.iloc[:,:-1]
        # label
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
        
        M=[]
        dx=[]
        
        for i in range(number_of_features):
            M.append(Real('M_'+str(i)))
            dx.append(Real('dx_'+str(i)))
        
        
        def line_equation(x,y,x_1,y_1,x_2,y_2):
            return ( (x-x_1)/(x_1-x_2) == (y-y_1)/(y_1-y_2) )
        
        
        def create_constraints( p, q, initial_point, final_point ):
            
            if final_point[1] < initial_point[1]:
                #swap
                temp = initial_point
                initial_point = final_point
                final_point = temp
                
            #print(initial_point,final_point)
            a = -(final_point[1]-initial_point[1])
            b = (final_point[0]-initial_point[0])
            c = -(a *initial_point[0]  + b * initial_point[1])
            
            a=round(a,4)
            b=round(b,4)
            c=round(c,4)
            
            #print(a,b,c)
            
            constraint_1 = a * ( M[p]+dx[p] ) + b * ( M[q]+dx[q] ) + c >= 0       
            constraint_2 = ( M[q]+dx[q] ) < final_point[1]
            constraint_3 = ( M[q]+dx[q] ) >= initial_point[1]
            
            '''
            constraint_1 = a * ( M[p] ) + b * ( M[q] ) + c >= 0       
            constraint_2 = ( M[q] ) < final_point[1]
            constraint_3 = ( M[q] ) >= initial_point[1]
            '''
            
            return constraint_1, constraint_2, constraint_3  
        
        
        state_constraints=[]
        
        for state in range(len(states)):
            final_constraints=[]
            
            for i in range(number_of_features-1):
                for j in range(i+1,number_of_features):       
                    prefix = "Boundaries/"+ str(states[state]) +"/" + str(i) + "_" +str(j)+'/'
                    
                    files=os.listdir(prefix)
                    
                    if(len(files)==0):
                        continue
                    xor_constraints=[]
                    for f in range(len(files)):
                        count=0        
                        
                        points=[]    
                        for point in open(prefix+files[f]):
                            point=point.split()            
                            points.append(tuple((float(point[0]),float(point[1]))))
                        points=np.asarray(points) 
                        
                        constraints=[]
                        for k in range(points.shape[0]-1):
                            constraints.append( create_constraints( i, j, points[k], points[k+1]  ) )
                        
                    
        
                        and_constraints=[]
                        for k in range( len( constraints )):            
                            and_constraints.append( And( constraints[k][0], constraints[k][1], constraints[k][2] ) )
                        
                        if len(and_constraints) == 0:
                            continue
                        
                        current = and_constraints[0]
                        xor_constraint = and_constraints[0]
                        
                        for k in range( len( and_constraints )-1 ):
                            xor_constraint=Xor(current,and_constraints[k+1])
                            current=xor_constraint
                            
                        xor_constraints.append(xor_constraint)    
                    
                    boundary_constraints=[]
                    for f in range( len( files ) ):
                        for point in open( prefix + files[f] ):
                            point=point.split()
                            constraint_1= M[i] == float(point[0])
                            constraint_2= M[j] == float(point[1])
                            boundary_constraints.append(And(constraint_1,constraint_2))
                      
                    or_boundary_constraints = Or(boundary_constraints)
                    or_constraints = Or(xor_constraints)
                    final_constraints.append( Or( or_constraints, or_boundary_constraints ) )
                    #final_constraints.append( or_constraints )
            state_constraints.append( And( final_constraints ) )
        
        #print(state_constraints)
        
        print(len(state_constraints))
        
        
        
        n_sat=0
        n_unsat=0
        
        
        label = 57
        threshold = 0
        X = dataset.iloc[y == label ].values
        
        for i in range(10):
            s=Solver()
            s.add(state_constraints[57])
            for j in range(number_of_features):
                s.add( M[j] == X[i][j] )
                s.add( dx[j] / X[i][j] <=  ( threshold / 100 ) )
                s.add( dx[j] / X[i][j] >=  -( threshold / 100 ) )
            if (s.check() == sat):
                print(i, X[i], s.model())
            if s.check()==sat:
                n_sat+=1
            else:
                n_unsat+=1
            
        print("SAT: ",n_sat)
        print("UNSAT ",n_unsat)   
        
        
        
        def toFloat(str):
          return float(Fraction(str))
        
        
        
        
        sensor_counts = []
        involved_sensor_count = []
        
        
        for threshold in [10,20,30]:
            n_threats = 0
            total = 0
            
            sensor_count = [0]*number_of_features
            counts = []
            
            #file = open("threshold_not_constrained.txt", "w")
            
            for label in range(len(states)):
                X = dataset.loc[dataset['Alarms'] == states[label]].values
                for target in range(len(states)):
                    if target != label:
                        s=Solver()
                        s.add(state_constraints[target])
                        for j in range(number_of_features):
                            s.add( M[j] == X[0][j] )
                            s.add( dx[j] / X[i][j] <=  ( threshold / 100 ) )
                            s.add( dx[j] / X[i][j] >=  -( threshold / 100 ) )
                        
                        base=Solver()
                        base.add(state_constraints[target])
                        for j in range(number_of_features):
                            base.add( M[j] == X[0][j] )
                            base.add( dx[j] / X[i][j] == 0 )
                       
                        total += 1 
                        if base.check()== unsat and s.check() == sat:
                            #content = "Threat_Number:" + str(n_threats)+ "\n" + "Label:" + str(label) + "\n" + "Target:" + str(target) + "\n" +  "Current Label:" + states[label] + "\n" + "Target label: " + states[target] + "\n" +  "Current Data" + str(X[0])  + "\n"+ "Model: " + str(s.model()) +"\n\n\n\n\n"
                            #file.write(content)
                            n_threats += 1
                            count = 0
                            for k in range(number_of_features):
                                if( s.model()[dx[k]] != None and toFloat( str(s.model()[dx[k]] ) )  != 0):
                                    print(toFloat( str(s.model()[dx[k]] ) ))
                                    sensor_count[k] += 1
                                    count+=1
                            counts.append(count)
                        print(total, n_threats)
                        
            print("Finally", total, n_threats)  
            sensor_counts.append(sensor_count)
            involved_sensor_count.append(counts)
            
        with open('important_sensors.txt', 'w') as filehandle:
            for listitem in sensor_counts:
                filehandle.write('%s\n' % listitem)
                
            
        from z3 import *
        x, y = Reals('x y')
        g  = Solver()
        g= Or( x == y +3 +4*y )
        
        t = Tactic('split-clause')
        r = t(g)
        #for g in r: 
            print(g)
            
        split_all = Repeat(OrElse(Tactic('split-clause'),
                                  Tactic('skip')))
        print (split_all(g))
            