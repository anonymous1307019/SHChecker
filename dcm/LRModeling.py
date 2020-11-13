# importing libraries
import csv
import pandas as pd
import numpy as np
import math
from z3 import *
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from fractions import Fraction
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax

class LRModeling:
  def __init__(self, dataset, input_val, test_size):
    self.dataset = dataset
    self.test_size = test_size
    self.input_val = input_val
    
  # z3-solver returns model output in fraction format
  # converting z3 model outputs into floating point numbers
  def toFloat(self, str):
    return float(Fraction(str))

  # rectified linear units activation function
  def relu(self, val):
      return val * (val > 0)

  def exponential(self, val):
      result = 0
      for i in range(100):
        result = result + val**i / math.factorial(i)   
      return result

  
  def soft_max(self, index, values):
      sum =  0
      for value in values:
        sum =  sum + self.exponential(value)
      return self.exponential(values[index]) / sum

  def lr_preprocessing(self):    
    # features
    self.X = self.dataset.iloc[:,:-1]
    # labels
    self.y= preprocessing.LabelEncoder().fit_transform( self.dataset.iloc[:,-1] )

    # counting input output nodes
    num_inp_nodes = self.X.shape[1]
    num_out_nodes = len(np.unique(self.y))

    # train-test spilitting
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 0)

    self.nodes_in_layers = [num_inp_nodes]
    self.nodes_in_layers.append(num_out_nodes)


  
  def formal_modeling(self):
    self.lr_modeling()

    solver = Solver()
    z3_input = [Real('z3_input_' + str(i)) for i in range(num_inp_nodes)]
    z3_output = [Real('z3_output_' + str(i)) for i in range(num_out_nodes)]
    
    
    ## input and output value constraint of layer 0
    for i in range(len(z3_input[0])): solver.add( z3_input[0][i] == input_val[0][i] )

    for i in range(len(z3_output)):
        arr = []
        for j in range(len(z3_input)):
          arr.append(z3_input[j]) 
        solver.add(z3_output[i] == self.soft_max(i, arr) )

    solver.check()

    final_layer = len(z3_output) - 1
    
    label = -1
    maxm = 0

    final_layer = 1
    # finding argmax and assigning it as a label
    for i in range(len(z3_output[final_layer])):
      # determining current softmax value
      cVal = self.toFloat(str( solver.model()[z3_output[final_layer][i]]))
      if cVal > maxm:
          maxm = cVal
          label = i
      
    # returning the label
    return label