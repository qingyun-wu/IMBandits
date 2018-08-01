import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import operator

def featureUniform(dimension, argv = None):
	vector = np.array([random.random() for i in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm
	return vector

Degree_threshold = 40
NodeList = pickle.load(open( './Small_NodeList.list', "rb" ))
NodeTheta = pickle.load(open( './Final_NodeTheta.dic', "rb" ))

EdgeFeatureDic = pickle.load( open( './Small_Final_Normalized_edgeFeatures_uniform.dic', "rb" ))
print len(EdgeFeatureDic), len(NodeTheta)
dimension = 7
PDic = {}

for key in EdgeFeatureDic:
	PDic[key] = np.dot(EdgeFeatureDic[key], NodeTheta[key[1]])

pickle.dump( PDic, open( './Small_Final_Edge_P_Uniform', "wb" ))
