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

dimension = 7
featureDic = {}
G = pickle.load(open('./Small_Final_SubG.G', 'rb'))
for u in G.nodes():
	for v in G[u]:
		featureDic[(u,v)] = featureUniform(dimension)

pickle.dump( featureDic, open( './Small_Final_Normalized_edgeFeatures_uniform.dic', "wb" ))
