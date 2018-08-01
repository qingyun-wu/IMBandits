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
	vector = np.array([random() for _ in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm
	return vector


file_address_5 = './flickrEdges.txt'


featureDic = {}
thetaDic = {}
PDic = {}
NodeDegree = {}
dimension = 7
f_write = open('Normalized_edgeFeaturesNUS.txt', 'w')
with open(file_address_5) as f:
	counter = 0
	for line in f:
		if counter >=4:
			data = line.split(' ')
			u = int(data[0])
			v = int(data[1])
			if u not in NodeDegree:
				NodeDegree[u] = 1
			else:
				NodeDegree[u]  +=1
			if v not in NodeDegree:
				NodeDegree[v] = 1
			else:
				NodeDegree[v]  +=1


			#OriginalFeature = np.array(data[3:], dtype=np.float)
			#featureDic[(u,v)] = OriginalFeature/sum(OriginalFeature)
			#thetaDic[(u,v)] = featureUniform(dimension)
			#PDic[(u,v)] = np.dot(thetaDic[(u,v)], featureDic[(u,v)])
			#f_write.write( str(u) + ' ' + str(v) +  +'\n') 
			#print u,v, featureDic[(u,v)]
			
			#print 'maxDegree', max(NodeDegree.iteritems(), key=operator.itemgetter(1))[1]
			#print 'maxDegree', max(NodeDegree.iteritems(), key=operator.itemgetter(1))[1], min(NodeDegree.iteritems(), key=operator.itemgetter(1))[1]
			#print 'AverageDegree', sum(NodeDegree.values())/float(len(NodeDegree))
		counter +=1
print 'Finish Processing, Start dumping'
print 'Total Nodes', len(NodeDegree)
print 'maxDegree', max(NodeDegree.iteritems(), key=operator.itemgetter(1))[1], min(NodeDegree.iteritems(), key=operator.itemgetter(1))[1]
print 'AverageDegree', sum(NodeDegree.values())/float(len(NodeDegree))

ThreeDegree_Counter = 0
TenDegree_Counter = 0
FortyDegree_Counter = 0
FinalNodeList =[]
FinalNodeDegree  = {}
Degree_threshold = 40

for key in NodeDegree:
	if NodeDegree[key] <= 40:
		ThreeDegree_Counter +=1
	elif NodeDegree[key] > 500:
		TenDegree_Counter +=1

	if NodeDegree[key] > Degree_threshold:
		FinalNodeList.append(key)
		FinalNodeDegree[key] = NodeDegree[key]
print ThreeDegree_Counter, TenDegree_Counter, FortyDegree_Counter

print 'Total Nodes', len(FinalNodeList)
print 'maxDegree', max(FinalNodeDegree.iteritems(), key=operator.itemgetter(1))[1], min(FinalNodeDegree.iteritems(), key=operator.itemgetter(1))[1]
print 'AverageDegree', sum(FinalNodeDegree.values())/float(len(FinalNodeDegree))

	

pickle.dump( FinalNodeList, open( './edgeFeaturesFlickr/NodesDegreeLargerthan' + str(Degree_threshold) + '.list', "wb" ))


#pickle.dump( featureDic, open( './edgeFeaturesFlickr/Normalized_edgeFeaturesNUS.dic', "wb" ))
#pickle.dump( thetaDic, open( './edgeFeaturesFlickr/Normalized_edgeThetasNUS.dic', "wb" ))
#pickle.dump( PDic, open( './edgeFeaturesFlickr/Normalized_edgePNUS.dic', "wb" ))

#Generate edge theta

 
#f_write.close()
