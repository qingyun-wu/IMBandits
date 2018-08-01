import matplotlib.pyplot as plt
import time
import pickle
import networkx as nx
import random

Degree_threshold = 40
NodeList = pickle.load(open( './NodesDegreeLargerthan' + str(Degree_threshold) + '.list', "rb" ))
print 'Done with loading List'

NodeNum = len(NodeList)
print NodeNum
Small_NodeList = [ NodeList[i] for i in sorted(random.sample(xrange(len(NodeList)), NodeNum/5 )) ]
NodeList = Small_NodeList
print len(NodeList)
pickle.dump( NodeList, open( './Small_NodeList.list', "wb" ))


file_address = './flickrEdges.txt'
start = time.time()
G = nx.DiGraph()
print 'Start Reading'
with open(file_address) as f:
	#n, m = f.readline().split(',')
	for line in f:
		if line[0] != '#':
			u, v = map(int, line.split(' '))
			if u in NodeList and v in NodeList:
				try:
					G[u][v]['weight'] += 1

				except:
					G.add_edge(u,v, weight=1)


print 'Start Dumping'
pickle.dump( G, open( 'Small_Final_SubG.G', "wb" ))
#G = pickle.load(open(file_address + '.G', 'rb'))
#It may takes two minutes
print 'Built Flixster graph G', time.time() - start, 's'