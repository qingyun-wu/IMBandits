import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt
import time
import pickle
from conf import *

def gen_P(iteration):
	TopicList = pickle.load(open(topic_address , "rb" ))
	EdgeFeatureDic = pickle.load(open(feature_address, "rb" ))
	dimension = 4
	PDic = {}

	for key in EdgeFeatureDic:
		PDic[key] = np.dot(EdgeFeatureDic[key], TopicList[iteration])

	return PDic

def featureUniform(dimension, argv = None):
	vector = np.array([random.random() for i in range(dimension)])
	l2_norm = np.linalg.norm(vector, ord =2)
	vector = vector/l2_norm
	return vector

def ReadGraph_Flixster(file_address):
	start = time.time()
	G = nx.DiGraph()
	with open(file_address) as f:
		n, m = f.readline().split(',')
		for line in f:
			u, v = list(map(int, line.split(',')))
			try:
				G[u][v]['weight'] += 1

			except:
				G.add_edge(u,v, weight=1)
	print('Built Flixster graph G', time.time() - start, 's')

	return G

def ReadSmallGraph_Flixster(file_address):
	start = time.time()
	total = 701784775
	G = nx.DiGraph()
	with open(file_address) as f:
		n, m = f.readline().split(',')
		for line in f:
			u, v = list(map(int, line.split(',')))
			if u < int(total) and v < int(total):
				try:
					G[u][v]['weight'] += 1

				except:
					G.add_edge(u,v, weight=1)
	
	print('Built Flixster graph G', time.time() - start, 's')
	return G
	
def ReadGraph_NetHEPT_Epinions(file_address):
	start = time.time()
	G = nx.DiGraph()
	with open(file_address) as f:
		for line in f:
			if line[0] != '#':
				u, v = list(map(int, line.split()))
				try:
					G[u][v]['weight'] += 1
				except:
					G.add_edge(u,v, weight=1)
	print('Built NetHEPT_Epinions graph G', time.time() - start, 's')
	return G

def ReadGraph_Flickr(file_address):
	start = time.time()
	G = nx.DiGraph()
	with open(file_address) as f:
		for line in f:
			if line[0] != '#':
				u, v = list(map(int, line.split(' ')))
				try:
					G[u][v]['weight'] += 1
				except:
					G.add_edge(u,v, weight=1)
	print('Built Flickr graph G', time.time() - start, 's')
	return G




