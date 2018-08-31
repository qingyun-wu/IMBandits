from random import choice, random, sample
import numpy as np
import networkx as nx
import numpy as np
from BanditAlg.BanditAlgorithms_LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import collections

class CLUBArmStruct(LinUCBArmStruct):
	def __init__(self,featureDimension,  lambda_, armID):
		LinUCBArmStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_, armID = armID)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0.1
		self.d = featureDimension
	def updateParameters(self, articlePicked_FeatureVector, click, alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		#alpha_2 = 1
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.ArmTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))
		# if self.CBPrime == 0:
		# 	print self.counter

	def updateParametersofClusters(self,clusters,armID,Graph, arms, sortedArmList):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			armID_GraphIndex = sortedArmList.index(armID)
			if clusters[i] == clusters[armID_GraphIndex]:
				self.CA += float(Graph[armID_GraphIndex, i])*(arms[ sortedArmList[i]  ].A - self.I)
				self.Cb += float(Graph[armID_GraphIndex, i])*arms[sortedArmList[i] ].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector, time):
		mean = np.dot(self.CTheta, article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		if pta > self.pta_max:
			pta = self.pta_max
		return pta

class CLUBAlgorithm:
	def __init__(self, G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, FeatureScaling, feedback = 'edge',  cluster_init="Erdos-Renyi"):
		self.time = 0
		self.G = G
		self.oracle = oracle
		self.seed_size = seed_size
		self.feedback = feedback

		self.dimension = dimension
		self.alpha = alpha
		self.alpha_2 = alpha_2
		self.lambda_ = lambda_
		self.FeatureScaling = FeatureScaling

		self.arms = {}  #Nodes
		self.currentP =nx.DiGraph()
		for u in self.G.nodes():
			for v in self.G[u]:
				self.arms[(u, v)] = CLUBArmStruct(dimension,lambda_, (u, v))
				self.currentP.add_edge(u,v, weight=random())
		n = len(self.arms)
		#print 'usersNum', n
		#print len(self.users.keys()), type(self.users.keys())
		self.armIDSortedList = list(self.arms.keys())
		self.armIDSortedList.sort()
		#print len(self.userIDSortedList)
		self.SortedArms = collections.OrderedDict(sorted(self.arms.items()))

		if (cluster_init=="Erdos-Renyi"):
			p = 3*math.log(n)/n
			self.Graph = np.random.choice([0, 1], size=(n,n), p=[1-p, p])
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
			self.clusters = []
		else:
			self.Graph = np.ones([n,n]) 
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
			self.clusters = []
			
	def decide(self, feature_vec):
		self.time +=1
		N_components, component_list = connected_components(csr_matrix(self.Graph))
		print('N_components:',N_components)
		# print 'End connected component'
		self.clusters = component_list
		for (u, v) in self.G.edges():				
			self.SortedArms[(u, v)].updateParametersofClusters(self.clusters, (u,v), self.Graph, self.SortedArms, self.armIDSortedList)
			self.currentP[u][v]['weight']  = self.SortedArms[(u, v)].getProb(self.alpha, feature_vec, self.time)
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges, feature_vec):
		for u in S:
			for (u, v) in self.G.edges(u):
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.SortedArms[(u, v)].updateParameters(feature_vec, reward, self.alpha_2)
				self.updateGraphClusters((u, v), 'False')
		# print 'Start connected component'
		
	def updateGraphClusters(self, armID, binaryRatio):
		n = len(self.SortedArms)
		for j in self.SortedArms:
			# print self.SortedUsers[userID].CBPrime, self.SortedUsers[j].CBPrime
			ratio = float(np.linalg.norm(self.SortedArms[armID].ArmTheta - self.SortedArms[j].ArmTheta,2))/float(self.SortedArms[armID].CBPrime + self.SortedArms[j].CBPrime)
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if ratio > 1:
				ratio = 0
			elif binaryRatio == 'True':
				ratio = 1
			elif binaryRatio == 'False':
				ratio = 1.0/math.exp(ratio)
			#print 'ratio',ratio
			armID_GraphIndex = self.armIDSortedList.index(armID)
			j_GraphIndex = self.armIDSortedList.index(j)
			self.Graph[armID_GraphIndex][j_GraphIndex] = ratio
			self.Graph[j_GraphIndex][armID_GraphIndex] = self.Graph[armID_GraphIndex][j_GraphIndex]
		# print 'N_components:',N_components
		return
	def getLearntParameters(self, armID):
		return self.arms[armID].ArmTheta
	def getP(self):
		return self.currentP