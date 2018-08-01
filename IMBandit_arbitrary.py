import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt

from BanditAlgrithms_arbitrary import UCB1Algorithm, eGreedyAlgorithm, UCB1Struct
from IC.IC import runIC, runICmodel
from IC.runIAC	import weightedEp, runIAC, runIACmodel, randomEp, uniformEp
from generalGreedy import generalGreedy
from degreeDiscount import degreeDiscountIAC

class simulateOnlineData:
	def __init__(self, G, oracle, Pp,seed_size, iterations,batchSize):
		self.G = G
		self.oracle = oracle
		self.Pp = Pp
		self.seed_size = seed_size
		self.iterations = iterations
		self.batchSize = batchSize
		self.get_reward = runIACmodel #From IM model, the total number of activated nodes 
		self.plot = True
		return

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, " Elapsed time", datetime.datetime.now() - self.startTime

	def runAlgorithms(self, algorithms):
		# get cotheta for each user
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d') 
		timeRun_Save = datetime.datetime.now().strftime('_%m_%d_%H_%M') 

		
		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}

		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			
		for iter_ in range(self.iterations):
			optS = self.oracle(self.G, self.seed_size, self.Pp)
			optimal_reward, live_nodes, live_edges = self.get_reward(G, optS, Pp)
#			print 'optimal' ,optimal_reward
			for alg_name, alg in algorithms.items(): 
				S, Ep = alg.decide(iter_) #S is the selected seed nodes set.
				reward, live_nodes, live_edges = self.get_reward(G, S, Ep)
#				print reward
				if alg.feedback == 'node' :
					alg.updateParameters(S, live_nodes)
				elif alg.feedback == 'edge':
					alg.updateParameters(S, live_nodes, live_edges)
#					for u in S:
#						for (u, v) in self.G.edges(u):
#							print alg.articles[(u,v)].getProb(iter_)

				AlgRegret[alg_name].append(optimal_reward - reward)
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
			
		if (self.plot==True): 
			# plot the results	
			f, axa = plt.subplots(1, sharex=True)
			for alg_name in algorithms.iterkeys():	
				axa.plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
			axa.legend(loc='upper left',prop={'size':9})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("Regret")
			axa.set_title("Accumulated Regret")
			plt.show()

if __name__ == '__main__':
	import time
	start = time.time()

	# read in graph
	G = nx.DiGraph()
	with open('graphdata/../graphdata/hep.txt') as f:
		n, m = f.readline().split()
		for line in f:
			u, v = map(int, line.split())
			try:
				G[u][v]['weight'] += 1
			except:
				G.add_edge(u,v, weight=1)
	print 'Built graph G'
	print time.time() - start, 's'

#	Pp = weightedEp(G)
	Pp = randomEp(G,0.2)
	#P = nx.DiGraph()
	#p = 0.1


	# # read in T
	# with open('lemma1.txt') as f:
	#	 T = []
	#	 k = int(f.readline())
	#	 for line in f:
	#		 T.append(int(line))
	# print 'Read %s activated nodes' %k
	# print time.time() - start
	# S = [131, 639, 287, 267, 608, 100, 559, 124, 359, 66]
	# k = len(S)
	# T = runIC(G,S)

	# highdegreeS = highdegreeSet(G,T,k)

	# console = []
	iterations = 500
	batchSize = 1
	seed_size = 3
	# oracle = generalGreedy
	oracle = degreeDiscountIAC

	simExperiment = simulateOnlineData(G = G, oracle = oracle, Pp= Pp, seed_size = seed_size, iterations = iterations, batchSize = batchSize)
	algorithms = {}
	algorithms['egreedy'] = eGreedyAlgorithm(G, seed_size, oracle, 0.1)
#	algorithms['egreedy_0.8'] = eGreedyAlgorithm(G, seed_size, oracle, 0.8)
	algorithms['ucb1'] = UCB1Algorithm(G, seed_size, oracle)


	simExperiment.runAlgorithms(algorithms)