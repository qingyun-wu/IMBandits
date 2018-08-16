import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt
import time
import pickle 
import os
from conf import save_address

from utilFunc import ReadGraph_NetHEPT_Epinions, ReadGraph_Flixster, ReadGraph_Flickr, ReadSmallGraph_Flixster
from BanditAlgorithms import UCB1Algorithm, eGreedyAlgorithm 
from BanditAlgorithms_LinUCB import N_LinUCBAlgorithm, LinUCBAlgorithm
from BanditAlgorithms_CLUB import CLUBAlgorithm
from IM_CAB import CABAlgorithm
from IC.IC import runIC, runICmodel, runICmodel_n
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp
from generalGreedy import generalGreedy
from degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar

class simulateOnlineData:
    def __init__(self, G, P, oracle, seed_size, iterations, batchSize):
        self.G = G
        #self.P = P
        self.TrueP = P
        self.oracle = oracle
        self.seed_size = seed_size
        self.iterations = iterations
        self.batchSize = batchSize
        self.get_reward = runICmodel #IM model
        self.get_reward_n = runICmodel_n #IM model
        self.plot = True
        return

    def batchRecord(self, iter_):
        print("Iteration %d"%iter_, " Elapsed time", datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms):
        # get cotheta for each user
        self.startTime = datetime.datetime.now()
        timeRun = datetime.datetime.now().strftime('_%m_%d') 
        timeRun_Save = datetime.datetime.now().strftime('_%m_%d_%H_%M') 

        filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun_Save + '.csv')
        filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun_Save + '.csv')
        
        fileSig = '_seedsize'+str(self.seed_size) + '_iter'+str(self.iterations)+'_'+str(self.oracle.__name__)+'_'+dataset
        
        filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun_Save+fileSig + '.csv')
        filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun_Save + '.csv')
        filenameWriteResult = os.path.join(save_address, fileSig + timeRun + '.csv')
        
        tim_ = []
        self.BatchCumlateRegret = {}
        self.AlgRegret = {}

        for alg_name, alg in list(algorithms.items()):
            self.AlgRegret[alg_name] = []
            self.BatchCumlateRegret[alg_name] = []

        with open(filenameWriteRegret, 'w') as f:
            f.write('Time(Iteration)')
            f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
            f.write('\n')

        #optS = self.oracle(self.G, self.seed_size) 
        optS = self.oracle(self.G, self.seed_size, self.TrueP)
        for iter_ in range(self.iterations):

            # optimal_reward, live_nodes, live_edges = self.get_reward(G, optS, self.TrueP)
            optimal_reward, live_nodes, live_edges = self.get_reward_n(G, optS, self.TrueP)
            print('oracle', optS, optimal_reward)

            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() #S is the selected seed nodes set.
                # S2 = alg.decide() #S is the selected seed nodes set.

                # reward, live_nodes, live_edges = self.get_reward(G, S, self.TrueP)
                reward, live_nodes, live_edges = self.get_reward_n(G, S, self.TrueP)
                # reward = self.get_reward_n(G, S, self.TrueP)
                # print alg_name, S, reward
                #print 'live_edges', live_edges
                if alg.feedback == 'node':
                    alg.updateParameters(S, live_nodes)
                elif alg.feedback == 'edge':
                    alg.updateParameters(S, live_nodes, live_edges)

                self.AlgRegret[alg_name].append(reward)
                #AlgRegret[alg_name].append(reward)

                #Calculate P estimation error
                alg_P = alg.getP()
                p_error = 0
                for u in S:
                    for (u, v) in self.G.edges(u):
                        p_error += (alg_P[u][v]['weight'] - self.TrueP[u][v]['weight'])**2
                # print alg_name, p_error

            if iter_%self.batchSize == 0:
                self.batchRecord(iter_)
                tim_.append(iter_)
                for alg_name in algorithms.keys():
                    # BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                    self.BatchCumlateRegret[alg_name].append(sum(self.AlgRegret[alg_name][-1:]))

                with open(filenameWriteRegret, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(self.BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')

        # for (u,v) in self.G.edges():
        #   print 'oracle', u, v, self.TrueP[u][v]['weight']
        #   for alg_name, alg in algorithms.items(): 
        #       print alg_name, u, v,alg.currentP[u][v]['weight']

        

        
        # plot the results  
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(tim_, self.BatchCumlateRegret[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, self.BatchCumlateRegret[alg_name][-1]))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Regret")
        axa.set_title("Accumulated Regret")
        plt.savefig('./SimulationResults/Regret' + str(timeRun_Save )+'.pdf')
        plt.show()

if __name__ == '__main__':
    #Datasets edge file address
    graph30_edgefile_address = './datasets/graph30/graph30.txt'

    NetHEPT_edgefile_address = './datasets/NetHEPT/cit-HepTh.txt'
    Flixster_edgefile_address = './datasets/Flixster-dataset/data/edges.csv'
    Epinions_edgefile_address = './datasets/Epinions/soc-Epinions1.txt'
    Flickr_edgefile_address = './datasets/Flickr/flickrEdges.txt'

    
    start = time.time()
    dataset = 'Flickr' #Choose from 'default', 'NetHEPT', 'Flickr'
    FeatureScaling = 1.0
    batchSize = 1
    alpha = 0.1
    lambda_ = 0.3
    gamma = 0.1
    alpha_2 = 0.1    

    oracle = degreeDiscountIAC2

    if dataset == 'Flickr':
        dimension = 4
        seed_size = 40
        iterations = 60

        G = pickle.load(open('./datasets/Flickr/Small_Final_SubG.G', 'rb'), encoding='latin1')
        print(len(G.nodes()))
        Flickr_Linear_PDic = pickle.load(open('./datasets/Flickr/Small_Final_Edge_P_Uniform_dim4', 'rb'), encoding='latin1')
        print(len(Flickr_Linear_PDic))
        P = nx.DiGraph()
        for (u,v) in G.edges():
            P.add_edge(u,v, weight = FeatureScaling*Flickr_Linear_PDic[(u,v)])

        print(len(G.nodes()), len(G.edges()))
        print(len(P.nodes()), len(P.edges()))
        Feature_Dic = pickle.load(open('./datasets/Flickr/Small_Final_Normalized_edgeFeatures_uniform_dim4.dic', 'rb'), encoding='latin1')
        print('Done with Loading Feature')
    elif dataset == 'NetHEPT':
        dimension = 4
        seed_size = 30
        iterations = 60
        G = ReadGraph_NetHEPT_Epinions(NetHEPT_edgefile_address)
        NetHEPT_Linear_PDic = pickle.load(open('./datasets/NetHEPT/Final_Edge_P_Uniform_dim4', 'rb'), encoding='latin1')
        P = nx.DiGraph()
        for (u,v) in G.edges():
            P.add_edge(u,v, weight = FeatureScaling*NetHEPT_Linear_PDic[(u,v)])

        print(len(G.nodes()), len(G.edges()))
        print(len(P.nodes()), len(P.edges()))
        Feature_Dic = pickle.load(open('./datasets/NetHEPT/Normalized_edgeFeatures_uniform_dim4.dic', 'rb'), encoding='latin1')
        print('Done with Loading Feature')
    elif dataset == 'Epinions':
        G = ReadGraph_NetHEPT_Epinions(NetHEPT_edgefile_address)
        P = weightedEp(G)   
    elif dataset == 'graph30':
        dimension = 4
        seed_size = 3
        iterations = 200
        # FeatureScaling = 0.2
        G = ReadGraph_NetHEPT_Epinions(graph30_edgefile_address)
        graph30_Linear_PDic = pickle.load(open('./datasets/graph30/Final_Edge_P_Uniform', 'rb'), encoding='latin1')
        P = nx.DiGraph()
        for (u,v) in G.edges():
            P.add_edge(u,v, weight = FeatureScaling*graph30_Linear_PDic[(u,v)])

        print(len(G.nodes()), len(G.edges()))
        print(len(P.nodes()), len(P.edges()))
        Feature_Dic = pickle.load(open('./datasets/graph30/Normalized_edgeFeatures_uniform_4.dic', 'rb'), encoding='latin1')
        print('Done with Loading Feature')  
    else:
        G = nx.DiGraph()
        with open('graphdata/../graphdata/hep.txt') as f:
            n, m = f.readline().split()
            for line in f:
                u, v = list(map(int, line.split()))
                try:
                    G[u][v]['weight'] += 1
                except:
                    G.add_edge(u,v, weight=1)
        print('Built graph G')
        print(time.time() - start, 's')
        P = weightedEp(G)
        print(time.time() - start, 'get P')


    simExperiment = simulateOnlineData(G = G, P = P, oracle = oracle, seed_size = seed_size, iterations = iterations, batchSize = batchSize)
    algorithms = {}
    algorithms['LinUCB'] = N_LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_, Feature_Dic, FeatureScaling)
    algorithms['Uniform_LinUCB'] = LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_, Feature_Dic)

    algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    # algorithms['egreedy_0'] = eGreedyAlgorithm(G, seed_size, oracle, 0)
    algorithms['egreedy_0.1'] = eGreedyAlgorithm(G, seed_size, oracle, 0.1)
    # algorithms['egreedy_1'] = eGreedyAlgorithm(G, seed_size, oracle, 1.0)
    #algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    algorithms['CLUB_Erodos'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling)
    # algorithms['CLUB_0.2'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    algorithms['CLUB_1'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    # algorithms['CLUB_4'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, 4.0, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    algorithms['CAB'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, gamma)

    simExperiment.runAlgorithms(algorithms)