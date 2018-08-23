import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from conf import *

from BanditAlgorithms import UCB1Algorithm, eGreedyAlgorithm 
from BanditAlgorithms_LinUCB import N_LinUCBAlgorithm, LinUCBAlgorithm
from BanditAlgorithms_CLUB import CLUBAlgorithm
from IM_CAB import CABAlgorithm
from IC.IC import runIC, runICmodel, runICmodel_n
from generalGreedy import generalGreedy
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp

class simulateOnlineData:
    def __init__(self, G, P, oracle, seed_size, iterations, batchSize, dataset):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.batchSize = batchSize
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.result_oracle = []

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()
        optS = self.oracle(self.G, self.seed_size, self.TrueP)

        for iter_ in range(self.iterations):
            optimal_reward, live_nodes, live_edges = runICmodel_n(G, optS, self.TrueP)
            self.result_oracle.append(optimal_reward)
            print('oracle', optimal_reward)
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() 
                reward, live_nodes, live_edges = runICmodel_n(G, S, self.TrueP)

                if alg.feedback == 'node':
                    alg.updateParameters(S, live_nodes)
                elif alg.feedback == 'edge':
                    alg.updateParameters(S, live_nodes, live_edges)

                self.AlgReward[alg_name].append(reward)

            self.resultRecord(iter_)

        self.showResult()

    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
            fileSig = '_seedsize'+str(self.seed_size) + '_iter'+str(self.iterations)+'_'+str(self.oracle.__name__)+'_'+self.dataset
            self.filenameWriteReward = os.path.join(save_address, 'AccReward' + timeRun + fileSig + '.csv')

            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n') 
        else:
            # if run in the experiment, save the results
            if iter_ % self.batchSize == 0:
                print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
                self.tim_.append(iter_)
                for alg_name in algorithms.keys():
                    self.BatchCumlateReward[alg_name].append(sum(self.AlgReward[alg_name][-1:]))
                with open(self.filenameWriteReward, 'a+') as f:
                    f.write(str(iter_))
                    f.write(',' + ','.join([str(self.BatchCumlateReward[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')

    def showResult(self):
        print('average reward for oracle:', np.mean(self.result_oracle))
        # plot average reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(tim_, self.BatchCumlateReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Average Reward")
        plt.show()
        # plot accumulated reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            result = [sum(simExperiment.BatchCumlateReward[alg_name][:i]) for i in range(len(tim_))]
            axa.plot(tim_, result, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Accumulated Reward")
        plt.show()

if __name__ == '__main__':
    start = time.time()

    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    feature_dic = pickle.load(open(feature_address, 'rb'), encoding='latin1')

    P = nx.DiGraph()
    for (u,v) in G.edges():
        P.add_edge(u, v, weight=FeatureScaling*prob[(u,v)])
    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)

    simExperiment = simulateOnlineData(G, P, oracle, seed_size, iterations, batchSize, dataset)

    algorithms = {}
    algorithms['LinUCB'] = N_LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_, feature_dic, FeatureScaling)
    algorithms['Uniform_LinUCB'] = LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_, feature_dic)
    algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    # algorithms['egreedy_0'] = eGreedyAlgorithm(G, seed_size, oracle, 0)
    algorithms['egreedy_0.1'] = eGreedyAlgorithm(G, seed_size, oracle, 0.1)
    # algorithms['egreedy_1'] = eGreedyAlgorithm(G, seed_size, oracle, 1.0)
    #algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    algorithms['CLUB_Erodos'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, feature_dic, FeatureScaling)
    # algorithms['CLUB_0.2'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    algorithms['CLUB_1'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, feature_dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    # algorithms['CLUB_4'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, 4.0, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    # algorithms['CAB'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, gamma)

    simExperiment.runAlgorithms(algorithms)