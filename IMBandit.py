import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from conf import *
from Tool.utilFunc import *

from BanditAlg.BanditAlgorithms import UCB1Algorithm, eGreedyAlgorithm 
from BanditAlg.BanditAlgorithms_LinUCB import N_LinUCBAlgorithm, LinUCBAlgorithm
from BanditAlg.BanditAlgorithms_CLUB import CLUBAlgorithm
from BanditAlg.BanditAlgorithms_CAB import CABAlgorithm
from IC.IC import runIC, runICmodel, runICmodel_n
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp

class simulateOnlineData:
    def __init__(self, G, oracle, seed_size, iterations, batchSize, feature_dic, topic_list, dataset):
        self.G = G
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.batchSize = batchSize
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.result_oracle = []
        self.feature_dic = feature_dic
        self.topic_list = topic_list


    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()
        for iter_ in range(self.iterations):
            TrueP = self.get_TrueP(iter_)
            optS = self.oracle(self.G, self.seed_size, TrueP)
            optimal_reward, live_nodes, live_edges = runICmodel_n(G, optS, TrueP)
            self.result_oracle.append(optimal_reward)
            print('oracle', optimal_reward)
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() 
                reward, live_nodes, live_edges = runICmodel_n(G, S, TrueP)

                if alg.feedback == 'node':
                    alg.updateParameters(S, live_nodes, self.topic_list[iter_])
                elif alg.feedback == 'edge':
                    alg.updateParameters(S, live_nodes, live_edges, self.topic_list[iter_])

                self.AlgReward[alg_name].append(reward)

            self.resultRecord(iter_)

        self.showResult()

    def get_TrueP(self, iter_):
        graph_p = nx.DiGraph()
        for key in self.feature_dic:
            prob = np.dot(self.feature_dic[key], self.topic_list[iter_])
            graph_p.add_edge(key[0], key[1], weight=FeatureScaling*prob)
        return graph_p
        
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
            axa.plot(self.tim_, self.BatchCumlateReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Average Reward")
        plt.show()
        plt.savefig('./SimulationResults/AvgReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
        # plot accumulated reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            result = [sum(self.BatchCumlateReward[alg_name][:i]) for i in range(len(self.tim_))]
            axa.plot(self.tim_, result, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Accumulated Reward")
        plt.show()
        plt.savefig('./SimulationResults/AcuReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')

if __name__ == '__main__':
    start = time.time()

    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    feature_dic = pickle.load(open(feature_address, 'rb'), encoding='latin1')
    topic_list = pickle.load(open(topic_address , "rb" ), encoding='latin1')

    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)

    simExperiment = simulateOnlineData(G, oracle, seed_size, iterations, batchSize, feature_dic, topic_list, dataset)

    algorithms = {}
    algorithms['LinUCB'] = N_LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_, FeatureScaling)
    algorithms['Uniform_LinUCB'] = LinUCBAlgorithm(G, seed_size, oracle, dimension, alpha, lambda_)
    algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    # algorithms['egreedy_0'] = eGreedyAlgorithm(G, seed_size, oracle, 0)
    algorithms['egreedy_0.1'] = eGreedyAlgorithm(G, seed_size, oracle, 0.1)
    # algorithms['egreedy_1'] = eGreedyAlgorithm(G, seed_size, oracle, 1.0)
    #algorithms['UCB1'] = UCB1Algorithm(G, seed_size, oracle)
    algorithms['CLUB_Erodos'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, FeatureScaling)
    # algorithms['CLUB_0.2'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    algorithms['CLUB_1'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, FeatureScaling, feedback = 'edge',  cluster_init="none")
    # algorithms['CLUB_4'] = CLUBAlgorithm(G, seed_size, oracle, dimension, alpha, 4.0, lambda_, Feature_Dic, FeatureScaling, feedback = 'edge',  cluster_init="none")
    # algorithms['CAB'] = CABAlgorithm(G, seed_size, oracle, dimension, alpha, alpha_2, lambda_, Feature_Dic, FeatureScaling, gamma)

    simExperiment.runAlgorithms(algorithms)

