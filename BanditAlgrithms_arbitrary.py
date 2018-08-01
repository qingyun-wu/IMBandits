from random import choice, random, sample
import numpy as np
import networkx as nx

class ArticleBaseStruct(object):
    def __init__(self, articleID):
        self.articleID = articleID
        self.totalReward = 0.0
        self.numPlayed = 0
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1


class UCB1Struct(ArticleBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            return min(self.totalReward / self.numPlayed + np.sqrt(1.5*np.log(allNumPlayed) / self.numPlayed),0.2)
        
class eGreedyArticleStruct(ArticleBaseStruct):
    def getProb(self):
        if self.numPlayed == 0:
            pta = 0
        else:
            pta = self.totalReward/self.numPlayed
        return pta

class UCB1Algorithm:
    def __init__(self, G, seed_size, oracle, feedback = 'edge'):
        self.G = G
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.articles = {}
        for (u,v) in G.edges():
            self.articles[(u,v)] = UCB1Struct((u,v))

        self.TotalPlayCounter = 0
        
    def decide(self,iter_):
        self.TotalPlayCounter +=1
        self.Ep = {}
        for (u,v) in self.G.edges():
            # print (u,v)
#            self.articles[(u,v)] = UCB1Struct((u,v))
            self.Ep[(u,v)] = self.articles[(u,v)].getProb(iter_)
        S = self.oracle(self.G, self.seed_size, self.Ep)
        return S, self.Ep       
         
    def updateParameters(self, S, live_nodes, live_edges): 
        for u in S:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    self.articles[(u, v)].updateParameters(reward=1)
                else:
                    self.articles[(u, v)].updateParameters(reward=0)


class eGreedyAlgorithm(UCB1Algorithm):
    def __init__(self, G, seed_size, oracle, epsilon, feedback = 'edge'):
        UCB1Algorithm.__init__(self, G, seed_size, oracle, feedback)
#        self.articles = {}
#       for (u,v) in G.edges():
            # print (u,v)
#           self.articles[(u,v)] = eGreedyArticleStruct((u,v))
        self.epsilon = epsilon

    def decide(self,iter_):
        article_Picked = None
        self.Ep = {}
        for (u,v) in self.G.edges():
            self.Ep[(u,v)] = self.articles[(u,v)].getProb(iter_)
        if random() < self.epsilon: # random exploration
            S = sample(list(self.G.nodes()), self.seed_size)
        else:
            S = self.oracle(self.G, self.seed_size, self.Ep)# self.oracle(self.G, self.seed_size, self.articles)
        return S, self.Ep