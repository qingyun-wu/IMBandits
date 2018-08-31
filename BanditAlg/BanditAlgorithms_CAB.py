import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import datetime
import os.path
from conf import save_address
from sklearn import linear_model
from random import choice, random, sample
import networkx as nx
import numpy as np
from BanditAlg.BanditAlgorithms_LinUCB import *
import collections
class CABArmStruct(LinUCBArmStruct):
    def __init__(self, featureDimension,  lambda_, armID):
        LinUCBArmStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_, armID = armID)
        self.reward = 0
        self.I = lambda_*np.identity(n = featureDimension)  
        self.counter = 0
        self.CBPrime = 0
        self.CoTheta= np.zeros(featureDimension)
        self.d = featureDimension
        self.ID = armID
        self.cluster = []
    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b +=  articlePicked_FeatureVector*click
        self.AInv = np.linalg.inv(self.A)
        self.ArmTheta = np.dot(self.AInv, self.b)
        self.counter+=1
        #print(self.CoTheta)
    def getCBP(self, alpha, article_FeatureVector,time):
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
        pta = alpha * var#*np.sqrt(math.log10(time+1))
        return pta

class CABAlgorithm():
    def __init__(self, G, seed_size, oracle, dimension, alpha,  alpha_2, lambda_, FeatureDic, FeatureScaling, gamma):
        self.time = 0
        self.G = G
        self.oracle = oracle
        self.seed_size = seed_size
        self.dimension = dimension
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.lambda_ = lambda_
        self.gamma = gamma
        self.FeatureDic = FeatureDic
        self.FeatureScaling = FeatureScaling
        self.feedback = 'edge'
        self.arms = {}  #Nodes
        self.currentP =nx.DiGraph()
        for u in self.G.nodes():
            for v in self.G[u]:
                self.arms[(u, v)] = CABArmStruct(dimension, lambda_, (u, v))
                self.currentP.add_edge(u,v, weight=random())
        n = len(self.arms)

        self.armIDSortedList = list(self.arms.keys())
        self.armIDSortedList.sort()
        self.SortedArms = collections.OrderedDict(sorted(self.arms.items()))

        self.a=0

    def decide(self, feature_vec):
        self.time +=1
        self.updateGraphClusters(feature_vec)
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S

    def updateGraphClusters(self, feature_vec):
        for i in range(len(self.armIDSortedList)):
            id_i = self.armIDSortedList[i]
            WI = self.arms[id_i].ArmTheta
            clusterItem=[]
            CBI = self.arms[id_i].getCBP(self.alpha, feature_vec, self.time)
            WJTotal=np.zeros(WI.shape)
            CBJTotal=0.0
            for j in range(len(self.arms)):
                id_j = self.armIDSortedList[j]
                WJ = self.arms[id_j].ArmTheta
                CBJ = self.arms[id_j].getCBP(self.alpha, feature_vec, self.time)
                compare= np.dot(WI, feature_vec) - np.dot(WJ, feature_vec)               
                if (j != i):
                    if (abs(compare) <= CBI + CBJ):
                        clusterItem.append(self.arms[id_j])
                        WJTotal += WJ
                        CBJTotal += CBJ
                else:    
                    clusterItem.append(self.arms[id_j])
                    WJTotal += WI
                    CBJTotal += CBI
            CW= WJTotal/len(clusterItem)
            CB= CBJTotal/len(clusterItem)
            x_pta = np.dot(CW, feature_vec) + CB
            if x_pta > 1:
                x_pta = 1
            self.currentP[id_i]['weight']  = x_pta
            self.arms[id_i].cluster = clusterItem

    def updateParameters(self, S, live_nodes, live_edges, feature_vec):
        gamma = self.gamma
        for u in S:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    reward = live_edges[(u,v)]
                else:
                    reward = 0
                if (self.arms[(u, v)].getCBP(self.alpha, feature_vec, self.time) >= gamma):
                    self.arms[(u, v)].updateParameters(feature_vec, reward)
                else:
                    clusterItem = self.arms[(u, v)].cluster
                    for i in range(len(clusterItem)):
                        if(clusterItem[i].getCBP(self.alpha, feature_vec, self.time) < gamma):
                            clusterItem[i].updateParameters(feature_vec, reward)

    def getLearntParameters(self, armID):
        return self.arms[armID].ArmTheta
        
    def getP(self):
        return self.currentP