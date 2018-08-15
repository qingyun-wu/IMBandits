import numpy as np
from LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import datetime
import os.path
from conf import sim_files_folder, save_address
from sklearn import linear_model
class CABUserStruct(LinUCBUserStruct):
    def __init__(self,featureDimension,  lambda_, userID):
        LinUCBUserStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_)
        self.reward = 0
        self.I = lambda_*np.identity(n = featureDimension)  
        self.counter = 0
        self.CBPrime = 0
        self.CoTheta= np.zeros(featureDimension)
        self.d = featureDimension
        self.ID = userID
    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b +=  articlePicked_FeatureVector*click
        self.AInv = np.linalg.inv(self.A)
        self.CoTheta = np.dot(self.AInv, self.b)
        self.counter+=1
        #print(self.CoTheta)
    def getCBP(self, alpha, article_FeatureVector,time):
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
        pta = alpha * var#*np.sqrt(math.log10(time+1))
        return pta

class CABOriginalAlgorithm():
    def __init__(self,dimension,alpha,lambda_,n,alpha_2):
        self.time = 0
        #N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
        self.users = []
        #self.gamma=gamma
        #algorithm have n users, each user has a user structure
        self.userNum=n
        for i in range(n):
            self.users.append(CABUserStruct(dimension,lambda_, i)) 

        self.dimension = dimension
        self.alpha = alpha
        self.CanEstimateCoUserPreference = True
        self.CanEstimateUserPreference = False
        self.CanEstimateW = False
        self.CanEstimateV = False
        self.cluster=[]
        self.a=0
        self.startTime = datetime.datetime.now()
        timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
        self.filenameWritePara=os.path.join(save_address, str(self.alpha)+'_'+'dynamic'+timeRun+'.cluster')
    def decide(self,pool_articles,userID):
        
        maxPTA = float('-inf')
        articlePicked = None
        WI=self.users[userID].CoTheta
        for k in pool_articles:
            clusterItem=[]
            featureVector = k.contextFeatureVector[:self.dimension]
            CBI=self.users[userID].getCBP(self.alpha,featureVector,self.time)
            temp=np.zeros((self.dimension,))
            WJTotal=np.zeros((self.dimension,))
            CBJTotal=0.0
            for j in range(len(self.users)):
                WJ=self.users[j].CoTheta
                CBJ=self.users[j].getCBP(self.alpha,featureVector,self.time)
                compare= np.dot(WI,featureVector)-np.dot(WJ,featureVector)
                
                #rwd2=np.dot(self.users[j].CoTheta,featureVector)
                #diffR=abs(rwdi-rwdj)
                
                if(j!=userID):
                    
                    #print(diffR-abs(rwd1-rwd2))
                    if (abs(compare)<=CBJ+CBJ)&(self.users[j].CoTheta!=temp).all():
                        clusterItem.append(self.users[j])
                        WJTotal+=WJ
                        CBJTotal+=CBJ
                else: 
                
                    clusterItem.append(self.users[userID])
                    WJTotal+=WI
                    CBJTotal+=CBI
            CW= WJTotal/len(clusterItem)
            CB= CBJTotal/len(clusterItem)
            #CW=WI
            #CB=CBI
            x_pta = np.dot(CW,featureVector)+CB
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = k.id
                featureVectorPicked = k.contextFeatureVector[:self.dimension]
                picked = k
                maxPTA = x_pta
                self.cluster=clusterItem


        return picked
        
            
    def updateParameters(self, articlePicked, click,userID, gamma):
        #self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
        gamma = 0.1
        if(self.users[userID].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time)>=gamma):
            self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
        else:
            
        
            for i in range(len(self.cluster)):
                if(self.cluster[i].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time)<gamma/4):
                    self.cluster[i].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
                    self.a +=1
                clusterNow.append(self.cluster[i].ID)
            #clusterNow.append(userID)
            #clusterNow.append(articlePicked.id)
            if (clusterNow!=[]):
                clusterNow.append(self.a)
                clusterNow.append(userID)
                # print(clusterNow)
        
        with open(self.filenameWritePara, 'a+') as f:
            if(self.cluster!=[]):
                f.write(str(self.time/self.userNum))
                for i in range(len(self.cluster)):
                    f.write('\t'+str(self.cluster[i].ID))
                f.write('\n')
            

        self.a=0
        self.time +=1
        
                    
                    
                    
    def getCoTheta(self, userID):
        return self.users[userID].CoTheta

    def getVoterDistance(self):
        return [np.array(self.rdiffinside).mean(), np.array(self.rdiffoutside).mean()]

