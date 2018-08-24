# IMBandits

### Result

#### Parameter

```python

graph_address = './datasets/Flickr/Small_Final_SubG.G'
prob_address = './datasets/Flickr/Small_Final_Edge_P_Uniform_dim4'
feature_address = './datasets/Flickr/Small_Final_Normalized_edgeFeatures_uniform_dim4.dic'

dataset = 'Flickr' #Choose from 'default', 'NetHEPT', 'Flickr'
FeatureScaling = 1.0
batchSize = 1
alpha = 0.1
alpha_2 = 0.1 
lambda_ = 0.3
gamma = 0.1
dimension = 4
seed_size = 40
iterations = 300

oracle = degreeDiscountIAC
```

#### Experiment

```
average reward for oracle: 2013.1888888888889
LinUCB: 1967.69
Uniform_LinUCB: 1958.30
UCB1: 1934.53
egreedy_0: 1997.54
egreedy_0.1: 1934.25
egreedy_1: 1512.52
CLUB_Erodos: 1957.71
CLUB_1: 1956.60
```

![alt text](https://raw.githubusercontent.com/Elitack/IMBandits/master/pic/avgReward1.PNG)

![alt text](https://raw.githubusercontent.com/Elitack/IMBandits/master/pic/acuReward1.PNG)

#### Parameter

```python
graph_address = './datasets/Flickr/Small_Final_SubG.G'
prob_address = './datasets/Flickr/Small_Final_Edge_P_Uniform_dim4'
feature_address = './datasets/Flickr/Small_Final_Normalized_edgeFeatures_uniform_dim4.dic'

dataset = 'Flickr' #Choose from 'default', 'NetHEPT', 'Flickr'
FeatureScaling = 1.0
batchSize = 1
alpha = 0.1
alpha_2 = 0.1 
lambda_ = 0.3
gamma = 0.1
dimension = 4
seed_size = 40
iterations = 300

oracle = degreeDiscountIAC2
```

#### Experiment

```
average reward for oracle: 1954.6811111111112
LinUCB: 1907.56
Uniform_LinUCB: 1733.87
UCB1: 1729.36
egreedy_0: 1952.58
egreedy_0.1: 1924.06
egreedy_1: 1510.79
CLUB_Erodos: 1881.93
CLUB_1: 1874.47
```

![alt text](https://raw.githubusercontent.com/Elitack/IMBandits/master/pic/avgReward2.PNG)

![alt text](https://raw.githubusercontent.com/Elitack/IMBandits/master/pic/acuReward2.PNG)