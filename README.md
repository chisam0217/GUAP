# Graph Universal Attack by Adversarial Patching (GUAP)

## Usage
* PyTorch 0.4 or 0.5 
* Python 2.7 or 3.6
* networkx, scipy, sklearn, numpy, pickle

## Train the attack model 

**Example:** ```python generate_perturbation.py --dataset cora --radius 10 --fake_rate 0.025 --step 10```

*dataset: the network dataset you are going to attack* \
*radius: the radius of the l2 Norm Projection* \
*fake_rate: the ratio of patch nodes to the original graph size* \
*step: the learning step of updating the patch connection entries*


## Evaluate the test ASR
After finishing the training of the GUA, we then evaluate the test asr over the test nodes 

**Example:** ```python eval_baseline.py --dataset cora --radius 10 --fake_rate 0.025 --evaluate_mode universal```

*dataset: the network dataset you are going to attack* \
*radius: the radius of the l2 Norm Projection*
*evaluate_mode* has five values: 
* "universal": guap
* "rand_feat": guap with regenerated node features
* "no_connection": guap without patch connections
* "random_connection": guap with random patch connections
* "full_connection": guap with full patch connections

Some patch results trained by GUAP can be accessed from [Dropbox](https://www.dropbox.com/sh/w6osydcz4y8wkme/AAAE8O_v2kZ1Ojt7m-g7Khi-a?dl=0): \
**Cora**: radius = 10, step = 10, fake_rate=0.01 \
**Citeseer**: radius = 10, step = 10, fake_rate=0.01 \
**Pol.Blogs**: radius = 10, step = 10, fake_rate=0.05 \
You can directly use them for testing the attack performances. 

 


The verision of jupyter notebook is also supported as: evaluate.ipynb

