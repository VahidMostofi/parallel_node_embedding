#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sys

# In[3]:

if len(sys.argv) < 3:
    print("first arg: workdir, second arg: path to embeddings")
    sys.exit(1)

#work_dir = '../data/wikipedia_squ/'
#embedding_path = '/home/vahid/data/wikipedia_squ_emb_3_3.emb'
work_dir = sys.argv[1]
embedding_path = sys.argv[2]

if work_dir[-1] != "/":
    work_dir += "/"

# In[4]:


with open(work_dir + 'train_edges_true.txt') as f:
    train_edges_true = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
train_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in train_edges_true]

with open(work_dir + 'test_edges_true.txt') as f:
    test_edges_true = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
test_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in test_edges_true]

with open(work_dir + 'train_edges_false.txt') as f:
    train_edges_false = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
train_edges_false = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in train_edges_false]

with open(work_dir + 'test_edges_false.txt') as f:
    test_edges_false = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
test_edges_false = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in test_edges_false]


# In[5]:


dic = {}
with open(embedding_path) as f:
    for line in f.read().split('\n')[1:]:
        key = line.split(' ')[0]
        value = np.array([float(x) for x in line.split(' ')[1:]])
        dic[key] = value
print(len(dic))


# In[6]:


train_X = []
train_y = []
test_X = []
test_y = []
for e in train_edges_true:
    train_X.append(np.multiply(dic[str(e[0])], dic[str(e[1])]))
    train_y.append(1)
for e in train_edges_false:
    train_X.append(np.multiply(dic[str(e[0])], dic[str(e[1])]))
    train_y.append(0)
    
for e in test_edges_true:
    test_X.append(np.multiply(dic[str(e[0])], dic[str(e[1])]))
    test_y.append(1)
for e in test_edges_false:
    test_X.append(np.multiply(dic[str(e[0])], dic[str(e[1])]))
    test_y.append(0)
train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)
print(train_X.shape,  train_y.shape, test_X.shape, test_y.shape)


# In[7]:


edge_classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=250)
edge_classifier.fit(train_X, train_y)


# In[8]:


preds = edge_classifier.predict_proba(test_X)[:, 1]


# In[9]:


test_roc = roc_auc_score(test_y, preds)
test_ap = average_precision_score(test_y, preds)
print(test_roc, test_ap)

