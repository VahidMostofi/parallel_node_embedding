#!/usr/bin/env python
# coding: utf-8

import sys
import json
import time
import random
import argparse
import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser()

parser.add_argument('--batches', dest='batch_count', type=int, required=True, help='number of splits')
parser.add_argument('--dir', dest='working_dir', type=str, required=True, help='path to the directory to store outputs')
parser.add_argument('--combinations', dest='combinations', type=str, required=True, help='how many combinations we use, options: max, k as centeral circle size')
args = parser.parse_args()

details = {}
details["framework"] = "spark"

input_direcitory = args.working_dir
number_of_batches = args.batch_count
details["batch_count"] = number_of_batches
embed_dim = 32
number_of_walks = 10
length_of_walks = 80
node2vec_p = 0.3
node2vec_q = 0.3
input_path = input_direcitory
if input_path[-1] != "/":
    input_path += "/"
dataset_name = input_direcitory.split("/")[-1]
details["dataset_name"] = dataset_name
test_name = str(time.time())[:10]
details["test_name"] = test_name
# In[1]:


import findspark
#findspark.init(spark_path)
findspark.init()
import pyspark

# In[79]:


# import requests 
# def send_info(info):
#     requests.get(url='http://127.0.0.1:8080/' + str(info))
# send_info('test') 


# In[82]:


combinations = []
def generate_combinations(n, click_size):
    g = nx.Graph()
    for i in range(click_size):
        for j in range(i+1, click_size):
            g.add_edge(i,j)

    single_count = int((n-click_size) / click_size) + 1
    for i in range(click_size, n):
        connect_to = int((i-click_size) / single_count)
        g.add_edge(connect_to, i)
    combs = []
    for e in g.edges():
        e = (min(e[0],e[1]),max(e[0],e[1]))
        combs.append(e)
    return combs
details["args.combinations"] = args.combinations
if args.combinations == "max":
    for i in range(number_of_batches):
        for j in range((i+1), number_of_batches):
            combinations.append((i,j))
else:
    combinations = generate_combinations(number_of_batches, int(args.combinations))
print(combinations)
combGraph = nx.Graph()
for e in combinations:
    combGraph.add_edge(e[0],e[1])
print("len(combinations)",len(combinations))
details["combinations"] = combinations
details["number_of_combinations"] = len(combinations)

# In[13]:
app_name = dataset_name + "_" + str(number_of_batches) + "_" + str(len(combinations))
sc = pyspark.SparkContext(appName=app_name)

# In[83]:

train_edge_true = sc.textFile(input_path + "train_edges_true.txt") .map(lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1]))) .map(lambda x: (min(x[0],x[1]), max(x[0],x[1]), 1))
train_edge_false= sc.textFile(input_path + "train_edges_false.txt") .map(lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1]))) .map(lambda x: (min(x[0],x[1]), max(x[0],x[1]), 0))
train_edges = sc.union([train_edge_true, train_edge_false]).persist()
del train_edge_true
del train_edge_false

# In[84]:

nodes = train_edges.flatMap(lambda x: [x[0],x[1]]).distinct().persist()
nodes_count = len(nodes.collect()) #todo optimize


# In[85]:

print('node_count', nodes_count)
details["main_graph_nodes"] = nodes_count
# In[86]:

batch_size = 1 + (nodes_count // number_of_batches)
# print('batch_size', batch_size)
details["batch_size"] = batch_size
# In[87]:

print('batch_size', batch_size)

# In[88]:

node2parition = nodes.map(lambda x: (x, x // batch_size))

# In[89]:


bd_node2partition = sc.broadcast(node2parition.collectAsMap())
def find_partitions(edge):
    def is_in(batch0, batch1, combination):
        if batch0 == combination[0] and batch1 == combination[1]:
            return True
        if batch1 == combination[0] and batch0 == combination[1]:
            return True
        if batch1 == combination[0] and batch0 == combination[0]:
            return True
        if batch1 == combination[1] and batch0 == combination[1]:
            return True
        
    batch0 = bd_node2partition.value[edge[0]]
    batch1 = bd_node2partition.value[edge[1]]
    if batch0 != batch1:
        results = [((min(batch0,batch1), max(batch0, batch1)), edge)]
    else:
        results = []
    for comb in combinations:
        if is_in(batch0, batch1, comb):
            results.append((comb, edge))
    return results


# In[90]:

train_edges2comb = train_edges.flatMap(find_partitions)
train_edges2comb_filtered = train_edges2comb.filter(lambda ee: ee[0] in combinations)

# In[91]:


def embed_edge_list(edge_list):
    nx_g = nx.Graph()
    for e in edge_list:
        if e[2] == 1:
            nx_g.add_edge(e[0],e[1])
    for edge in nx_g.edges():
        nx_g[edge[0]][edge[1]]['weight'] = 1

    G = node2vec.Graph(nx_g, False, node2vec_p, node2vec_q)
    G.preprocess_transition_probs()
    
    walks = G.simulate_walks(number_of_walks, length_of_walks, verbose=False)
    model = learn_embeddings(walks)
    return model

def learn_embeddings(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=embed_dim, window=10, min_count=0, sg=1, workers=2, iter=1)
    return model


# In[92]:


#todo partition?!!
partioned = train_edges2comb_filtered.partitionBy(len(combinations))


# In[93]:


def make_model(x):
    comb,iteratable = x
    
    start_time = time.time()
    count = len(iteratable)
    model = embed_edge_list(iteratable)
    edges_embs = np.zeros([count, embed_dim])
    labels = np.zeros([count])
    alpha_map = {}
    beta_map = {}
    i = 0
    for e in iteratable:
        flag = False
        try:
            emb1 = model.wv.get_vector(str(e[0]))
        except:
            flag = True
            emb1 = np.zeros([embed_dim])
        try:
            emb2 = model.wv.get_vector(str(e[1]))
        except:
            flag = True
            emb2 = np.zeros([embed_dim])
        edges_embs[i,:] = np.multiply(emb1, emb2)
        labels[i] = e[2]
        i +=1
    
    edge_classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=250)
    edge_classifier.fit(edges_embs, labels)
    
    for node, batch in bd_node2partition.value.items():
        #todo any zeros?!
        if batch == comb[0]:
            try:
                emb1 = model.wv.get_vector(str(node))
            except:
                emb1 = np.zeros([embed_dim])
            alpha_map[node] = emb1
        if batch == comb[1]:
            try:
                emb1 = model.wv.get_vector(str(node))
            except:
                emb1 = np.zeros([embed_dim])
            beta_map[node] = emb1

#     send_info('end_make_model_' + str(time.time()) + '_' + str(time.time() - start_time))
    return (comb,(edge_classifier, {'alpha_map':alpha_map, 'beta_map': beta_map, 'alpha': comb[0], 'beta':comb[1]}))


# In[94]:


models = partioned.groupByKey().map(make_model).partitionBy(len(combinations))


# In[95]:


test_edge_true = sc.textFile(input_path + "test_edges_true.txt") .map(lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1]))) .map(lambda x: (min(x[0],x[1]), max(x[0],x[1]), 1))
test_edge_false= sc.textFile(input_path + "test_edges_false.txt") .map(lambda x: (int(x.split(' ')[0]), int(x.split(' ')[1]))) .map(lambda x: (min(x[0],x[1]), max(x[0],x[1]), 0))
test_edges = sc.union([test_edge_true, test_edge_false]).persist()
del test_edge_true
del test_edge_false


# In[98]:


test_edges2comb = test_edges.flatMap(find_partitions)
# test_edges2comb_filtered = test_edges2comb.filter(lambda ee: ee[0] in combinations)\
# test_edges2comb.collect()


# In[99]:


aa = test_edges2comb.map(lambda x: (x[1], x[0], nx.shortest_path(combGraph, x[0][0], x[0][1])))
bb = aa.map(lambda x: (x[0],x[1],len(x[2]), x[2]))
dist2 = bb.filter(lambda x: x[2] == 2)
dist3 = bb.filter(lambda x: x[2] == 3)
dist4 = bb.filter(lambda x: x[2] == 4)
# test_edges2comb.map(lambda x: (x[1], x[0])).collect()


# In[101]:


def min_first(a,b):
    return(min(a,b),max(a,b))

# x:(edge, comb, length, _)
def f_d2(x):
    edge, comb, length, path = x
    return (comb, (edge[:2], length, edge))
d2 = dist2.map(f_d2)
def f_d3(x):
    edge, comb, length, path = x
    return [(min_first(path[0],path[1]), (edge[:2], length, (edge[0],(edge[1],path[2], path[1], min_first(path[1],path[2])),edge[2]))),
            (min_first(path[1],path[2]), (edge[:2], length, ((edge[0],path[0], path[1], min_first(path[0],path[1])), edge[1],edge[2])))]
d3 = dist3.flatMap(f_d3)
def f_d4(x):
    edge, comb, length, path = x
    e0 = (edge[0],path[0], path[1], min_first(path[0],path[1]))
    e1 = (edge[1],path[3], path[2], min_first(path[2],path[3]))
    return (min_first(path[1],path[2]), (edge[:2], length, (e0,e1,edge[2])))
d4 = dist4.map(f_d4)


# In[102]:


def the_f(node, current_batch, embding_model):
    find_in = None

    if current_batch == embding_model['alpha']:
        node_vector = embding_model['alpha_map'][node]
        find_in = 'beta_map'
    if current_batch == embding_model['beta']:
        node_vector = embding_model['beta_map'][node]
        find_in = 'alpha_map'
    best_key = None
    distance = 10000
    for key,value in embding_model[find_in].items():
        dis = np.linalg.norm(value - node_vector) #todo other ways to compare?! cosine
        if dis < distance:
            distance = dis
            best_key = key
    return best_key
    
def d4_left_step(data):
    output = []
    model_parts = data[1]
    classifier_model = model_parts[0]
    embding_model = model_parts[1]
    for x in data[0]:
        comb, (edge, length, (e0,e1,label)) = x
        node = e0[0]
        current_batch = e0[1]
        target_batch = e0[2] #todo check this in ifs below
        using_comb = e0[3]
        assert using_comb[0] == embding_model['alpha'] and using_comb[1] == embding_model['beta']
        A = the_f(node, current_batch, embding_model)

        output.append((comb, (edge, length, (A, e1, label))))
        
    return output


def d4_right_step(data):
    output = []
    model_parts = data[1]
    classifier_model = model_parts[0]
    embding_model = model_parts[1]
    for x in data[0]:
        comb, (edge, length, (e0,e1,label)) = x
        node = e1[0]
        current_batch = e1[1]
        target_batch = e1[2]
        using_comb = e1[3]
        assert using_comb[0] == embding_model['alpha'] and using_comb[1] == embding_model['beta']
        B = the_f(node, current_batch, embding_model)

        output.append((comb, (edge, length, (e0, B, label))))
        
    return output

    
d4_left_resolved = d4.map(lambda e: (e[1][2][0][3],e)).partitionBy(len(combinations)).groupByKey(). join(models).mapValues(d4_left_step).flatMap(lambda e: e[1])

d4_resolved = d4_left_resolved.map(lambda e: (e[1][2][1][3],e)).partitionBy(len(combinations)).groupByKey(). join(models).mapValues(d4_right_step).flatMap(lambda e: e[1])


# In[104]:


#
d3_left_resolved = d3.filter(lambda e: type(e[1][2][0]) != int).map(lambda e: (e[1][2][0][3],e)). partitionBy(len(combinations)).groupByKey().join(models).mapValues(d4_left_step).flatMap(lambda e: e[1])

d3_right_resolved= d3.filter(lambda e: type(e[1][2][1]) != int).map(lambda e: (e[1][2][1][3],e)). partitionBy(len(combinations)).groupByKey().join(models).mapValues(d4_right_step).flatMap(lambda e: e[1])



# In[106]:


#todo assert, later!
# print('these 3 must be equal')
# print(len(d3.collect()), len(d3_right_resolved.collect()) + len(d3_left_resolved.collect()))
# print(len(d3_right_resolved.collect()) == len(d3_left_resolved.collect()))
# print(len(d4_resolved.collect()) == len(d4.collect()))


# In[107]:

# if we have maximum number of combinations, we only need have paths of length 2
if len(combinations) == (number_of_batches) * (number_of_batches - 1):
    predictables = d2
else:
    predictables = d2.union(d3_right_resolved).union(d3_left_resolved).union(d4_resolved)

# In[108]:

predictables_grouped = predictables.partitionBy(len(combinations)).groupByKey()

# In[109]:

predictables_models = predictables_grouped.join(models)

# In[110]:

def predict(values):
    test_edges_iter = values[0]
    predictable_model = values[1][0]
    edge_embeddings = values[1][1] # dictionary with keys: alpha_map, beta_map, alpha, beta
    count = len(test_edges_iter)
    edges_embs = np.zeros([count, embed_dim])
    labels = np.zeros([count])
    i = 0
    debug_count = 0
    names = []
    lengths = []
    for ee in test_edges_iter:
        edge = ee[0]
        length = ee[1]
        e = ee[2]
        
        if e[0] in edge_embeddings['alpha_map']:
            emb1 = edge_embeddings['alpha_map'][e[0]]
        elif e[0] in edge_embeddings['beta_map']:
            emb1 = edge_embeddings['beta_map'][e[0]]
            
        if e[1] in edge_embeddings['alpha_map']:
            emb2 = edge_embeddings['alpha_map'][e[1]]
        elif e[1] in edge_embeddings['beta_map']:
            emb2 = edge_embeddings['beta_map'][e[1]]
        
        edges_embs[i,:] = np.multiply(emb1, emb2)
        labels[i] = e[2]
        lengths.append(length)
        names.append(str(edge[0]) + '_' + str(edge[1])) 
        i += 1
#     send_info(str(debug_count) + '/' + str(count))
    preds = predictable_model.predict_proba(edges_embs)[:, 1]
    return (preds, labels, names, lengths)


# In[ ]:


vv = predictables_models.mapValues(predict)
outt = vv.collect()


# In[75]:


preds = []
labels = []
all_names = {}
all_labels = {}
for i in range(0, len(outt)):
    names = outt[i][1][2]
    for idx, name in enumerate(names):
        if name not in all_names:
            all_names[name] = []
        all_names[name].append(outt[i][1][0][idx])
        all_labels[name] = outt[i][1][1][idx]
for name in all_labels.keys():
    preds.append(np.array(all_names[name]).mean())
    labels.append(all_labels[name])
preds = np.array(preds)
labels = np.array(labels)
print(preds.shape, labels.shape)
test_roc = roc_auc_score(labels, preds)
test_ap = average_precision_score(labels, preds)
print('roc_auc_score', test_roc)
print('avg_prc_score', test_ap)
details["test_roc"] = test_roc
details["test_ap"] = test_ap


# In[76]:

details['combinations_info'] = "["

ss = 0
for i in range(len(combinations)):
    ss += len(outt[i][1][0])
    print(outt[i][0], roc_auc_score(outt[i][1][1], outt[i][1][0]), average_precision_score(outt[i][1][1], outt[i][1][0]), len(outt[i][1][0]))
    details['combinations_info'] += str((outt[i][0], roc_auc_score(outt[i][1][1], outt[i][1][0]), average_precision_score(outt[i][1][1], outt[i][1][0]), len(outt[i][1][0]))) + ","
details['combinations_info'] = details['combinations_info'][:-1]
details['combinations_info'] += "]"

print(ss)
details['number_of_evaluated_edges'] = ss

# In[77]:

details["distance_info"] = "["

ss = 0
for l in range(2,5):
    labels = []
    preds = []
    for i in range(len(combinations)):
        for j in range(len(outt[i][1][1])):
            if outt[i][1][3][j] == l:
                labels.append(outt[i][1][1][j])
                preds.append(outt[i][1][0][j])
    if len(labels) > 0:
        print(l, roc_auc_score(labels, preds), average_precision_score(labels, preds), len(labels))
        details["distance_info"] += str((l, roc_auc_score(labels, preds), average_precision_score(labels, preds), len(labels))) + ","
    else:
        print(l, 0)
        details["distance_info"] += str((l,0,0,0)) + ","
print(ss)
details["distance_info"] = details["distance_info"][:-1]
details["distance_info"] += "]"


# In[ ]:

sc.stop()
print('information about the run:')
print('input_directory', input_direcitory)
print('number_of_batches', number_of_batches)
# print('embed_dim', embed_dim)
# print('number_of_walks', number_of_walks)
# print('length_of_walks', length_of_walks)
# print('node2vec_p', node2vec_p)
# print('node2vec_q', node2vec_q)
print('len(combinations)', len(combinations))
print('combinations', combinations)
print('node_count', nodes_count)
print('batch_size', batch_size)

def list_to_string(lisst):
    combinations_string = "["
    for comb in lisst:
        combinations_string += "({},{}),".format(comb[0],comb[1])
    combinations_string = combinations_string[:-1]
    combinations_string += "]"
    return combinations_string
details["combinations"] = list_to_string(details["combinations"])
base_name = details['dataset_name'] + '_' + str(details['batch_count']) + '_' + str(details['number_of_combinations']) + '_'
with open('../results/' + base_name + str(test_name)+'.json', 'w') as fp:
    json.dump(details, fp, indent=4)
