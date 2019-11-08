import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import argparse
import sys
import os
import time
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from runnings import run_node2vec_combinations
import json

parser = argparse.ArgumentParser()

parser.add_argument('--splits', nargs=1, dest='splits_count', type=int, required=True, help='number of splits')
parser.add_argument('--dir', nargs=1, dest='working_dir', type=str, required=True, help='path to the directory to store outputs')
parser.add_argument('--combinations', dest='combinations', type=str, required=True, help='how many combinations we use, currenlty only supports min and max')
args = parser.parse_args()

details = {}

working_dir = args.working_dir[0]
if working_dir[-1] != '/':
    working_dir += '/'
splits_count = args.splits_count[0]

if splits_count == 4:
    if args.combinations == 'min':
        combinations = [(0,1),(1,2),(2,3)]
    elif args.combinations == 'max':
        combinations = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
elif splits_count == 8:
    if args.combinations == 'min':
        combinations = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 3), (3, 7), (1, 5)]
    elif args.combinations == 'max':
        combinations = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),(1,4),(2,4),(3,4),(0,5),(1,5),(2,5),(3,5),(4,5),(0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7)]
elif splits_count == 16:
    if args.combinations == 'min':
        combinations = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (0, 4), (4, 8), (8, 12), (12, 15), (8, 14), (4, 13), (4, 11), (4, 15), (0, 8), (3, 10), (1, 5), (2, 8), (6, 10)]
    elif args.combinations == 'max':
        combinations = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),(1,4),(2,4),(3,4),(0,5),(1,5),(2,5),(3,5),(4,5),(0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(0,8),(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(0,9),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9),(8,9),(0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10),(7,10),(8,10),(9,10),(0,11),(1,11),(2,11),(3,11),(4,11),(5,11),(6,11),(7,11),(8,11),(9,11),(10,11),(0,12),(1,12),(2,12),(3,12),(4,12),(5,12),(6,12),(7,12),(8,12),(9,12),(10,12),(11,12),(0,13),(1,13),(2,13),(3,13),(4,13),(5,13),(6,13),(7,13),(8,13),(9,13),(10,13),(11,13),(12,13),(0,14),(1,14),(2,14),(3,14),(4,14),(5,14),(6,14),(7,14),(8,14),(9,14),(10,14),(11,14),(12,14),(13,14),(0,15),(1,15),(2,15),(3,15),(4,15),(5,15),(6,15),(7,15),(8,15),(9,15),(10,15),(11,15),(12,15),(13,15),(14,15)]

details['splits_count'] = splits_count
details['combinations'] = combinations
details['number_of_combinations'] = len(combinations)

print('split_counts', splits_count)
graph = nx.read_edgelist(working_dir + '/all_edges.txt')
split_size = int(len(graph.nodes()) / splits_count) + 1

details['main_graph_nodes'] = len(graph.nodes())
details['main_graph_edges'] = len(graph.edges())

print('combinations', combinations)
print('# of combinations', len(combinations))
print('split_size', split_size)
print('working_dir', working_dir)

details['split_size'] = split_size
details['dataset_name'] = working_dir.split('/')[-2]

def write_edgelist(edgelist, path):
    with open(path, 'w+') as f:
        for e_idx, e in enumerate(edgelist):
            line =  str(e[0]) + ' ' + str(e[1])
            if e_idx != len(edgelist) - 1:
                line += '\n'
            f.write(line)
    print('file', path, 'created.')

def read_edgelist(file_path):
    with open(file_path) as f:
        lines = f.read().split('\n')
        if lines[-1].strip() == '':
            lines = lines[:-1]
        return [(line.split(' ')[0],line.split(' ')[1]) for line in lines]

def is_in(A1, A2, n1, n2):
    n1 = int(n1)
    n2 = int(n2)
    if(n1 in A1 and n2 in A2):
        return True
    if(n1 in A2 and n2 in A1):
        return True
    if(n1 in A1 and n2 in A1):
        return True
    if(n2 in A2 and n1 in A2):
        return True
    return False

if not os.path.exists(working_dir):
    print('the working directory doesn\'t exist.')
    os.makedirs(working_dir)
    print('created', working_dir)

parts_graph = nx.Graph()
combination_edges = []
for comb in combinations:
    parts_graph.add_edge(comb[0],comb[1])
    combination_edges.append([])

assert nx.diameter(parts_graph) <= 3

train_edges_true       = read_edgelist(working_dir + 'train_edges_true.txt')
train_edges_false      = read_edgelist(working_dir + 'train_edges_false.txt')
test_edges_true        = read_edgelist(working_dir + 'test_edges_true.txt')
test_edges_false       = read_edgelist(working_dir + 'test_edges_false.txt')
extra_test_edges_false = read_edgelist(working_dir + 'extra_test_edges_false.txt')

test_edges = []

for e in test_edges_true:
    test_edges.append((e[0],e[1],1))
    
for e in test_edges_false:
    test_edges.append((e[0],e[1],0))

details['true_train_edges_count'] = len(train_edges_true)
details['true_test_edges_count'] = len(test_edges_true)
print('input read')

#======================CREATE BATCHES AND COMBINATIONS START=========================
test_name = str(time.time())[:10]
working_dir += test_name + '/'
print('test_name', test_name)

details['test_name'] = test_name

if not os.path.exists(working_dir):
    os.makedirs(working_dir)
    print('created', working_dir)
print('changing working_dir to', test_name)
print('creating splits and combinations...')

split_ranges = []
for i in range(splits_count):
    split_ranges.append(range(i * split_size, (i+1) * split_size))

split_nodes = []
for _ in range(splits_count):
    split_nodes.append([])
for node in graph.nodes():
    for split_idx in range(splits_count):
        if int(node) in split_ranges[split_idx]:
            split_nodes[split_idx].append(node)
details['splits_nodes_counts'] = [len(l) for l in split_nodes]
ignored_edges = []
for e in list(graph.edges()):
    ignored = True
    for comb_idx, comb in enumerate(combinations):
        if(is_in(split_ranges[comb[0]], split_ranges[comb[1]], e[0], e[1])):
            combination_edges[comb_idx].append((e[0],e[1]))
            ignored = False
    if ignored:
        ignored_edges.append((e[0],e[1]))

number_of_ignored_edges = len(ignored_edges)
fraction_of_ignored_edges = len(ignored_edges) / len(graph.edges())
print(len(ignored_edges),'edgs ignored', len(ignored_edges) / len(graph.edges()) )

details['ignored_edges_count'] = len(ignored_edges)
details['ignored_edges_fraction'] = len(ignored_edges) / len(graph.edges())
write_edgelist(ignored_edges, working_dir + 'ignored_edges.txt')

for comb_idx, comb in enumerate(combinations):
    write_edgelist(combination_edges[comb_idx], working_dir + str(comb[0]) + '_' + str(comb[1]) + '_edges.txt')

details['combinations_edges_count'] = [
    (str(comb[0]) + '_' + str(comb[1]), len(combination_edges[comb_idx]))
 for comb in combinations
 ]

for split_idx in range(splits_count):
    f_name = working_dir + str(split_idx) + '_nodes.txt'
    with open(f_name, 'w+') as f:
        for node in split_nodes[split_idx]:
            f.write(str(node) + '\n')
    print('file',f_name,'created.')
print('splits and combinations created.')
#======================CREATE BATCHES AND COMBINATIONS FINISH=========================
#======================RUN NODE2VEC ON SPLITS START=========================
splits_node2vec_runnning_times = run_node2vec_combinations(
    combinations,
    concurrent_count=6,
    working_dir=working_dir,
    verbose=True)
details['splits_node2vec_runnning_times'] = splits_node2vec_runnning_times

def get_edge_embeddings(emb_mappings, edge_list):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        if node1 in emb_mappings:
            emb1 = emb_mappings[node1]
        else:
            continue
        if node2 in emb_mappings:
            emb2 = emb_mappings[node2]
        else:
            continue
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def read_combinations_models():
    combinations_models = {}
    print('training node2vec on each combination. we have {} combinations.'.format(len(combinations)))
    for comb_idx, comb in tqdm(enumerate(combinations)):
        comb_name = str(comb[0]) + '_' + str(comb[1])
        first_node2vec = {}
        second_node2vec = {}
        
        edge_list = combination_edges[comb_idx]
        edge_list =[(str(min(int(edge[0]), int(edge[1]))), str(max(int(edge[0]), int(edge[1])))) for edge in edge_list]

        for node in split_nodes[comb[0]]:
            if node.strip() != '':
                first_node2vec[node.strip()] = None

        for node in split_nodes[comb[1]]:
            if node.strip() != '':
                second_node2vec[node.strip()] = None

        all_nodes = list(first_node2vec.keys()) + list(second_node2vec.keys())
        emb_mappings = {}
        with open(working_dir+'embeddings/'+comb_name+'.emb') as f:
            lines = f.read().split('\n')
            for line in lines[1:-1]:
                line = line.split(' ')
                key = line[0]
                value = [float(f) for f in line[1:]]
                emb_mappings[key] = np.array(value)

        split_train_edges_false = []
        for e in train_edges_false:
            assert int(e[0]) <= int(e[1])
            if e[0] in emb_mappings and e[1] in emb_mappings:
                split_train_edges_false.append(e)

        while(len(split_train_edges_false) < len(edge_list)):
            idx_i = int(all_nodes[np.random.randint(0, len(all_nodes))])
            idx_j = int(all_nodes[np.random.randint(0, len(all_nodes))])

            if idx_i == idx_j:
                continue

            false_edge = (str(min(idx_i, idx_j)), str(max(idx_i, idx_j)))
            idx_i = false_edge[0]
            idx_j = false_edge[1]
            if idx_i not in emb_mappings:
                continue
            if idx_j not in emb_mappings:
                continue
            # Make sure false_edge not an actual edge, and not a repeat
            if false_edge in split_train_edges_false:
                continue
            if false_edge in edge_list:
                continue

            split_train_edges_false.append(false_edge)
        
        for e in split_train_edges_false:
            assert int(e[0]) <= int(e[1])

        edge_list_set = set(edge_list)
        split_train_edges_false_set = set(split_train_edges_false)
        assert len(edge_list_set.intersection(split_train_edges_false_set)) == 0

        pos_train_edge_embs = get_edge_embeddings(emb_mappings, edge_list)
        neg_train_edge_embs = get_edge_embeddings(emb_mappings, split_train_edges_false)

        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        train_edge_labels = np.concatenate([
            np.ones(pos_train_edge_embs.shape[0]), np.zeros(neg_train_edge_embs.shape[0])
        ])

        assert pos_train_edge_embs.shape[0] == neg_train_edge_embs.shape[0]
        
        edge_classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=250)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        for key in first_node2vec.keys():
            if key in emb_mappings:
                first_node2vec[key] = emb_mappings[key]
            else:
                first_node2vec[key] = np.zeros((128,))

        for key in second_node2vec.keys():
            if key in emb_mappings:
                second_node2vec[key] = emb_mappings[key]
            else:
                second_node2vec[key] = np.zeros((128,))

        emb_mappings = {}
        for key in first_node2vec.keys():
            emb_mappings[key] = first_node2vec[key]
        for key in second_node2vec.keys():
            emb_mappings[key] = second_node2vec[key]
        
        combinations_models[comb] = {
            'model': edge_classifier,
            'first_node2vec': first_node2vec,
            'second_node2vec': second_node2vec,
            'emb_mappings': emb_mappings
        }
        
    return combinations_models

combinations_models = read_combinations_models()

#======================RUN NODE2VEC ON SPLITS FINISH=========================
#======================RUN TEST ON TEST DATA START=========================

def get_similar_in_other_half(values, node):
    lookforin = None
    node_vector = values['emb_mappings'][node]
    
    if node in values['first_node2vec']:
        lookforin = values['second_node2vec']
    elif node in values['second_node2vec']:
        lookforin = values['first_node2vec']
    
    node_name = None
    distance = 10000
    for other_node, other_vector in lookforin.items():
        dis =  np.linalg.norm(other_vector - node_vector)
        if dis < distance:
            distance = dis
            node_name = other_node
    return node_name, values['emb_mappings'][node_name]

def get_split_idx(node, verbose=False):
    for split_idx in range(splits_count):
        if node in split_nodes[split_idx]:
            if verbose:
                print(node,'is in', split_idx)
            return split_idx
    assert False
    
def get_combination(b1,b2=None):
    if b2 == None:
        for comb in combinations_models.keys():
            if b1 in comb:
                return comb
    else:
        for comb in combinations_models.keys():
            if (b1,b2) == comb:
                return comb

def query_edge_embedding(node1, node2):

    split_x = min(get_split_idx(node1), get_split_idx(node2))
    split_y = max(get_split_idx(node1), get_split_idx(node2))
    # print('split_x({}),split_y({})'.format(node1,node2),split_x,split_y)
    path = nx.shortest_path(parts_graph, split_x, split_y)
    
    if len(path) == 1 or len(path) == 2:
        if len(path) == 1:
            comb_name = get_combination(path[0])
        elif len(path) == 2:
            comb_name = get_combination(path[0], path[1])

        values = combinations_models[comb_name]
        edge_emb = np.multiply(
            values['emb_mappings'][node1], 
            values['emb_mappings'][node2]
        )
        return edge_emb, len(path), comb_name
    
    elif len(path) == 3:
        alpha = None
        theta = None
        if node1 in split_nodes[path[0]]:
            alpha = node1
            theta = node2
        elif node1 in split_nodes[path[2]]:
            alpha = node2
            theta = node1

        comb_01 = (min(int(path[0]),int(path[1])), max(int(path[0]),int(path[1])))
        comb_12 = (min(int(path[1]),int(path[2])), max(int(path[1]),int(path[2])))
        alpha_embeding = combinations_models[comb_01]['emb_mappings'][alpha]
        theta_embeding = combinations_models[comb_12]['emb_mappings'][theta]
        
        #alpha_prim = the most similar one to alpha in path[1] & (path[0],path[1])
        alpha_prime, alpha_prime_vec = get_similar_in_other_half(combinations_models[comb_01], alpha)
        alpha_prime_in_12_vec = combinations_models[comb_12]['emb_mappings'][alpha_prime]

        #theta_prim = the most similar one to theta in path[1] & (path[1],path[2]) #todo I can use this too

        comb_name = comb_12
        values = combinations_models[comb_name]
        edge_emb = np.multiply(
            alpha_prime_in_12_vec, 
            theta_embeding #theta is already in 12
        )
        return edge_emb, len(path), comb_name
    
    elif len(path) == 4:
        if node1 in split_nodes[path[0]]:
            alpha = node1
            theta = node2
        elif node1 in split_nodes[path[3]]:
            alpha = node2
            theta = node1

        comb_01 = (min(int(path[0]),int(path[1])), max(int(path[0]),int(path[1])))
        comb_12 = (min(int(path[1]),int(path[2])), max(int(path[1]),int(path[2])))
        comb_23 = (min(int(path[2]),int(path[3])), max(int(path[2]),int(path[3])))

        alpha_embeding = combinations_models[comb_01]['emb_mappings'][alpha]
        theta_embeding = combinations_models[comb_23]['emb_mappings'][theta]

    #     alpha_prim = the most similar one to alpha in path[1] & (path[0],path[1])
        alpha_prime, alpha_prime_vec = get_similar_in_other_half(combinations_models[comb_01], alpha)
        alpha_prime_in_12_vec = combinations_models[comb_12]['emb_mappings'][alpha_prime]

    #     theta_prim = the most similar one to theta in path[2] & (path[2],path[3])
        theta_prim, theta_prim_vec = get_similar_in_other_half(combinations_models[comb_23], theta)
        theta_prime_in_12_vec = combinations_models[comb_12]['emb_mappings'][theta_prim]

        comb_name = comb_12
        values = combinations_models[comb_name]
        edge_emb = np.multiply(
            alpha_prime_in_12_vec, 
            theta_prime_in_12_vec
        )
        return edge_emb, len(path), comb_name
    else:
        assert False

# ignored_edges  = read_edge_list(working_dir + 'ignored_edges.txt')
# ignored_edges = [(e[0],e[1],1) for e in ignored_edges] 

test_edge_lbls = []
test_edge_pred = []
distances = []
print('predicting started')
stime = time.time()
for e in tqdm(test_edges):
    edge_emb, distance, model_name = query_edge_embedding(e[0],e[1])
    distances.append(distance)
    pred = combinations_models[model_name]['model'].predict_proba([edge_emb])[:, 1]
    test_edge_pred.append(pred)
    test_edge_lbls.append(e[2])

print(time.time() - stime)
test_roc = roc_auc_score(test_edge_lbls, test_edge_pred)
test_ap = average_precision_score(test_edge_lbls, test_edge_pred)

distances_details = {}

for d in distances:
    distances_details[d] = distances_details.get(d,0) + 1
details['distances'] = distances_details
details['test_roc'] = test_roc
details['test_ap'] = test_ap
print(test_roc)
print(test_ap)
#======================RUN TEST ON TEST DATA FINISH=========================

#write details to file
with open('results/'+str(test_name)+'.json', 'w') as fp:
    json.dump(details, fp, indent=4)
