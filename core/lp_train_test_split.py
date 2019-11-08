import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser()
# todo give batch count to arg parser
parser.add_argument('--input', nargs=1, dest='input_file', type=str, required=True, help='path to edge list of input graph')
parser.add_argument('--dir', nargs=1, dest='working_dir', type=str, required=True, help='path to the directory to store outputs')
parser.add_argument('--test-fraction', dest='test_fraction', nargs='?', const=0.3, type=float, required=False)
parser.add_argument('--remove-batches', dest='remove_batches', nargs='?', const=512, type=int,default=512, help='how many edge remove in each step')
parser.add_argument('--random-seed', dest='random_seed', nargs='?', const=4, type=int,default=4, help='random seed')
args = parser.parse_args()

input_file_path = args.input_file[0]
working_dir = args.working_dir[0]
if working_dir[-1] != '/':
    working_dir += '/'
test_fraction = args.test_fraction
BATCH_COUNT = args.remove_batches
random_seed = args.random_seed

if not os.path.exists(working_dir):
    print('the working directory doesn\'t exist.')
    os.makedirs(working_dir)
    print('created', working_dir)

def write_edgelist(edgelist, path):
    with open(path, 'w+') as f:
        for e_idx, e in enumerate(edgelist):
            line =  str(e[0]) + ' ' + str(e[1])
            if e_idx != len(edgelist) - 1:
                line += '\n'
            f.write(line)
    print('file', path, 'created.')

graph = nx.read_edgelist(input_file_path)
graphs = list(nx.connected_component_subgraphs(graph))

if len(graphs) > 1:
    print('the graph is not connected, choosing the largest connected component')
graph = max(graphs, key=len)
print('lcc features', len(graph.nodes),len(graph.edges))

test_edge_count = int(test_fraction * len(graph.edges))
print('test_edge_count',test_edge_count)
removed_edges = []

node_label_mapping = {}
new_labels = list(range(len(graph.nodes())))

random.Random(random_seed).shuffle(new_labels)
for node_idx, node_label in enumerate(list(graph.nodes())):
    node_label_mapping[node_label] = str(new_labels[node_idx])
graph = nx.relabel_nodes(graph, node_label_mapping)
# print(graph.nodes())
# print(node_label_mapping)
print(len(graph.nodes()), max([int(n) for n in graph.nodes()]))
assert len(graph.nodes()) -1 == max([int(n) for n in graph.nodes()])

while len(removed_edges) < test_edge_count:
    idx = 0

    edge_list = list(graph.edges())
    edge_list = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in edge_list]
    random.Random(4).shuffle(edge_list)
    print('BATCH_COUNT is', BATCH_COUNT)
    for idx in tqdm(range(0,len(edge_list),BATCH_COUNT)):
        batch_edges = edge_list[idx:idx+BATCH_COUNT]

        for e in batch_edges:
            graph.remove_edge(*e)
        if nx.is_connected(graph):
            for e in batch_edges:
                removed_edges.append(e)
        else:
            for e in batch_edges:
                graph.add_edge(*e)

        if len(removed_edges) > test_edge_count:
            break
        idx += BATCH_COUNT
    BATCH_COUNT = max(int(BATCH_COUNT/2),1)
    print('test_edge_count',test_edge_count)
    print('len(removed_edges)/test_edge_count',str(len(removed_edges))+'/'+str(test_edge_count))

test_edge_count = len(removed_edges)
test_edges_true = removed_edges
test_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in test_edges_true]
train_edges_true = list(graph.edges())
train_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in train_edges_true]

test_edges_true_set = set(test_edges_true)
train_edges_true_set= set(train_edges_true)

assert len(test_edges_true_set.intersection(train_edges_true_set)) == 0

edge_list_dict = {}
for e in test_edges_true_set:
    if e[0] not in edge_list_dict:
        edge_list_dict[e[0]] = []
    edge_list_dict[e[0]].append(e[1])

for e in train_edges_true_set:
    if e[0] not in edge_list_dict:
        edge_list_dict[e[0]] = []
    edge_list_dict[e[0]].append(e[1])

nodes = list(graph.nodes())
test_edges_false = set()
while(len(test_edges_false) < test_edge_count * 2):
    idx_i = int(nodes[np.random.randint(0, len(graph.nodes()))])
    idx_j = int(nodes[np.random.randint(0, len(graph.nodes()))])

    if idx_i == idx_j:
        continue

    false_edge = (str(min(idx_i, idx_j)), str(max(idx_i, idx_j)))
    idx_i = false_edge[0]
    idx_j = false_edge[1]
    # Make sure false_edge not an actual edge, and not a repeat
    if idx_i in edge_list_dict:
        if idx_j in edge_list_dict[idx_i]:
            continue
    if false_edge in test_edges_false:
        continue

    test_edges_false.add(false_edge)
print('test_edges_false extracted')

train_edges_false = set()
while(len(train_edges_false) < len(train_edges_true)):
    idx_i = int(nodes[np.random.randint(0, len(graph.nodes()))])
    idx_j = int(nodes[np.random.randint(0, len(graph.nodes()))])

    if idx_i == idx_j:
        continue

    false_edge = (str(min(idx_i, idx_j)), str(max(idx_i, idx_j)))
    idx_i = false_edge[0]
    idx_j = false_edge[1]
    # Make sure false_edge not an actual edge, and not a repeat
    if idx_i in edge_list_dict:
        if idx_j in edge_list_dict[idx_i]:
            continue
    if false_edge in train_edges_false:
        continue
    if false_edge in test_edges_false:
        continue

    train_edges_false.add(false_edge)
print('train_edges_false extracted')

####### performe some test ########
for ss_idx, ss in enumerate([
    train_edges_true,
    train_edges_false,
    test_edges_true,
    test_edges_false]):
    for e in ss:
        if int(e[0]) > int(e[1]):
            print('problem at', ss_idx)
            assert False

train_edges_true_set = set(train_edges_true)
train_edges_false_set = set(train_edges_false)
test_edges_true_set = set(test_edges_true)
test_edges_false_set = set(test_edges_false)
assert len(train_edges_true_set.intersection(train_edges_false_set)) == 0
assert len(train_edges_true_set.intersection(test_edges_true_set)) == 0
assert len(train_edges_true_set.intersection(test_edges_false_set)) == 0

assert len(train_edges_false_set.intersection(test_edges_true_set)) == 0
assert len(train_edges_false_set.intersection(test_edges_false_set)) == 0

assert len(test_edges_true_set.intersection(test_edges_false_set)) == 0
test_edges_false = list(test_edges_false)
extra_test_edges_false = test_edges_false[:test_edge_count]
test_edges_false = test_edges_false[test_edge_count:]

print('len(train_edges_true)', len(train_edges_true))
print('len(train_edges_false)',len(train_edges_false))
print('len(test_edges_true)',  len(test_edges_true))
print('len(test_edges_false)', len(test_edges_false))
print('len(extra_test_edges_false)', len(extra_test_edges_false))

write_edgelist(train_edges_true, working_dir + 'train_edges_true.txt')
write_edgelist(train_edges_false, working_dir + 'train_edges_false.txt')
write_edgelist(test_edges_true, working_dir + 'test_edges_true.txt')
write_edgelist(test_edges_false, working_dir + 'test_edges_false.txt')
write_edgelist(extra_test_edges_false, working_dir + 'extra_test_edges_false.txt')
write_edgelist(train_edges_true + test_edges_true, working_dir + 'all_edges.txt')