import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import argparse
import sys
import os

working_dir = '../data/amazon/'
if working_dir[-1] != '/':
    working_dir += '/'

random_seed = 4564

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

nodes = set()

test_edges_true = []
with open('../data/amazon/test_edges_true.txt') as f:
	test_edges_true = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
test_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in test_edges_true]

train_edges_true = []
with open('../data/amazon/train_edges_true.txt') as f:
	train_edges_true = [(line.split(' ')[0], line.split(' ')[1]) for line in f.read().split('\n')]
train_edges_true = [(str(min(int(e[0]),int(e[1]))),str(max(int(e[0]),int(e[1])))) for e in train_edges_true]

test_edges_true_set = set(test_edges_true)
train_edges_true_set= set(train_edges_true)

for e in train_edges_true_set.union(test_edges_true_set):
	if e[0] not in nodes:
		nodes.add(e[0])
	if e[1] not in nodes:
		nodes.add(e[1])

assert len(test_edges_true_set.intersection(train_edges_true_set)) == 0

test_edge_count = len(test_edges_true)
nodes = list(nodes)
edge_list_dict = {}
for e in test_edges_true_set:
    if e[0] not in edge_list_dict:
        edge_list_dict[e[0]] = []
    edge_list_dict[e[0]].append(e[1])

for e in train_edges_true_set:
    if e[0] not in edge_list_dict:
        edge_list_dict[e[0]] = []
    edge_list_dict[e[0]].append(e[1])


test_edges_false = set()
while(len(test_edges_false) < test_edge_count * 2):
    idx_i = int(nodes[np.random.randint(0, len(nodes))])
    idx_j = int(nodes[np.random.randint(0, len(nodes))])

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
    idx_i = int(nodes[np.random.randint(0, len(nodes))])
    idx_j = int(nodes[np.random.randint(0, len(nodes))])

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

# write_edgelist(train_edges_true, working_dir + 'train_edges_true.txt')
write_edgelist(train_edges_false, working_dir + 'train_edges_false.txt')
# write_edgelist(test_edges_true, working_dir + 'test_edges_true.txt')
write_edgelist(test_edges_false, working_dir + 'test_edges_false.txt')
# write_edgelist(extra_test_edges_false, working_dir + 'extra_test_edges_false.txt')
# write_edgelist(train_edges_true + test_edges_true, working_dir + 'all_edges.txt')