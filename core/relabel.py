import random
import sys
if len(sys.argv) < 3:
    print('first arg: input file, second arg: output file')
    sys.exit(0)
edges = []
with open(sys.argv[1]) as f:
    edges = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in f.readlines()]

node_counts = 0
nodes = {}
for e in edges:
    if e[0] not in nodes:
        nodes[e[0]] = -1
    if e[1] not in nodes:
        nodes[e[1]] = -1

node_counts = len(nodes)
new_node_names = list(range(node_counts))
random.shuffle(new_node_names)

c = 0
for node in nodes.keys():
    nodes[node] = new_node_names[c]
    c += 1
new_edges = [(nodes[e[0]], nodes[e[1]]) for e in edges]
with open(sys.argv[2], 'w') as f:
    for line in new_edges[:-1]:
        f.write('{} {}\n'.format(line[0], line[1]))
    f.write('{} {}'.format(line[0], line[1]))