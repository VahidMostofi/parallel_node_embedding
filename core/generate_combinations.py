import networkx as nx
n = 8
out = '['
for i in range(n):
    for j in range(i):
        out += '({},{}),'.format(j,i)
out = out[:-1]
out += ']'
print('maximum')
print(out)

print('minimum')
combinations = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),
(0,4),(4,8),(8,12),(12,15),(8,14),(4,13),(4,11),(4,15),(0,8),(3,10),(1,5),(2,8),(6,10)
]

g = nx.Graph()
for e in combinations:
    g.add_edge(*e)

for gg in nx.all_pairs_bellman_ford_path(g):
    start = gg[0]
    for kk,vv in gg[1].items():
        if len(vv) > 4:
            print(start, kk, vv)

print(nx.diameter(g))
print(combinations)