import networkx as nx
import argparse

def generate_combinations(n, click_size):
    g = nx.Graph()
    for i in range(click_size):
        for j in range(i, click_size):
            g.add_edge(i,j)

    single_count = int((n-click_size) / click_size)
    for i in range(click_size, n):
        connect_to = int((i-click_size) / single_count)
        g.add_edge(connect_to, i)
    combs = []
    for e in g.edges():
        e = (min(e[0],e[1]),max(e[0],e[1]))
        combs.append(e)
    return combs

def main(args):
    combs = generate_combinations(args.n[0], args.click_size[0])
    output = '['
    for c in combs:
        output += '({}, {}),'.format(c[0],c[1])
    output = output[:-1]
    output += ']'
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', nargs=1, dest='n', type=int, required=True, help='number of batches')
    parser.add_argument('--click', nargs=1, dest='click_size', type=int, required=True, help='size of central click')
    args = parser.parse_args()

    main(args)