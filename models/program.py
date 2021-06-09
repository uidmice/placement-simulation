import numpy as np
import networkx as nx

from models.operator import Operator
from models.stream import Stream

class Program:
    def __init__(self, G, ddl_s, ddl_t):
        assert nx.is_directed_acyclic_graph(G), "Program graph is not a DAG"
        assert ddl_s in G.nodes, "Operator " + str(ddl_s) + "is not in the graph"
        assert ddl_t in G.nodes, "Operator " + str(ddl_t) + "is not in the graph"

        self.G = G
        self.ddl_s = ddl_s
        self.ddl_t = ddl_t
        self.operators = {}
        self.streams = {}

        for n in self.G.nodes:
            self.operators[n] = Operator(n, self.G.nodes[n]['computation'])

        count = len(self.operators)
        for edge in self.G.edges:
            s =  Stream(count, edge[0], edge[1], self.G.edges[edge]['bytes'])
            self.streams[count] = s
            self.operators[edge[0]].out_streams.append(s)
            self.operators[edge[1]].in_streams.append(s)
            self.operators[edge[1]].root = False


    def _examine_graph(self, G):
        Levels = {0:[]}
        dy = 5
        dx = 13
        pos = {}
        for n in nx.topological_sort(G):
            pos[n] = [0,0]
            if G.in_degree(n) == 0:
                G.nodes[n]["Level"] = 0
                Levels[0].append(n)
            else:
                G.nodes[n]["Level"] = max((G.nodes[a]["Level"] for a in G.predecessors(n))) + 1
                pos[n][0] = G.nodes[n]["Level"] * (dx - G.nodes[n]["Level"] )
                if G.nodes[n]["Level"] not in Levels.keys():
                    Levels[G.nodes[n]["Level"]] = []
                Levels[G.nodes[n]["Level"]].append(n)
        loc = {}
        for i in range(max([len(a) for a in Levels.values()])):
            loc[i] = list(range(-dy*i, dy*i+1, 2*dy))
        # print(loc)
        for k, v in Levels.items():
            if k == 0:
                for i, node in enumerate(v):
                    pos[node][1] = loc[len(v)-1][i]
            else:
                pre_position = {}
                for node in v:
                    pre_position[node] = np.average([pos[id][1] for id in G.predecessors(node)])
                sort = dict(sorted(pre_position.items(), key=lambda item: item[1]))
                for i, node in enumerate(sort.keys()):
                    pos[node][1] = loc[len(v)-1][i]+2*k * np.sign(loc[len(v)-1][i]) + 0.4*k*k + 0.2*k*k*k
        return pos, Levels

    def plot(self):
        pos, levels = self._examine_graph(self.G)
        nx.draw(self.G, pos=pos, with_labels=True, font_weight='bold')