import networkx as nx
import numpy as np

from models.program import Program
from models.network import Network

class Mapper:
    def __init__(self, program: Program, network: Network, start_operator, end_operator, mapping=None):
        self.program = program
        self.network = network
        self.pinned = []
        self.mapping = mapping
        self.start_op = start_operator
        self.end_op = end_operator
        assert start_operator in self.program.G.nodes
        assert end_operator in self.program.G.nodes
        self.path_in_between = list(nx.all_simple_paths(self.program.G, source=self.start_op, target=self.end_op))

        if mapping:
            assert type(mapping) == dict, "Already mapped items should be passed in dict"
            for mapped in mapping:
                assert mapped in self.program.G.nodes, f"Mapped operator {mapped} is not in the program graph"
                assert mapping[mapped] in self.network.edge_servers or mapping[mapped] in self.network.end_devices
                self.pinned.append(mapped)

    def evaluate(self, mapping, average=True):
        critical_time = 0
        for path in self.path_in_between:
            compute_time = np.sum([self.program.operators[op].estimate_compute_time(self.network.nodes[mapping[op]]) for op in path])
            communication_time = 0
            for i in range(len(path) - 1):
                op1 = path[i]
                op2 = path[i + 1]
                kbytes = self.program.get_stream_kbytes_between_operators(op1, op2)
                communication_time += self.network.latency_between_nodes(op1, op2, kbytes=kbytes, average=average)
            if compute_time + communication_time > critical_time:
                critical_time = compute_time + communication_time

        return critical_time


class heuMapper(Mapper):
    def __init__(self, program: Program, network: Network, start_operator, end_operator, mapping: dict,  delta=10, p_explore=0.3):
        super().__init__(program, network,  start_operator, end_operator, mapping)
        assert len(self.pinned) > 1, "Heuristic placement requires at least two pinned operators"
        self.delta = delta
        self.p_explore = p_explore


    def map(self, num_heuristic_restriction=5, N=1):
        best_mapping = self.single_map(num_heuristic_restriction)
        critical_time = self.evaluate(best_mapping)
        for _ in range(N - 1):
            mapping = self.single_map(num_heuristic_restriction)
            time = self.evaluate(mapping)
            if time < critical_time:
                best_mapping = mapping
                critical_time = time
        return best_mapping


    def single_map(self, num_heuristic_restriction):
        hier, node_loc = self.cluster()
        program_graph = self.program.G.to_undirected()
        operator_restriction = self.domain_restriction(hier, node_loc, self.p_explore)
        mapping = self.mapping.copy()

        done = False
        while not done:
            connected_to_mapped = [neighbor for n in mapping for neighbor in program_graph.neighbors(n)]
            can_be_placed = [n for n in connected_to_mapped if self.network.is_operating(operator_restriction[n]) and n not in mapping]
            for n in can_be_placed:
                candidates = self.network.get_operating_devices(operator_restriction[n])
                mapping[n] = np.random.choice(candidates)
            unplaced = list(set(program_graph.nodes) - set(mapping.keys()))
            if not len(unplaced):
                done = True
            else:
                operator_restriction_node = {n: self.network.random_node(operator_restriction[n]) for n in operator_restriction}
                for _ in range(num_heuristic_restriction):
                    picked = np.random.choice(unplaced)
                    neighbors = list(program_graph.neighbors(picked))
                    bytes = np.array([program_graph.edges[picked, n]['bytes'] for n in neighbors])
                    picked_neighbor = np.random.choice(neighbors, p=bytes/np.sum(bytes))
                    source = operator_restriction_node[picked]
                    target = operator_restriction_node[picked_neighbor]
                    if picked_neighbor in mapping:
                        target = mapping[picked_neighbor]
                    path = nx.shortest_path(self.network.G_nodes, source, target)
                    if len(path) > 1:
                        operator_restriction_node[picked] = path[1]
                for operator in unplaced:
                    operator_restriction[operator] = self.network.get_domain_id(operator_restriction_node[operator])
        return mapping


    def cluster(self):
        program_graph = nx.Graph(self.program.G.to_undirected())
        N = len(program_graph.nodes)
        partitions = [program_graph]
        node_list = {}

        cluster_id = 0
        hier = nx.DiGraph()
        program_graph.graph['id'] = cluster_id
        hier.add_node(cluster_id, nodes=list(program_graph.nodes), level=0)

        while len(node_list.keys()) < N:
            for subgraph in partitions:
                if not nx.is_connected(subgraph):
                    partitions.remove(subgraph)
                    for comp in nx.connected_components(subgraph):
                        cluster_id += 1
                        hier.add_edge(subgraph.graph['id'], cluster_id)
                        hier.nodes[cluster_id]['nodes'] = list(comp)
                        hier.nodes[cluster_id]['level'] = hier.nodes[subgraph.graph['id']]['level'] + 1
                        if len(comp) == 1:
                            singleton = comp.pop()
                            node_list[singleton] = cluster_id
                        else:
                            subsubgraph = subgraph.subgraph(comp).copy()
                            subsubgraph.graph['id'] = cluster_id
                            partitions.append(subsubgraph)
                else:
                    nodes = np.random.choice(subgraph.nodes, 2, replace=False)
                    node1 = nodes[0]
                    node2 = nodes[1]
                    p = nx.shortest_path(subgraph, node1, node2, weight='bytes')
                    for i in range(len(p) - 1):
                        subgraph.edges[p[i], p[i+1]]['bytes'] -= self.delta
                        if subgraph.edges[p[i], p[i+1]]['bytes']  <= 0:
                            subgraph.remove_edge(p[i], p[i+1])

        return hier, node_list

    def domain_restriction(self, hier: nx.DiGraph, loc, p_explore):
        operator_restriction = {n: self.network.get_domain_id(self.mapping[n]) for n in self.pinned}
        cluster_restriction = {loc[n]: self.network.get_domain_id(self.mapping[n]) for n in self.pinned}
        for cluster in reversed(list(nx.topological_sort(hier))):
            if cluster not in cluster_restriction:
                subcluster = hier.successors(cluster)
                restricted_subcluster = set(subcluster) & set(cluster_restriction.keys())
                if len(restricted_subcluster):
                    restriction_domains = set([cluster_restriction[c] for c in restricted_subcluster])
                    common_domain = self.network.common_domain(list(restriction_domains))
                    if (not self.network.get_parent_domain_id(common_domain)) or np.random.random() > p_explore:
                        cluster_restriction[cluster] = common_domain
                    else:
                        parent_domain = np.random.choice(self.network.get_parent_domain_id(common_domain))
                        cluster_restriction[cluster] = parent_domain
                    for node in hier.nodes[cluster]['nodes']:
                        if node not in operator_restriction:
                            operator_restriction[node] = cluster_restriction[cluster]
        return operator_restriction