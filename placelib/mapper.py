import networkx as nx
import numpy as np

from models.program import Program
from models.network import Network
from placelib.util import mapInfo, graph_find_a_root

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

        self.total_graph = nx.complete_graph(self.network.edge_servers + self.network.end_devices)
        for edge in self.total_graph.edges:
            n1 = edge[0]
            n2 = edge[1]
            self.total_graph.edges[edge]['paths'] = []
            for path in nx.all_simple_paths(self.network.G_nodes, n1, n2):
                rate = 0
                delay = 0
                for i in range(len(path) - 1):
                    rate += 1/self.network.G_nodes.edges[path[i], path[i+1]]['bw']
                    delay += self.network.G_nodes.edges[path[i], path[i+1]]['weight']/1000
                    if i < len(path) - 2:
                        delay += self.network.nodes[path[i+1]].delay()
                self.total_graph.edges[edge]['paths'].append(mapInfo(path, rate, delay))

    def _stream_delay_min_(self, edge, kbytes):
        if edge[0] == edge[1]:
            return 0
        if edge not in self.total_graph.edges:
            edge = [edge[1], edge[0]]
        return min([p.data_rate * kbytes + p.delay for p in self.total_graph.edges[edge]['paths']])

    def evaluate(self, mapping, average=True):
        critical_time = 0
        for path in self.path_in_between:
            compute_time = np.sum([self.program.operators[op].estimate_compute_time(self.network.nodes[mapping[op]]) for op in path])
            communication_time = 0
            for i in range(len(path) - 1):
                op1 = path[i]
                op2 = path[i + 1]
                kbytes = self.program.get_stream_kbytes_between_operators(op1, op2)
                map_edge = (mapping[op1], mapping[op2])
                communication_time += self._stream_delay_min_(map_edge, kbytes)
            if compute_time + communication_time > critical_time:
                critical_time = compute_time + communication_time
        return critical_time


class heuMapper(Mapper):
    def __init__(self, program: Program, network: Network, start_operator, end_operator, mapping: dict, N, delta=10, p_explore=0.3):
        super().__init__(program, network,  start_operator, end_operator, mapping)
        assert len(self.pinned) > 1, "Heuristic placement requires at least two pinned operators"
        self.delta = delta
        self.p_explore = p_explore
        self.cluster_graph = []
        for _ in range(N):
            hier, node_loc = self.cluster()
            self.cluster_graph.append((hier, node_loc))

    def map(self, num_heuristic_restriction=5, N=1):
        best_mapping = self.single_map(num_heuristic_restriction, 0)
        critical_time = self.evaluate(best_mapping)
        for _ in range(N - 1):
            mapping = self.single_map(num_heuristic_restriction, _+1)
            time = self.evaluate(mapping)
            if time < critical_time:
                best_mapping = mapping
                critical_time = time
        return best_mapping


    def single_map(self, num_heuristic_restriction, i):
        (hier, node_loc) = self.cluster_graph[i]
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


class exhMapper(Mapper):
    def __init__(self, program: Program, network: Network, start_operator, end_operator, mapping: dict):
        super().__init__(program, network, start_operator, end_operator, mapping)
        self.shortest_paths = {}
        for n in self.network.G_nodes.nodes:
            self.shortest_paths[n] = {}
            for m in self.network.G_nodes.nodes:
                self.shortest_paths[n][m] = self.network.get_shortest_path(n, m)

    def map(self):

        def helper(sub_program: nx.DiGraph):
            if not nx.number_of_nodes(sub_program):
                return {}, {}
            if nx.number_of_nodes(sub_program) == 1:
                for n in sub_program.nodes:
                    return {n: {placement: self.network.nodes[placement].delay(self.program.operators[n]) for placement in self.total_graph.nodes}}, {}

            root = graph_find_a_root(sub_program)
            root_edges = list(sub_program.edges(root))
            sub_program.remove_node(root)
            node_delay, stream_delay = helper(sub_program)
            for edge in root_edges:
                stream_delay[edge] = {}
                for map_edge in self.total_graph.edges:
                    d = self._stream_delay_min_(map_edge, self.program.G.edges[edge]['bytes'])
                    stream_delay[edge][map_edge]= d
                    stream_delay[edge][(map_edge[1], map_edge[0])] = d
            node_delay[root] = {placement: self.network.nodes[placement].delay(self.program.operators[root]) for placement in self.total_graph.nodes}
            return node_delay, stream_delay

        class Unit:
            def __init__(self, op, in_link, logic_parents: list, graph_list):
                self.op = op
                self.in_link = in_link
                if in_link:
                    in_link.add_next_unit(self)
                self.out_links = {}
                self.graph_list = graph_list

                self.parent_map = {}
                self.critical_link = None

                if logic_parents:
                    cur = self.in_link
                    while len(self.parent_map) < len(logic_parents) and cur:
                        if cur.operator in logic_parents:
                            self.parent_map[cur.operator] = cur
                        cur = cur.unit.in_link
                    if len(self.parent_map) != len(logic_parents):
                        raise ValueError

            def add_links(self, option):
                self.out_links[option] = Link(self, option, self.graph_list)

        class Link:
            def __init__(self, unit: Unit, option, graph):
                self.unit = unit
                self.operator = unit.op
                self.option = option
                self.next_unit = None

                delay = 0
                self.critical_path = None
                for parent, link in self.unit.parent_map.items():
                    op_edge = (parent, self.operator)
                    this_delay = link.cost
                    if link.option != self.option:
                        map_edge = (link.option, self.option)
                        this_delay += graph.stream_delay[op_edge][map_edge]
                    if this_delay > delay:
                        delay = this_delay
                        self.critical_path = link
                self.cost = graph.node_delay[self.operator][option] + delay

            def add_next_unit(self, unit):
                assert not self.next_unit
                self.next_unit = unit


        class GraphList:
            def __init__(self, program_graph, start_op, end_op, node_delay, stream_delay):
                self.root = Unit(start_op, None, [], self)

                self.start_op = start_op
                self.terminal = end_op
                self.unit_list = {start_op: [self.root]}
                self.program_graph = program_graph
                self.operator_order = nx.topological_sort(program_graph)
                self.node_delay = node_delay
                self.stream_delay = stream_delay

                for option in node_delay[self.start_op]:
                    self.root.add_links(option)

                previous_node = start_op
                start = False
                for op in self.operator_order:
                    if not start and op != start_op:
                        continue
                    elif op == start_op:
                        start = True
                        continue
                    self.unit_list[op] = []
                    parents = list(program_graph.predecessors(op))
                    for unit in self.unit_list[previous_node]:
                        for link in unit.out_links.values():
                            new_unit = Unit(op, link, parents, self )
                            self.unit_list[op].append(new_unit)
                            for option in node_delay[op]:
                                new_unit.add_links(option)
                    previous_node = op


            def get_optimal_solution(self, terminal):
                terminal_units = self.unit_list[terminal]
                all_solutions = {link: link.cost for unit in terminal_units for link in unit.out_links.values()}
                optimal = min(all_solutions, key=all_solutions.get)
                mapping = {}
                cur = optimal
                while cur:
                    mapping[cur.operator] = cur.option
                    cur = cur.unit.in_link
                return mapping

        g = self.program.G.copy()
        out_edges = list(g.edges(self.start_op))
        in_edges = list(g.in_edges(self.end_op))
        g.remove_node(self.start_op)
        g.remove_node(self.end_op)
        node_delay, stream_delay = helper(g)
        source = self.mapping[self.start_op]
        sink = self.mapping[self.end_op]

        node_delay[self.start_op] = {source: self.network.nodes[source].delay(self.program.operators[self.start_op])}
        node_delay[self.end_op] = {sink: self.network.nodes[sink].delay(self.program.operators[self.end_op])}
        for edge in out_edges:
            stream_delay[edge] = {(source, next_node): self._stream_delay_min_((source, next_node), self.program.G.edges[edge]['bytes']) for next_node in self.total_graph.nodes}
        for edge in in_edges:
            stream_delay[edge] = {(previous_node, sink): self._stream_delay_min_((previous_node, sink), self.program.G.edges[edge]['bytes']) for previous_node in self.total_graph.nodes}

        data = nx.DiGraph()
        data.add_nodes_from(self.program.G)
        data.add_edges_from(self.program.G.edges)
        graph = GraphList(data, self.start_op, self.end_op, node_delay, stream_delay)

        return graph.get_optimal_solution(self.end_op)





