import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

from models.domain import Domain
from models.device import Device
from models.router import Router
from models.link import Link


class Network:
    '''
        The Network object captures the hierarchical network model

        Parameters
        ----------
        G_nodes : nx.Graph
            The network graph. Nodes in the graph represents either Devices (including end-devices and edge-servers) or Routers. Edges are communication links.
        G_domain : nx.Graph
            The hierarchical domain graph. A node in the graph is either a transit domain, a stub domain, or a LAN.
        pos_node : dict {device_id : (x, y) }
            The location of the each nodes in the network graph G_node

        Attributes
        ----------
        arg : str
            This is where we store arg,
    '''
    def __init__(self, G_nodes: nx.Graph, G_domain: nx.Graph, pos_node):
        self.G_nodes = G_nodes
        self.G_domain = G_domain
        self.pos_node = pos_node
        self.domains = {}
        self.nodes = {}
        self.end_devices = [n for n in self.G_nodes.nodes if self.G_nodes.nodes[n]['type']=='host']
        self.edge_servers = [n for n in self.G_nodes.nodes if self.G_nodes.nodes[n]['type'] == 'edge']
        self.transit_domain = [n for n in self.G_domain.nodes if self.G_domain.nodes[n]['type'] == 'transit']
        self.stub_domain = [n for n in self.G_domain.nodes if self.G_domain.nodes[n]['type'] == 'stub']
        self.lan_domain = [n for n in self.G_domain.nodes if self.G_domain.nodes[n]['type'] == 'lan']

        for d in self.G_domain.nodes:
            type = self.G_domain.nodes[d]['type']
            self.domains[d] = Domain(d, type)
            if type == 'transit':
                self.G_domain.nodes[d]['level'] = 3
            elif type == 'stub':
                self.G_domain.nodes[d]['level'] = 2
            else:
                self.G_domain.nodes[d]['level'] = 1
            for n in self.G_domain.adj[d]:
                if d in self.transit_domain and n in self.stub_domain:
                    self.domains[d].leaf_domains.append(n)
                elif d in self.stub_domain and n in self.transit_domain:
                    self.domains[d].parent_domains.append(n)
                elif d in self.stub_domain and n in self.lan_domain:
                    self.domains[d].leaf_domains.append(n)
                elif d in self.lan_domain and n in self.stub_domain:
                    self.domains[d].parent_domains.append(n)

        for n in self.G_nodes.nodes:
            type = self.G_nodes.nodes[n]['type']
            if type in ['host', 'edge']:
                self.nodes[n] = Device(n, self.G_nodes.nodes[n]['rate'])
            else:
                self.nodes[n] = Router(n, self.G_nodes.nodes[n]['rate'])
            self.domains[self.G_nodes.nodes[n]['domain']].add_node(n, type)

        for e in self.G_domain.edges:
            self.G_domain.edges[e]['weight'] = self.latency_between_domains(e[0], e[1], 10)

    def get_shortest_path(self, node1, node2):
        return nx.shortest_path(self.G_nodes, node1, node2, weight='weight')

    def latency_between_nodes(self, node1, node2, kbytes): # ms
        path = nx.shortest_path(self.G_nodes, node1, node2, weight='weight')
        d = 0
        for i in range(len(path) - 1):
            d += self.G_nodes[path[i]][path[i+1]]['weight']/1000 + kbytes/self.G_nodes[path[i]][path[i+1]]['bw']
        for i in range(len(path) - 2):
            d += self.nodes[path[i+1]].delay
        return d * 10

    def latency_from_node_to_domain(self, node, domain, kbytes):
        target_nodes = self.domains[domain].nodes
        return np.average([self.latency_between_nodes(node, tn, kbytes) for tn in target_nodes])

    def latency_between_domains(self, domain1, domain2, kbytes):
        nodes = self.domains[domain1].nodes
        return np.average([self.latency_from_node_to_domain(node, domain2, kbytes) for node in nodes])

    def get_domain_id(self, node):
        return self.G_nodes.nodes[node]['domain']

    def get_parent_domain_id(self, domain):
        return self.domains[domain].parent_domains

    def children_lan_domain(self, domain):
        if domain in self.lan_domain:
            return [domain]
        if domain in self.stub_domain:
            return self.domains[domain].leaf_domains.copy()
        if domain in self.transit_domain:
            stubs = self.domains[domain].leaf_domains
            return [item for n in stubs for item in self.domains[n].leaf_domains]

    def sub_graph_domains(self, domain):
        if domain in self.lan_domain:
            return {domain}
        a =  set().union(*[self.sub_graph_domains(d) for d in self.domains[domain].leaf_domains])
        a.add(domain)
        return a

    def is_operating(self, domain):
        return self.domains[domain].function == 'operating'

    def is_routing(self, domain):
        return self.domains[domain].function == 'routing'

    def get_operating_devices(self, domain):
        if self.is_routing(domain):
            return []
        return [n for n in self.domains[domain].nodes if n in self.edge_servers or n in self.end_devices]

    def random_node(self, domain):
        return np.random.choice(self.domains[domain].nodes)

    def common_domain(self, domain_list: list):
        if len(domain_list) == 0:
            return None

        if len(domain_list) == 1:
            return domain_list[0]

        if len(domain_list) == 2:
            d1 = domain_list[0]
            d2 = domain_list[1]
            if d1 in self.sub_graph_domains(d2):
                return d2
            if d2 in self.sub_graph_domains(d1):
                return d1
            p = nx.shortest_path(self.G_domain, d1, d2, weight='weight')
            current_level = self.G_domain.nodes[d1]['level']
            current_domain = d1
            for i in range(len(p) - 1):
                if self.G_domain.nodes[p[i+1]]['level'] > current_level:
                    current_level = self.G_domain.nodes[p[i+1]]['level']
                    current_domain = p[i+1]
            return current_domain

        return self.common_domain([domain_list[0], self.common_domain(domain_list[1:])])




    def draw_nodes(self, show=False):
        plt.figure(figsize=(8, 8))
        node_color = {'transit': 'blue', 'stub': 'orangered', 'lan': 'g', 'edge': 'grey', 'host': 'lawngreen', 'gateway': 'darkgreen'}
        link_color = {'T': 'cornflowerblue', 'TT': 'dodgerblue', 'TS': 'tomato', 'S': 'tomato', 'SL': 'g', 'L': 'lime'}
        node_size = {'transit': 150, 'stub': 150, 'host': 150,'edge': 150, 'gateway': 150}
        width_map = {'T': 6, 'TT': 6, 'TS': 6, 'S': 6, 'SL': 6, 'L': 6}
        color_map = [node_color[self.G_nodes.nodes[n]['type']] for n in self.G_nodes.nodes]
        edge_map = [link_color[self.G_nodes.edges[e]['type']] for e in self.G_nodes.edges]

        nx.draw(self.G_nodes, pos=self.pos_node,
                node_size=[node_size[self.G_nodes.nodes[n]['type']] for n in self.G_nodes.nodes], edge_color=edge_map,
                width=[width_map[self.G_nodes.edges[e]['type']] for e in self.G_nodes.edges], node_color=color_map)
        if show:
            plt.show()

    def draw_domains(self, show=False):
        plt.figure(figsize=(8,4))
        node_color = {'transit': 'blue', 'stub': 'r', 'lan': 'g', 'host': 'g', 'gateway': 'b'}
        color_map = [node_color[self.G_domain.nodes[n]['type']] for n in self.G_domain.nodes]
        nx.draw(self.G_domain, pos=graphviz_layout(self.G_domain, prog="dot"),  node_color=color_map)
        if show:
            plt.show()







