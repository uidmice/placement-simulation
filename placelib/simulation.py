from placelib.util import *
from models.program import Program
from models.network import Network
from placelib.mapper import heuMapper


class Simulation:
    '''
        The Simulation object contains is an instance of the simulation

        Parameters
        ----------
        arg : str
            The arg is used for ...
        *args
            The variable arguments are used for ...
        **kwargs
            The keyword arguments are used for ...

        Attributes
        ----------
        arg : str
            This is where we store arg,
    '''
    def __init__(self, args):

        self.rnd = np.random.RandomState(args.seed)
        self.args = args

        '''
        Generate network graph
        '''
        G_domain, G_nodes, pos_domain, pos_node = network_random_topology(T=1, NT=3, S=5, NS=1, L=2, NL=4, ET=3, ES=2, EST=1, ETT=2, ELS=1, rnd=self.rnd)

        network_edge_server_random_placement(G_nodes, pos_node, args.edge_server_transit_percentage, args.edge_server_stub_percentage, args.edge_server_gateway_percentage, self.rnd)
        network_random_capacity(G_nodes, args.router_cap_distr, args.router_factor, args.router_cap_p1, args.router_cap_p2,
                                args.device_cap_distr, args.device_cap_p1, args.device_cap_p2,
                                args.edge_server_cap_distr, args.edge_server_cap_p1, args.edge_server_cap_p2, self.rnd)

        '''
        Generate application graph
        '''

        # G_app = program_random_graph(args.num_roots, args.num_operators) # the number of operators in the returned graph possibly not equal to num_operators
        G_app = program_linear_graph(args.num_operators)
        program_random_requirement(G_app, args.op_comp_distr, args.op_comp_p1, args.op_comp_p2,
                                   args.stream_byte_distr, args.stream_byte_p1, args.stream_byte_p2, self.rnd)

        self.node_s = [n for n in G_app.nodes if G_app.in_degree(n) == 0][0]
        self.node_t = [n for n in nx.descendants(G_app, self.node_s) if G_app.out_degree(n) == 0][0]
        self.network = Network(G_nodes, G_domain, pos_node)
        self.program = Program(G_app, self.node_s, self.node_t)

        self.network.draw_nodes(True)

        self.source = self.rnd.choice(self.network.end_devices)
        self.target = self.rnd.choice(self.network.end_devices)

        if self.args.alg == 'heuristic':
            self.mapper = heuMapper(self.program, self.network, {self.node_s: self.source, self.node_t: self.target},
                                    self.node_s, self.node_t)
            print(self.mapper.map(self.args.num_heuristic_restriction, self.args.num_tries))






