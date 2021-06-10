from placelib.util import *
from models.program import Program
from models.network import Network
from placelib.mapper import heuMapper


class Simulation:
    def __init__(self, args, seed=3):

        self.rnd = np.random.RandomState(seed)
        self.args = args

        '''
        Generate network graph
        '''
        G_domain, G_nodes, pos_domain, pos_node = network_random_topology(T=1, NT=3, S=5, NS=1, L=2, NL=4, ET=3, ES=2, EST=1, ETT=2, ELS=1, rnd=self.rnd)

        network_edge_server_random_placement(G_nodes, pos_node, args.edge_server_transit_percentage, args.edge_server_stub_percentage, args.edge_server_gateway_percentage, self.rnd)
        network_random_capacity(G_nodes, args.router_cap_dist, args.router_factor, args.router_proc_time_lower, args.router_proc_time_upper,
                                args.device_cap_dist, args.device_compute_lower, args.device_compute_upper,
                                args.edge_server_dist, args.edge_compute_lower, args.edge_compute_upper, self.rnd)

        '''
        Generate application graph
        '''

        # G_app = program_random_graph(args.num_roots, args.num_operators) # the number of operators in the returned graph possibly not equal to num_operators
        G_app = program_linear_graph(args.num_operators)
        program_random_requirement(G_app, args.op_comp_distr, args.op_comp_lower, args.op_comp_upper,
                                   args.stream_byte_distr, args.stream_byte_lower, args.stream_byte_upper, self.rnd)

        self.node_s = [n for n in G_app.nodes if G_app.in_degree(n) == 0][0]
        self.node_t = [n for n in nx.descendants(G_app, self.node_s) if G_app.out_degree(n) == 0][0]
        self.network = Network(G_nodes, G_domain, pos_node)
        self.program = Program(G_app, self.node_s, self.node_t)



        self.source = self.rnd.choice(self.network.end_devices)
        self.target = self.rnd.choice(self.network.end_devices)

        self.mapper = heuMapper(self.program, self.network, {self.node_s: self.source, self.node_t: self.target})
        print(self.mapper.map(10))





