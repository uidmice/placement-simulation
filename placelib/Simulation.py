from placelib.util import *
from placelib.Network import *
from placelib.Program import *


class Simulation:
    def __init__(self, seed=3):
        rnd = np.random.RandomState(seed)

        '''
        Generate network graph
        '''
        G_domain, G_nodes, pos_domain, pos_node = network_random_topology(T=1, NT=3, S=5, NS=1, L=2, NL=4, ET=3, ES=2, EST=1, ETT=2, ELS=1, rnd=rnd)

        # Network parameters
        edge_server_transit_percentage = 0.1
        edge_server_stub_percentage = 0.3
        edge_server_gateway_percentage = 0.4

        router_cap_dist = 'uniform'
        router_factor = 'proc_time'
        device_cap_dist = 'uniform'
        edge_server_dist = 'uniform'
        router_proc_time_lower = 5 #us
        router_proc_time_upper = 15 #us
        device_compute_lower = 2 #MIPS
        device_compute_upper = 5 #MIPS
        edge_compute_lower = 8 #MIPS
        edge_compute_upper = 12 #MIPS

        network_edge_server_random_placement(G_nodes, pos_node, edge_server_transit_percentage, edge_server_stub_percentage, edge_server_gateway_percentage, rnd)
        network_random_capacity(G_nodes, router_cap_dist, router_factor, router_proc_time_lower, router_proc_time_upper,
                                device_cap_dist, device_compute_lower, device_compute_upper,
                                edge_server_dist, edge_compute_lower, edge_compute_upper, rnd)
        # print(G_nodes.nodes.data())
        # print(G_nodes.edges.data())
        # print(G_domain.nodes.data())
        # print(G_domain.edges.data())

        network = Network(G_nodes, G_domain, pos_node)

        '''
        Generate application graph
        '''
        num_operators = 10
        # num_roots = 2
        op_comp_distr = 'uniform'
        stream_byte_distr = 'uniform'
        op_comp_lower = 5000
        op_comp_upper = 100000
        stream_byte_lower = 10
        stream_byte_upper = 60000

        # G_app = program_random_graph(num_roots, num_operators) # the number of operators in the returned graph possibly not equal to num_operators
        G_app = program_linear_graph(num_operators)
        program_random_requirement(G_app, op_comp_distr, op_comp_lower, op_comp_upper,
                                   stream_byte_distr, stream_byte_lower, stream_byte_upper, rnd)

        print(G_app.nodes.data())
        print(G_app.edges.data())
        program = Program(G_app, 0, num_operators - 1)
        program.plot()

