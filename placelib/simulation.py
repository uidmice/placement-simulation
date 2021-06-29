import time
from placelib.util import *
from models.program import Program
from models.network import Network
<<<<<<< Updated upstream
from placelib.mapper import heuMapper
=======
from placelib.mapper import heuMapper, exhMapper
import models.network
>>>>>>> Stashed changes

import time

class Simulation:
    '''
        The Simulation object contains is an instance of the simulation

        Parameters
        ----------
        args : namespace object
            The populated namespace from argument input

        Attributes
        ----------
        network : Network
            An instance of the network model created using the parameters provided in the args
        program : Program
            An instance of the program model
        args : namespace object
            This is where we store args
        rnd : np.RandomState
            Container for the pseudo-random number generator used for the whole simulation
        node_s : int
            Operator id of the operator source (randomly picked from the set of the roots)
        node_t : int
            Operator id of the operator sink (randomly picked)
        source : int
            Id of the device where node_s is pinned (currently it is randomly picked from the network)
        target : int
            Id of the device where node_t is pinned (currently it is randomly picked from the network)
        mapper : Mapper
            An instance of a mapper created based on the algorithm selected
    '''
    def __init__(self, args, seed):

        self.rnd = np.random.RandomState(seed)
        self.args = args

        # Generate network graph
        G_domain, G_nodes, pos_domain, pos_node = network_random_topology(T=1, NT=3, S=5, NS=1, L=2, NL=4, ET=3, ES=2, EST=1, ETT=2, ELS=1, rnd=self.rnd)

        network_edge_server_random_placement(G_nodes, pos_node, args.edge_server_transit_percentage, args.edge_server_stub_percentage, args.edge_server_gateway_percentage, self.rnd)
        network_random_capacity(G_nodes, args.router_cap_distr, args.router_factor, args.router_cap_p1, args.router_cap_p2,
                                args.device_cap_distr, args.device_cap_p1, args.device_cap_p2,
                                args.edge_server_cap_distr, args.edge_server_cap_p1, args.edge_server_cap_p2, self.rnd)


        # Generate application graph
        # G_app = program_random_graph(args.num_roots, args.num_operators) # the number of operators in the returned graph possibly not equal to num_operators
        G_app = program_linear_graph(args.num_operators)
        program_random_requirement(G_app, args.op_comp_distr, args.op_comp_p1, args.op_comp_p2,
                                   args.stream_byte_distr, args.stream_byte_p1, args.stream_byte_p2, self.rnd)

        self.node_s = [n for n in G_app.nodes if G_app.in_degree(n) == 0][0]
        self.node_t = [n for n in nx.descendants(G_app, self.node_s) if G_app.out_degree(n) == 0][0]
        
        # print("app.nodes: ", G_app.nodes)
        # print("G_nodes: ")
        # repr(G_nodes)
        # print("node_s: ", G_app.nodes[self.node_s])
        # print("node_t: ", G_app.nodes[self.node_t])
        self.network = Network(G_nodes, G_domain, pos_node)
        self.program = Program(G_app, self.node_s, self.node_t)
        
        # self.network.draw_nodes(True) # comment out for testing runtime
        
        self.source = self.rnd.choice(self.network.end_devices)
        self.target = self.rnd.choice(self.network.end_devices)
<<<<<<< Updated upstream
        print("self.source: ", self.source)
        print("self.target: ", self.target)
        # print("node_s: ", self.node_s)
        # print("node_t: ", self.node_t)
        print("\n", file=open("sim_results.txt", "a"))
        print("self.source: ", self.source, file=open("sim_results.txt", "a"))
        print("self.target: ", self.target, file=open("sim_results.txt", "a"))
=======
        
        print("\n", file=open("sim_results.txt", "a"))
        print("source: ", self.source, file=open("sim_results.txt", "a"))
        print("target: ", self.target, file=open("sim_results.txt", "a"))
        
        
>>>>>>> Stashed changes
        # time2 = time.time()
        # print("generate time = ", time2 - time1)
        
        print("\n", file=open("sim_results.txt", "a"))
        print("self.source: ", self.source, file=open("sim_results.txt", "a"))
        print("self.target: ", self.target, file=open("sim_results.txt", "a"))

        if self.args.alg == 'heuristic':
            self.mapper = heuMapper(self.program, self.network, self.node_s, self.node_t,
                                    {self.node_s: self.source, self.node_t: self.target})
            
        else:
            raise NotImplementedError

    def map(self):
        '''Run the mapper and return the mapped result (a dict {operator : device}) and the runtime'''
        time1 = time.time()
        if self.args.alg == 'heuristic':
            map = self.mapper.map(self.args.num_heuristic_restriction, self.args.num_tries)
        else:
            raise NotImplementedError
        time2 = time.time()
        return map, time2 - time1

    def evaluate(self, map, average=True):
        """ Return the end to end latency between the source and target as a result of the mapping"""
        times_called = 0
<<<<<<< Updated upstream
        delay =  self.mapper.evaluate(map, average, times_called)
=======
        delay = self.mapper.evaluate(map, average, times_called)
>>>>>>> Stashed changes
        times_called += 1
        return delay










