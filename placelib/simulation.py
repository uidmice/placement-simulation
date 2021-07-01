import time
import os
import datetime
from placelib.util import *
from models.program import Program
from models.network import Network
from placelib.mapper import heuMapper, exhMapper


import cProfile, pstats
import pickle


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
        name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.alg}'
        if args.logdir_suffix:
            name += f'_{args.logdir_suffix}'
        self.logdir = os.path.join(
            args.logdir, name
        )
        self.logdir = os.path.join(self.logdir, f'seed_{seed}')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)


        # Generate network graph
        G_domain, G_nodes, pos_domain, pos_node = network_random_topology(args.T, args.NT, args.S, args.NS, args.L, args.NL, ET=3, ES=2, EST=1, ETT=2, ELS=1, rnd=self.rnd)

        network_edge_server_random_placement(G_nodes, pos_node, args.edge_server_transit_percentage, args.edge_server_stub_percentage, args.edge_server_gateway_percentage, self.rnd)
        network_random_capacity(G_nodes, args.router_cap_distr, args.router_factor, args.router_cap_p1, args.router_cap_p2,
                                args.device_cap_distr, args.device_cap_p1, args.device_cap_p2,
                                args.edge_server_cap_distr, args.edge_server_cap_p1, args.edge_server_cap_p2, self.rnd)
        self.network = Network(G_nodes, G_domain, pos_node)
        self.source = self.rnd.choice(self.network.end_devices)
        self.target = self.rnd.choice(self.network.end_devices)

        pickle.dump([self.network,self.source, self.target], open(os.path.join(self.logdir, 'network.pkl'), 'wb'))



        # Generate application graph
        # G_app = program_random_graph(args.num_roots, args.num_operators) # the number of operators in the returned graph possibly not equal to num_operators
        G_app = program_linear_graph(args.num_operators)
        program_random_requirement(G_app, args.op_comp_distr, args.op_comp_p1, args.op_comp_p2,
                                   args.stream_byte_distr, args.stream_byte_p1, args.stream_byte_p2, self.rnd)

        self.node_s = [n for n in G_app.nodes if G_app.in_degree(n) == 0][0]
        self.node_t = [n for n in nx.descendants(G_app, self.node_s) if G_app.out_degree(n) == 0][0]
        self.program = Program(G_app, self.node_s, self.node_t)
        pickle.dump([self.program,self.node_s, self.node_t], open(os.path.join(self.logdir, 'program.pkl'), 'wb'))

        for n in self.network.end_devices + self.network.edge_servers:
            if G_nodes.degree(n) != 1:
                print(f'Node {n} has degree {G_nodes.degree(n)}')

        if self.args.alg == 'heuristic':
            self.mapper = heuMapper(self.program, self.network, self.node_s, self.node_t,
                                    {self.node_s: self.source, self.node_t: self.target}, self.args.num_tries)
        elif self.args.alg == 'exhaustive':
            self.mapper = exhMapper(self.program, self.network, self.node_s, self.node_t,
                                    {self.node_s: self.source, self.node_t: self.target})
        else:
            raise NotImplementedError

    def map(self):
        '''Run the mapper and return the mapped result (a dict {operator : device}) and the runtime'''
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        profiler = cProfile.Profile()
        profiler.enable()
        if self.args.alg == 'heuristic':
            mapping = self.mapper.map(self.args.num_heuristic_restriction, self.args.num_tries)
        elif self.args.alg == 'exhaustive':
            mapping = self.mapper.map()
        else:
            raise NotImplementedError
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime').strip_dirs()
        stats.dump_stats(os.path.join(self.logdir, f'{now}_profile_stats'))
        t = 0
        for func in stats.stats.keys():
            if 'map' in func:
                t = stats.stats[func][3]
        pickle.dump([mapping, t], open(os.path.join(self.logdir, f'{now}_results.pkl'), 'wb'))
        return mapping, t

    def evaluate(self, map, average=True):
        """ Return the end to end latency between the source and target as a result of the mapping"""
        delay = self.mapper.evaluate(map, average)
        return delay










