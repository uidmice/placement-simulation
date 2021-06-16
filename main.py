import argparse

from placelib.simulation import Simulation

import time


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def get_args():

    parser = argparse.ArgumentParser(description='Placement Simulation Parameters')
    parser.add_argument('--alg',
                        default='heuristic',
                        help='Placement Algorithm: heuristic, exhaustive, greedy (default: heuristic)')
    parser.add_argument('--logdir',
                        default='runs',
                        help='exterior log directory')
    parser.add_argument('--logdir_suffix',
                        default='',
                        help='log directory suffix')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default 1)')



    ########################## Network Parameters########################

    ## For edge server:
    parser.add_argument('--edge_server_t',
                        type=restricted_float,
                        default=0.1,
                        dest='edge_server_transit_percentage',
                        help='Density of edge server at transit layer [0.0, 1.0] (default: 0.1)')
    parser.add_argument('--edge_server_s',
                        type=restricted_float,
                        default=0.3,
                        dest='edge_server_stub_percentage',
                        help='Density of edge server at stub layer [0.0, 1.0] (default: 0.3)')
    parser.add_argument('--edge_server_lan',
                        type=restricted_float,
                        default=0.3,
                        dest='edge_server_gateway_percentage',
                        help='Density of edge server at bottom layer (LAN) [0.0, 1.0] (default: 0.3)')

    edge_server_cap_distr_group = parser.add_mutually_exclusive_group()
    edge_server_cap_distr_group.add_argument('--edge_server_cap_uniform',
                                       action='store_const',
                                       dest='edge_server_cap_distr',
                                       const='uniform',
                                       default='uniform')
    edge_server_cap_distr_group.add_argument('--edge_server_cap_normal',
                                       action='store_const',
                                       dest='edge_server_cap_distr',
                                       const='normal')
    parser.add_argument('--edge_server_cap_p1',
                        type=float,
                        default=8,
                        help='First parameter for edge server capacity; it is the lower bound if uniform distrribution is selected or mean if normal distribution. The unit is million instructions per second (MIPS)  (default: 8)')
    parser.add_argument('--edge_server_cap_p2',
                        type=float,
                        default=12,
                        help='Second parameter for edge server capacity; it is the upper bound if uniform distribution is selected or mean if normal distribution. The unit is million instructions per second (MIPS)  (default: 12)')

    ## For router:
    router_cap_distr_group = parser.add_mutually_exclusive_group()
    router_cap_factor_group = parser.add_mutually_exclusive_group()
    router_cap_distr_group.add_argument('--router_cap_uniform',
                                       action='store_const',
                                       dest='router_cap_distr',
                                       const='uniform',
                                       default='uniform')
    router_cap_distr_group.add_argument('--router_cap_normal',
                                       action='store_const',
                                       dest='router_cap_distr',
                                       const='normal')
    router_cap_factor_group.add_argument('--router_proc_time',
                                       action='store_const',
                                       dest='router_factor',
                                       const='proc_time',
                                       default='proc_time')
    router_cap_factor_group.add_argument('--router_rate',
                                       action='store_const',
                                       dest='router_factor',
                                       const='rate')
    parser.add_argument('--router_cap_p1',
                        type=float,
                        default=5,
                        help='First parameter for router capacity; it is the lower bound if uniform distribution is selected or mean if normal distribution. The unit is "us" if --router_proc_time (default), or "packet per second" other wise  (default: 5)')
    parser.add_argument('--router_cap_p2',
                        type=float,
                        default=15,
                        help='Second parameter for router capacity; it is the upper bound if uniform distribution is selected or mean if normal distribution. The unit is "us" if --router_proc_time (default), or "packet per second" other wise  (default: 15)')

    ## For end devices:
    device_cap_distr_group = parser.add_mutually_exclusive_group()
    device_cap_distr_group.add_argument('--device_cap_uniform',
                                            action='store_const',
                                            dest='device_cap_distr',
                                            const='uniform',
                                            default='uniform')
    edge_server_cap_distr_group.add_argument('--device_cap_normal',
                                            action='store_const',
                                            dest='device_cap_distr',
                                            const='normal')
    parser.add_argument('--device_cap_p1',
                        type=float,
                        default=2,
                        help='First parameter for end device capacity; it is the lower bound if uniform distribution is selected or mean if normal distribution. The unit is million instructions per second (MIPS)  (default: 2)')
    parser.add_argument('--device_cap_p2',
                        type=float,
                        default=5,
                        help='Second parameter for end device capacity; it is the upper bound if uniform distribution is selected or mean if normal distribution. The unit is million instructions per second (MIPS)  (default: 5)')


    ########################## Program Parameters########################
    parser.add_argument('--num_operators',
                        type=int,
                        default=9,
                        help='Number of operators in the program graph (default: 9)')
    parser.add_argument('--num_roots',
                        type=int,
                        default=1,
                        help='Number of roots in the program graph (default: 1)')
    ## For operators:
    operator_comp_distr_group = parser.add_mutually_exclusive_group()
    operator_comp_distr_group.add_argument('--op_comp_uniform',
                                       action='store_const',
                                       dest='op_comp_distr',
                                       const='uniform',
                                       default='uniform')
    operator_comp_distr_group.add_argument('--op_comp_normal',
                                            action='store_const',
                                            dest='op_comp_distr',
                                            const='normal')
    parser.add_argument('--op_comp_p1',
                        type=float,
                        default=5000,
                        help='First parameter for end device capacity; it is the lower bound if uniform distribution is selected or mean if normal distribution. The unit is number of instructions (default: 5000)')
    parser.add_argument('--op_comp_p2',
                        type=float,
                        default=100000,
                        help='Second parameter for end device capacity; it is the upper bound if uniform distribution is selected or mean if normal distribution. The unit is number of instructions  (default: 100000)')

    ## For streams:
    stream_bytes_distr_group = parser.add_mutually_exclusive_group()
    stream_bytes_distr_group.add_argument('--stream_byte_uniform',
                                          action='store_const',
                                          dest='stream_byte_distr',
                                          const='uniform',
                                          default='uniform')
    stream_bytes_distr_group.add_argument('--stream_byte_normal',
                                          action='store_const',
                                          dest='stream_byte_distr',
                                          const='normal')
    parser.add_argument('--stream_byte_p1',
                        type=float,
                        default=10,
                        help='First parameter for end device capacity; it is the lower bound if uniform distribution is selected or mean if normal distribution. The unit is kbytes (default: 10)')
    parser.add_argument('--stream_byte_p2',
                        type=float,
                        default=60000,
                        help='Second parameter for end device capacity; it is the upper bound if uniform distribution is selected or mean if normal distribution. The unit is kbytes  (default: 60000)')


    ################ heuristic algorithm ####################
    parser.add_argument('--num_heuristic_restriction',
                        type=int,
                        default=5,
                        help='Number of heuristic restrictions applied during mapping (default: 5)')
    parser.add_argument('--num_tries',
                        type=int,
                        default=10,
                        help='Number of tries (default: 10)')

    return parser.parse_args()





if __name__ == '__main__':
        args = get_args()
        print(args)
        print("\n", file=open("sim_results.txt", "a"))
        print(args, file=open("sim_results.txt", "a"))
        time1 = time.time()
        simulation = Simulation(args)
        time2 = time.time()
        print(time2 - time1)
        print("Simulation took:", time2 - time1, "seconds", file=open("sim_results.txt", "a"))





# plt.ion()
# plt.figure(p.fig.number)
# ax = p.fig.add_axes([0,0,1,1])
# mapping = {True: "red", False: 'blue' }
# for i in range(200):
#     p.step()
#     ax.clear()
#     colors = [mapping[p.G.nodes[n]['On']] for n in p.G.nodes]
#     nx.draw(p.G, pos=p.pos, node_color=colors, with_labels=True, font_weight='bold')
#     edge_labels = nx.get_edge_attributes(p.G, 'N')
#     edge_buf = nx.get_edge_attributes(p.G, 'Buffer')
#     node_labels = nx.get_node_attributes(p.G, 'f')
#     offset_labels = nx.get_node_attributes(p.G, 'offset')
#     time_label = ax.text(.1, .8, "t = %d" %(i), ha="center", va="center",transform = ax.transAxes)
#     formatted_edge_labels = {(elem[0], elem[1]): str(edge_buf[elem]) + '/'+str(edge_labels[elem]) for elem in edge_labels}
#     formatted_node_labels = {node: node_labels[node] for node in node_labels}
#     nx.draw_networkx_edge_labels(p.G, p.pos, edge_labels=formatted_edge_labels, font_color='red', label_pos=0.3)
#     nx.draw_networkx_labels(p.G, {a: (p.pos[a][0], p.pos[a][1] + 2) for a in formatted_node_labels},
#                             labels=formatted_node_labels, font_color='blue')
#     nx.draw_networkx_labels(p.G, {a: (p.pos[a][0]-2, p.pos[a][1]) for a in offset_labels},
#                             labels=offset_labels, font_color='red')
#
#     plt.draw()
#     plt.pause(0.05)
#
# p.set_offset([0,20])

# G = nx.gnp_random_graph(20, 0.2)
# plt.subplot()
# nx.draw_planar(G,  with_labels=True, font_weight='bold')
# plt.show()