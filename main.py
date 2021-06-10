from placelib.simulation import Simulation

class Arguments:
        def __init__(self):
                # Network parameters
                self.edge_server_transit_percentage = 0.1
                self.edge_server_stub_percentage = 0.3
                self.edge_server_gateway_percentage = 0.4

                self.router_cap_dist = 'uniform'
                self.router_factor = 'proc_time'
                self.device_cap_dist = 'uniform'
                self.edge_server_dist = 'uniform'
                self.router_proc_time_lower = 5 #us
                self.router_proc_time_upper = 15 #us
                self.device_compute_lower = 2 #MIPS
                self.device_compute_upper = 5 #MIPS
                self.edge_compute_lower = 8 #MIPS
                self.edge_compute_upper = 12 #MIPS

                self.num_operators = 9
                self.num_roots = 1
                self.op_comp_distr = 'uniform'
                self.stream_byte_distr = 'uniform'
                self.op_comp_lower = 5000
                self.op_comp_upper = 100000
                self.stream_byte_lower = 10
                self.stream_byte_upper = 60000

args = Arguments()
simulation = Simulation(args)







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