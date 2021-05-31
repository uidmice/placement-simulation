import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def network_edge_server_random_placement(G_nodes, pos_node, T_percentage, S_percentage, L_percentage, rnd=None):
    if not rnd:
        rnd = np.random.RandomState(3)
    assert  isinstance(rnd, np.random.RandomState)
    T_node = [n for n in G_nodes.nodes if G_nodes.nodes[n]['type']=='transit']
    S_node = [n for n in G_nodes.nodes if G_nodes.nodes[n]['type'] == 'stub']
    L_node = [n for n in G_nodes.nodes if G_nodes.nodes[n]['type'] == 'gateway']
    nT = int(len(T_node) * T_percentage)
    nS = int(len(S_node) * S_percentage)
    nL = int(len(L_node) * L_percentage)
    T_selected = rnd.choice(T_node, nT, False)
    S_selected = rnd.choice(S_node, nS, False)
    L_selected = rnd.choice(L_node, nL, False)
    N = len(G_nodes.nodes)
    for i, n in enumerate(T_selected):
        G_nodes.add_node(N + i, type='edge', domain=G_nodes.nodes[n]['domain'])
        G_nodes.add_edge(n, N+i, type='T', weight=0, bw=np.infty)

        pos_node[N+i] = pos_node[n]
    N = len(G_nodes.nodes)
    for i, n in enumerate(S_selected):
        G_nodes.add_node(N + i, type='edge', domain=G_nodes.nodes[n]['domain'])
        G_nodes.add_edge(n, N+i, type='S', weight=0, bw=np.infty)
        pos_node[N+i] = pos_node[n]

    N = len(G_nodes.nodes)
    for i, n in enumerate(L_selected):
        G_nodes.add_node(N + i, type='edge', domain=G_nodes.nodes[n]['domain'])
        G_nodes.add_edge(n, N + i, type='L', weight=0, bw=np.infty)
        pos_node[N+i] = pos_node[n]


def network_random_topology(T, NT, S, NS, L, NL, ET, ES, EST, ETT, ELS, rnd = None):
    if not rnd:
        rnd = np.random.RandomState(3)
    assert isinstance(rnd, np.random.RandomState)

    Nr = T * NT * (1 + S * NS)
    Nh = T * NT * S * NS * L * NL

    link_weight_factor = 10
    link_bandwidth_range_WAN = [50, 100, 150, 200] # mbps
    link_bandwidth_range_MAN = [50, 70]  # mbps
    link_bandwidth_range_LAN = [500, 1000, 1500]  # mbps

    X = int(np.sqrt(Nr + Nh))*10
    transit_domain_size = X/T/T/4
    transit_stub_distance = transit_domain_size/2
    stub_domain_size = np.sqrt((Nr + Nh)/T/NT/S)/4*10
    stub_LAN_distance = stub_domain_size/2
    LAN_size = stub_LAN_distance
    G_domain, pos_domain = _network_random_connected_graph(T, ETT, ((0, 0), (X, X)), rnd)
    G_nodes = nx.Graph()
    pos_node = {}

    n_transit_node_per_transit_domain = {}
    n_stub_domain_per_transit_node = {}
    n_LAN_per_stub_node = {}

    transit_domains = {}
    region_stub_domain = {}
    stub_domains = {}
    region_LANs = {}
    graphs_transit_domain = {}
    graphs_stub_domain = {}
    graphs_LAN = {}

    tn = _network_random_positive_numbers(NT, T, rnd)
    node_count = 0
    domain_count = T
    for i, n in enumerate(G_domain.nodes):
        n_transit_node_per_transit_domain[n] = tn[i]
        transit_domains[n] = _network_region(pos_domain[n], transit_domain_size, X)
        G_domain.nodes[n]['type'] = 'transit'
        G, pos = _network_random_connected_graph(n_transit_node_per_transit_domain[n], ET, transit_domains[n], rnd)
        for e in G.edges:
            G.edges[e]['type'] = 'T'
            G.edges[e]['weight'] = link_weight_factor * np.sqrt((pos[e[0]][0] - pos[e[1]][0])**2 + (pos[e[0]][1] - pos[e[1]][1])**2)
            G.edges[e]['bw'] = rnd.choice(link_bandwidth_range_WAN)

        mapping = {a: a + node_count for a in G.nodes}
        G = nx.relabel_nodes(G, mapping)
        for node in G.nodes:
            pos_node[node] = pos[node - node_count]
            G.nodes[node]['type'] = 'transit'
            G.nodes[node]['domain'] = n
            region_stub_domain[node] = _network_region(pos_node[node], transit_stub_distance, X)
        graphs_transit_domain[n] = G
        G_nodes = nx.compose(G_nodes, G)
        node_count += n_transit_node_per_transit_domain[n]
    for e in G_domain.edges:
        n1 = rnd.choice(list(graphs_transit_domain[e[0]].nodes))
        n2 = rnd.choice(list(graphs_transit_domain[e[1]].nodes))
        G_nodes.add_edge(n1, n2, type='TT', weight = link_weight_factor * np.sqrt((pos_node[n1][0] -pos_node[n2][0] )**2 + (pos_node[n1][1] -pos_node[n2][1] )**2),
                         bw=rnd.choice(link_bandwidth_range_WAN))
        G_domain.edges[e]['type'] = 'TT'

    # nx.draw(G_nodes, pos=pos_node)
    # plt.figure()
    # nx.draw(G_domain, pos=pos_domain)
    # print(G_nodes.nodes.data())
    # print(G_domain.nodes.data())
    # print(G_nodes.edges.data())
    # print(G_domain.edges.data())

    # print(list(region_stub_domain.keys()))


    transit_nodes = list(region_stub_domain.keys())
    ns = _network_random_positive_numbers(S, T * NT, rnd)
    ns_node = _network_random_positive_numbers(NS, S * T * NT, rnd)
    nts = _network_random_positive_numbers(EST, S * T * NT, rnd) - 1
    count = 0
    for i, n in enumerate(transit_nodes):
        n_stub_domain_per_transit_node[n] = ns[i]
        graphs_stub_domain[n] = []
        for j in range(ns[i]):
            pos_domain[domain_count] = (rnd.uniform(region_stub_domain[n][0][0], region_stub_domain[n][1][0]),
                                        rnd.uniform(region_stub_domain[n][0][1], region_stub_domain[n][1][1]))
            stub_domains[domain_count] = _network_region(pos_domain[domain_count], stub_domain_size, X)
            G_domain.add_edge(domain_count, G_nodes.nodes[n]['domain'], type='TS')
            G_domain.nodes[domain_count]['type'] = 'stub'
            G, pos = _network_random_connected_graph(ns_node[count], ES, stub_domains[domain_count], rnd)
            for e in G.edges:
                G.edges[e]['type'] = 'S'
                G.edges[e]['weight'] = link_weight_factor * np.sqrt(
                    (pos[e[0]][0] - pos[e[1]][0]) ** 2 + (pos[e[0]][1] - pos[e[1]][1]) ** 2)
                G.edges[e]['bw'] = rnd.choice(link_bandwidth_range_MAN)
            mapping = {a: a + node_count for a in G.nodes}
            G = nx.relabel_nodes(G, mapping)
            for node in G.nodes:
                pos_node[node] = pos[node - node_count]
                G.nodes[node]['type'] = 'stub'
                G.nodes[node]['domain'] = domain_count
                region_LANs[node] = _network_region(pos_node[node], stub_LAN_distance, X)

            graphs_stub_domain[n].append(G)
            G_nodes = nx.compose(G_nodes, G)
            rnd_node = rnd.choice(list(G.nodes))
            G_nodes.add_edge(n, rnd_node, type='TS', weight= link_weight_factor * np.sqrt((pos_node[rnd_node][0] - pos_node[n][0])**2 + (pos_node[rnd_node][1] - pos_node[n][1])**2),
                             bw=rnd.choice(link_bandwidth_range_WAN))
            while nts[count] > 0:
                node2 = rnd.choice(transit_nodes)
                node1 = rnd.choice(list(G.nodes))
                dis = np.sqrt(
                    (pos_node[node1][0] - pos_node[node2][0]) ** 2 + (pos_node[node1][1] - pos_node[node2][1]) ** 2)
                if dis < transit_stub_distance * 2:
                    G_nodes.add_edge(node1, node2, type='TS', weight= link_weight_factor * np.sqrt((pos_node[node1][0] - pos_node[node2][0])**2 + (pos_node[node1][1] - pos_node[node2][1])**2),
                                     bw=rnd.choice(link_bandwidth_range_WAN))
                    G_domain.add_edge(domain_count, G_nodes.nodes[node2]['domain'], type='TS')
                    nts[count] -= 1
            node_count += ns_node[count]
            count += 1
            domain_count += 1

    # nx.draw(G_nodes, pos=pos_node)
    # plt.figure()
    # nx.draw(G_domain, pos=pos_domain)
    # print(G_nodes.nodes.data())
    # print(G_domain.nodes.data())
    # print(G_nodes.edges.data())
    # print(G_domain.edges.data())
    # plt.show()

    stub_nodes = region_LANs.keys()
    nl = _network_random_positive_numbers(L, S * T * NT * NS, rnd)
    nl_node = _network_random_positive_numbers(NL, sum(nl), rnd)
    nl_edge = _network_random_positive_numbers(ELS, sum(nl), rnd) - 1
    count = 0
    for i, n in enumerate(stub_nodes):
        n_LAN_per_stub_node[n] = nl[i]
        graphs_LAN[n] = []
        for j in range(nl[i]):
            pos_domain[domain_count] = (rnd.uniform(region_LANs[n][0][0], region_LANs[n][1][0]),
                                        rnd.uniform(region_LANs[n][0][1], region_LANs[n][1][1]))
            stub_domains[domain_count] = _network_region(pos_domain[domain_count], LAN_size, X)
            G_domain.add_edge(domain_count, G_nodes.nodes[n]['domain'], type='SL')
            G_domain.nodes[domain_count]['type'] = 'lan'

            G = nx.Graph()
            G.add_node(node_count, type='gateway', domain=domain_count)
            router_loc = pos_domain[domain_count]
            pos_node[node_count] = router_loc
            for k in range(nl_node[count]):
                G.add_node(node_count + k + 1, type='host', domain=domain_count)
                G.add_edge(node_count, node_count + k + 1, type='L', bw=rnd.choice(link_bandwidth_range_LAN))
                node_loc = (
                    rnd.uniform(stub_domains[domain_count][0][0], stub_domains[domain_count][1][0]),
                    rnd.uniform(stub_domains[domain_count][0][1], stub_domains[domain_count][1][1]))
                pos_node[node_count + k + 1] = node_loc
                G[node_count][node_count + k + 1]['weight'] = link_weight_factor * np.sqrt((node_loc[0]-router_loc[0])**2 + (node_loc[1]-router_loc[1])**2)
            graphs_LAN[n].append(G)
            G_nodes = nx.compose(G_nodes, G)
            G_nodes.add_edge(n, node_count, type='SL', weight=link_weight_factor * np.sqrt((pos_node[node_count][0] - pos_node[n][0])**2 + (pos_node[node_count][1] - pos_node[n][1])**2),
                             bw=rnd.choice(link_bandwidth_range_LAN))
            node_count += nl_node[count] + 1
            count += 1
            domain_count += 1
    return G_domain, G_nodes, pos_domain, pos_node


def _network_random_connected_graph(n, en, region, rnd):
    G_t = nx.gnp_random_graph(n, en / n, seed=rnd)
    sub = list(nx.connected_components(G_t))
    for i in range(len(sub) - 1):
        G_t.add_edge(rnd.choice(tuple(sub[i])), rnd.choice(tuple(sub[i + 1])))
    assert nx.is_connected(G_t)
    pos_t = {}
    for n in G_t.nodes:
        pos_t[n] = (rnd.uniform(region[0][0], region[1][0]), rnd.uniform(region[0][1], region[1][1]))
    return G_t, pos_t


def _network_random_positive_numbers(mu, N, rnd):
    if mu == 1:
        return np.ones(N).astype(int)
    if N ==1:
        return np.array([int(mu)])
    rt = rnd.normal(mu, mu / 3, N).astype(int)
    rt[rt < 1] = 1
    makeup = N * mu - np.sum(rt)
    if makeup < 0:
        idx = [i for i, a in enumerate(rt) if a > 1]
        while len(idx) > 0 and makeup < 0:
            rt[rnd.choice(idx)] -= 1
            idx = [i for i, a in enumerate(rt) if a > 1]
            makeup += 1
    elif makeup > 0:
        idx = [i for i in range(N)]
        while makeup > 0:
            rt[rnd.choice(idx)] += 1
            makeup -= 1
    return rt


def _network_region(center, distance, X):
    x_min = max(0, center[0] - distance)
    x_max = min(X, center[0] + distance)
    y_min = max(0, center[1] - distance)
    y_max = min(X, center[1] + distance)
    return ((x_min, y_min), (x_max, y_max))


def _network_router_processing_time_uniform_distr(G_nodes, processing_time_range_lower, processing_time_range_upper, rnd):
    # time in us
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] in ['transit', 'stub', 'gateway']:
            G_nodes.nodes[n]['rate'] = 1000000/rnd.uniform(processing_time_range_lower, processing_time_range_upper)

def _network_router_processing_time_normal_distr(G_nodes, processing_time_mean, processing_time_u, rnd):
    # time in us
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] in ['transit', 'stub', 'gateway']:
            G_nodes.nodes[n]['rate'] = 1000000/max(1, rnd.normal(processing_time_mean, processing_time_u))

def _network_router_processing_rate_uniform_distr(G_nodes, processing_rate_range_lower, processing_rate_range_upper,  rnd):
    # rate in number of packets per second
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] in ['transit', 'stub', 'gateway']:
            G_nodes.nodes[n]['rate'] = rnd.uniform(processing_rate_range_lower, processing_rate_range_upper)

def _network_router_processing_rate_normal_distr(G_nodes, processing_rate_mean, processing_rate_u, rnd):
    # rate in number of packets per second
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] in ['transit', 'stub', 'gateway']:
            G_nodes.nodes[n]['rate'] = max(1, rnd.normal(processing_rate_mean, processing_rate_u))


def _network_device_compute_rate_uniform_distr(G_nodes, compute_rate_range_lower, compute_rate_range_upper,  rnd):
    # rate in million instructions per second (MIPS)
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] =='host':
            G_nodes.nodes[n]['rate'] = rnd.uniform(compute_rate_range_lower, compute_rate_range_upper)

def _network_device_compute_rate_normal_distr(G_nodes, compute_rate_mean, compute_rate_u, rnd):
    # rate in million instructions per second (MIPS)
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] =='host':
            G_nodes.nodes[n]['rate'] = max(1, rnd.normal(compute_rate_mean, compute_rate_u))

def _network_edge_server_compute_rate_uniform_distr(G_nodes, compute_rate_range_lower, compute_rate_range_upper,  rnd):
    # rate in million instructions per second (MIPS)
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] =='edge':
            G_nodes.nodes[n]['rate'] = rnd.uniform(compute_rate_range_lower, compute_rate_range_upper)

def _network_edge_server_compute_rate_normal_distr(G_nodes, compute_rate_mean, compute_rate_u, rnd):
    # rate in million instructions per second (MIPS)
    for n in G_nodes.nodes:
        if G_nodes.nodes[n]['type'] =='edge':
            G_nodes.nodes[n]['rate'] = max(1, rnd.normal(compute_rate_mean, compute_rate_u))

def network_random_capacity(G_nodes, router_cap_distr, router_factor, rp1, rp2, device_cap_distr, dp1, dp2, server_cap_distr, sp1, sp2, rnd=None):
    if not rnd:
        rnd = np.random.RandomState(3)
    assert isinstance(rnd, np.random.RandomState)
    supported_distribution = ['uniform', 'normal']
    router_factors = ['proc_time', 'rate']
    assert router_cap_distr in supported_distribution, str(router_cap_distr) + ' distribution not supported'
    assert server_cap_distr in supported_distribution, str(router_cap_distr) + ' distribution not supported'
    assert device_cap_distr in supported_distribution, str(router_cap_distr) + ' distribution not supported'
    assert router_factor in router_factors, "Setting router capacity based on " + str(router_factor) + ' not supported'
    assert all([n > 0 for n in [rp1, rp2, dp1, dp2, sp1, sp2]]), "Parameters should be all positive"
    if router_cap_distr == 'uniform':
        if router_factor == 'proc_time':
            _network_router_processing_time_uniform_distr(G_nodes, rp1, rp2, rnd)
        else:
            _network_router_processing_rate_uniform_distr(G_nodes, rp1, rp2, rnd)
    else:
        if router_factor == 'proc_time':
            _network_router_processing_time_normal_distr(G_nodes, rp1, rp2, rnd)
        else:
            _network_router_processing_rate_normal_distr(G_nodes, rp1, rp2, rnd)

    if device_cap_distr == 'uniform':
        _network_device_compute_rate_uniform_distr(G_nodes, dp1, dp2, rnd)
    else:
        _network_device_compute_rate_normal_distr(G_nodes, dp1, dp2, rnd)

    if server_cap_distr == 'uniform':
        _network_edge_server_compute_rate_uniform_distr(G_nodes, sp1, sp2, rnd)
    else:
        _network_edge_server_compute_rate_normal_distr(G_nodes, sp1, sp2, rnd)


def network_distance_plot(network):
    plt.figure()
    delay_in_LAN = [0] * len(network.end_devices)
    for i in range(len(network.end_devices)):
        n = network.end_devices[i]
        domain = network.get_domain_id(n)
        nodes = network.domains[domain].nodes
        delay_in_LAN[i] = np.average([network.latency_between_nodes(n, tn, 20) for tn in nodes if tn != n])

    delay_in_MAN = [0] * len(network.end_devices)
    delay_in_WAN = [0] * len(network.end_devices)
    delay_from_LAN_to_MAN = [0] * len(network.end_devices)
    delay_from_LAN_to_WAN = [0] * len(network.end_devices)

    for i in range(len(network.end_devices)):
        n = network.end_devices[i]
        lan = network.get_domain_id(n)
        parent_domain = network.get_parent_domain_id(lan)[0]
        delay_from_LAN_to_MAN[i] = network.latency_from_node_to_domain(n, parent_domain, 20)
        MAN_domains = network.children_lan_domain(parent_domain)
        delay_in_MAN[i] = np.average([network.latency_from_node_to_domain(n, d, 20) for d in MAN_domains if d != lan])

        parent_parent = network.get_parent_domain_id(parent_domain)[0]
        delay_from_LAN_to_WAN[i] = network.latency_from_node_to_domain(n, parent_parent, 20)
        WAN_domains = network.children_lan_domain(parent_parent)
        delay_in_WAN[i] = np.average(
            [network.latency_from_node_to_domain(n, d, 20) for d in WAN_domains if d not in MAN_domains])

    lim = max(delay_in_WAN)
    plt.hist(delay_in_LAN, bins=np.arange(0, lim, lim / 100), edgecolor='black', linewidth=0.8, fc=(0, 1, 1, 0.5),
             weights=np.ones(len(delay_in_LAN)) / len(delay_in_LAN), label=r'$L_{LDD}$')
    plt.hist(delay_from_LAN_to_MAN, bins=np.arange(0, lim, lim / 100), edgecolor='black', linewidth=0.8,
             fc=(1, 0, 0, 0.5), weights=np.ones(len(delay_in_WAN)) / len(delay_in_WAN), label=r'$L_{MDE}$')
    plt.hist(delay_in_MAN, bins=np.arange(0, lim, lim / 100), edgecolor='black', linewidth=0.8, fc=(1, 0, 1, 0.5),
             weights=np.ones(len(delay_in_MAN)) / len(delay_in_MAN), label='$L_{MDD}$')
    plt.hist(delay_from_LAN_to_WAN, bins=np.arange(0, lim, lim / 100), edgecolor='black', linewidth=0.8,
             fc=(0, 1, 0, 0.5), weights=np.ones(len(delay_in_WAN)) / len(delay_in_WAN), label='$L_{WDE}$')
    plt.hist(delay_in_WAN, bins=np.arange(0, lim, lim / 100), edgecolor='black', linewidth=0.8, fc=(1, 1, 0, 0.5),
             weights=np.ones(len(delay_in_WAN)) / len(delay_in_WAN), label='$L_{WDD}$')

    plt.xlabel('Delay(ms)')
    plt.legend()
    plt.show()


def program_random_graph(n_roots, n_nodes, p = 0.5):
    assert n_roots < n_nodes, "#of roots should be less than # of nodes"
    G = nx.gnp_random_graph(n_nodes, p, directed=True)
    DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    roots = [node for node in DAG.nodes() if DAG.in_degree(node) == 0]

    if len(roots) > n_roots:
        for i in range(len(roots) - n_roots):
            DAG.add_edge(roots[i], roots[i+1])
    elif len(roots) < n_roots:
        for i in range(n_roots - len(roots)):
            non_root = [node for node in DAG.nodes() if DAG.in_degree(node) != 0]
            DAG.add_edge(len(DAG.nodes())+i, non_root[np.random.randint(0, len(non_root))])
    return DAG

def program_linear_graph(num_operators):
    G = nx.DiGraph()
    for i in range(num_operators - 1):
        G.add_edge(i, i + 1)
    return G

def _program_operator_comp_uniform_distr(G, comp_range_lower, comp_range_upper, rnd):
    for n in G.nodes:
        G.nodes[n]['computation'] = int(rnd.uniform(comp_range_lower,comp_range_upper))


def _program_operator_comp_normal_distr(G, comp_mean, comp_u, rnd):
    for n in G.nodes:
        G.nodes[n]['computation'] = max(100, int(rnd.normal(comp_mean, comp_u)))


def _program_stream_bytes_uniform_distr(G, byte_range_lower, byte_range_upper, rnd):
    for e in G.edges:
        G.edges[e]['bytes'] = int(rnd.uniform(byte_range_lower, byte_range_upper))


def _program_stream_bytes_normal_distr(G, mean, u, rnd):
    for e in G.edges:
        G.edges[e]['bytes'] = max(1, int(rnd.normal(mean, u)))


def program_random_requirement(G_app, op_comp_distr, op1, op2, stream_byte_distr, sp1, sp2, rnd=None):
    if not rnd:
        rnd = np.random.RandomState(3)
    assert isinstance(rnd, np.random.RandomState)
    supported_distribution = ['uniform', 'normal']
    assert op_comp_distr in supported_distribution, str(op_comp_distr) + ' distribution not supported'
    assert stream_byte_distr in supported_distribution, str(stream_byte_distr) + ' distribution not supported'
    assert all([n > 0 for n in [op1, op2, sp1, sp2]]), "Parameters should be all positive"

    if op_comp_distr == 'uniform':
        _program_operator_comp_uniform_distr(G_app, op1, op2, rnd)
    else:
        _program_operator_comp_normal_distr(G_app, op1, op2, rnd)

    if stream_byte_distr == 'uniform':
        _program_stream_bytes_uniform_distr(G_app, sp1, sp2, rnd)
    else:
        _program_stream_bytes_normal_distr(G_app, sp1, sp2, rnd)
    return