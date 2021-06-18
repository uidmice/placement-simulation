class Domain:
    '''
            The domain model

            Parameters
            ----------
            id :
                an identifier for the domain object created (unique across all domains)
            type : string
                "transit", "stub", or "lan"

            Attributes
            ----------
            id :
                store id
            nodes :
                a list of node_ids of all devices/routers in that domain
            type :
                store type
            child_domains :
                a list of domain_ids of domains in the lower level that connects to this domain
            parent_domains : list
                a list of domain_ids of domains in the upper level that connects to this domain
            function : string
                "operating" if this domain contains at least one device; "routing" otherwise

        '''
    def __init__(self, id, type):
        self.id = id
        self.nodes = []
        self.type = type
        self.child_domains = []
        self.parent_domains = []
        if self.type == 'lan':
            self.function = 'operating'
            self.child_domains = None
        else:
            self.function = 'routing'

        if self.type == 'transit':
            self.parent_domains = None

    def add_node(self, node, node_type):
        """ Add one node to the domain providing the node object and the type of the node"""
        self.nodes.append(node)
        if node_type == 'edge':
            self.function = 'operating'