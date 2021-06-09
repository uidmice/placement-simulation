class Domain:
    def __init__(self, id, type):
        self.id = id
        self.nodes = []
        self.type = type
        self.leaf_domains = []
        self.parent_domains = []
        if self.type == 'lan':
            self.function = 'operating'
            self.leaf_domains = None
        else:
            self.function = 'routing'

        if self.type == 'transit':
            self.parent_domains = None

    def add_node(self, node, node_type):
        self.nodes.append(node)
        if node_type == 'edge':
            self.function = 'operating'