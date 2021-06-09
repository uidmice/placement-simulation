class Link:
    def __init__(self, from_node, to_node, bw, delay):
        self.from_node = from_node
        self.to_node = to_node
        self.bw = bw # mbps
        self.delay = delay/1000 # ms

    def link_delay(self, kbytes): # ms
        return self.delay + kbytes/self.bw
