class Router:
    def __init__(self, id, proc_rate, traffic=10):
        self.id = id
        self.proc_rate = proc_rate
        self.traffic = traffic #packet per second
        assert self.traffic < self.proc_rate, "Traffic overload on router " + str(self.id)
        self.delay = 1000/((self.proc_rate - traffic)) # in ms

    def load_traffic(self, extra=1):
        self.traffic += extra
        assert self.traffic < self.proc_rate, "Traffic overload on router " + str(self.id)
        self.delay = 1000 / (self.proc_rate - self.traffic)

    def offload_traffic(self, remove=1):
        self.traffic = max(self.traffic-remove, 0)
        self.delay = 1000 / ((self.proc_rate - self.traffic))