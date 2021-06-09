class Operator:
    def __init__(self, id, computation=4000, memory_requirement=10):
        self.id = id
        self.root = True
        self.out_streams = []
        self.in_streams = []
        self.device = None
        self.delay = None   # in ms
        self.computation = computation
        self.memory_requirement = memory_requirement

    def map(self, device):
        if device.load(self):
            self.device = device
            self.delay = self.computation / device.compute_rate / 1000 # in ms
            return True
        else:
            print("Operator "+str(self.id) +" cannot be mapped to device " + str(device.id))
            return False
