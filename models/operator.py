from models.device import  Device
class Operator:
    '''
        The operator model

        Parameters
        ----------
        id :
            unique identifier for the unique operator object created
        computation :
            the amount of computation (in number of instructions) required for running this operator
        memory_requirement (optional) :
            the amount of available memory on the device

        Attributes
        ----------
        id :
            store id
        computation :
            store computation
        memory_requirement :
            store memory requirement
        device :
            where it is mapped. Initially None when it has not been mapped
        out_streams : list
            a list of Stream objects for outgoing streams
        in_streams : list
            a list of stream objects for data streams that are received by this operator

    '''

    def __init__(self, id, computation, memory_requirement=10):
        self.id = id
        self.out_streams = []
        self.in_streams = []
        self.device = None
        self.computation = computation
        self.memory_requirement = memory_requirement

    def map(self, device: Device):
        """map this operator to a specific device; return True if successful, False otherwise"""
        if device.load(self):
            self.device = device
            return True
        else:
            print(f"Operator {self.id} cannot be mapped to device {device.id}")
            return False

    def estimate_compute_time(self, device):
        """ return the estimate computation time of running this operator on the device"""
        return device.delay(self)

