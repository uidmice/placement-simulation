

class Device:
    '''
        The device model

        Parameters
        ----------
        id :
            unique identifier for the unique device object created
        compute_rate :
            the computation rate of the device in million instructions per second (MIPS)
        memory (optional) :
            the amount of available memory on the device

        Attributes
        ----------
        id :
            store id
        compute_rate :
            store proc_rate
        memory_capacity :
            store the total amount of memory on the device
        available_memory
            the current amount of available memory (updated over time)
        compute_usage :
            the current percentage of time used for computation (must be smaller than 1)
        op_compute_usage : dict {Operator : percentage_of_time}
            a dictionary that stores the percentage of computation time for each operator
        operators : list
            a list of operators that have been loaded on the device
        tentative_compute_usage :
            the amount of compute to be put on the device determined by the mapping algorithm (without actually mapping the operators)
        tentative_memory_usage :
            the amount of memory to be used determined by the mapper (without actually mapping the operators)
        tentative_op : list
            a list of operators to be placed on the device (used by the mapper)
    '''

    def __init__(self, id, compute_rate, memory=100000):
        self.id = id
        self.compute_rate = compute_rate
        self.memory_capacity = memory
        self.available_memory = memory
        self.compute_usage = 0
        self.op_compute_usage = {}
        self.operators = []
        self.tentative_compute_usage = 0
        self.tentative_memory_usage = 0
        self.tentative_op = []

    def load(self, operator):
        """ Load operator onto the device. Return False is there is no sufficient compute or memory on the device """

        if (self.compute_usage + operator.computation / self.compute_rate / 1000000 > 1):
            print(f"Computation overload on device {self.id}")
            return False
        if self.available_memory < operator.memory_requirement:
            print(f"No sufficient memory on device {self.id}")
            return False
        self.operators.append(operator)
        self.op_compute_usage[operator] = operator.computation / self.compute_rate / 1000000
        self.compute_usage += self.op_compute_usage[operator]
        self.available_memory -= operator.memory_requirement
        return True

    def offload(self, operator):
        """Offload an operator from the device"""
        if operator in self.operators:
            self.operators.remove(operator)
            self.available_memory += operator.memory_requirement
            self.compute_usage = max(0, self.compute_usage - self.op_compute_usage[operator.id])

    def tentative_load(self, operator):
        """ Tentatively load an operator when running the algorithm. Return False is there is no sufficient compute or memory on the device """
        if (self.compute_usage + self.tentative_compute_usage + operator.computation / self.compute_rate / 1000000 > 1):
            print(f"Computation overload on device {self.id}")
            return False
        if self.available_memory - self.tentative_memory_usage < operator.memory_requirement:
            print(f"No sufficient memory on device {self.id}")
            return False
        self.tentative_op.append(operator)
        self.op_compute_usage[operator] = operator.computation / self.compute_rate / 1000000
        self.tentative_compute_usage += self.op_compute_usage[operator]
        self.tentative_memory_usage += operator.memory_requirement
        return True

    def delay(self, operator):
        """ Return the expected delay in ms of the operator """
        if operator in self.op_compute_usage:
            return self.op_compute_usage[operator]
        return operator.computation / self.compute_rate / 1000

    def reset(self, include_mapped=False):
        """ Clear all tentative mapping. If include_mapped is True, everything inclusing mapeped is reset"""
        self.tentative_memory_usage = 0
        self.tentative_compute_usage = 0
        self.tentative_op = []
        if include_mapped:
            self.available_memory = self.memory_capacity
            self.compute_usage = 0
            self.operators = []
