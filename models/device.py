from models.operator import Operator

class Device:
    def __init__(self, id, compute_rate, memory=100000):
        self.id = id
        self.compute_rate = compute_rate
        self.memory = memory
        self.memory_usage = 0
        self.compute_usage = 0
        self.operators = []

    def load(self, operator: Operator):
        self.operators.append(operator)
        self.memory_usage += operator.memory_requirement
        self.compute_usage += operator.computation / self.compute_rate

        if self.compute_usage > 1:
            print("Computation overload on device " + str(self.id))
            return False
        if self.memory_usage > self.memory:
            print("Memory overload on device " + str(self.id))
            return False
        return True

    def offload(self, operator: Operator):
        if operator in self.operators:
            self.operators.remove(operator)
            self.memory_usage -= operator.memory_requirement
            self.compute_usage -= operator.computation / self.compute_rate
