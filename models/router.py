import numpy as np

class Router:
    '''
        The router model

        Parameters
        ----------
        id :
            unique identifier for the unique router instance created
        proc_rate :
            the processing rate of the router in the number of packets it can handle per second
        traffic (optional) : int
            the amount of traffic ( in packets per second) already been loaded on the router. Defualt 10

        Attributes
        ----------
        id :
            store id
        proc_rate :
            store proc_rate
        traffic :
            the current traffic (updated over time)
        expected_delay :
            the expected delay given the traffic and the processing rate (updated over time)
    '''
    def __init__(self, id, proc_rate, traffic=10):
        self.id = id
        self.proc_rate = proc_rate
        self.traffic = traffic #packet per second
        assert self.traffic < self.proc_rate, "Traffic overload on router " + str(self.id)
        self.expected_delay = 1000 / ((self.proc_rate - traffic)) # in ms

    def load_traffic(self, extra=1):
        """load extra=1 amount of traffic to the router. Return True if successful, otherwise (overloaded) False """
        if self.traffic + extra < self.proc_rate:
            self.traffic += extra
            self.expected_delay = 1000 / (self.proc_rate - self.traffic)
            return True
        else:
            return False


    def offload_traffic(self, remove=1):
        """ remove remove=1 amount of traffic from the router. """
        self.traffic = max(self.traffic-remove, 0)
        self.expected_delay = 1000 / ((self.proc_rate - self.traffic))

    def delay(self, average=True):
        """ return delay in ms at the router.
            If average is set True, the expectation of the delay is returned.
            Otherwise, exponential processing time is assumed
        """
        if average:
            return self.expected_delay
        return np.random.exponential(self.expected_delay)

