from placelib.simulation import Simulation

class Experiment:
    '''
        An instance of the experiment using the same

        Parameters
        ----------
        args : Namespace
            The populated namespace from argument input
        *args
            The variable arguments are used for ...
        **kwargs
            The keyword arguments are used for ...

        Attributes
        ----------
        arg : str
            This is where we store arg,
    '''
    def __init__(self, args):
        self.args = args
        self.num_experiments = args.num_experiments
        self.seeds = args.seeds
        self.simulations = {seed:Simulation(args, seed) for seed in self.seeds}

    def __call__(self):
        mappings = {}
        runtimes = {}
        delays = {}
        for seed in self.seeds:
            mappings[seed] = []
            runtimes[seed] = []
            delays[seed] = []
            for _ in range(self.num_experiments):
                mapping, runtime = self.simulations[seed].map()
                delay = self.simulations[seed].evaluate(mapping)
                print(delay)
                mappings[seed].append(mapping)
                runtimes[seed].append(runtime)
                delays[seed].append(delay)
            return mappings, runtimes, delays
