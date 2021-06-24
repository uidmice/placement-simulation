from placelib.simulation import Simulation

class Experiment:
    '''
        An instance of the experiment using the same set of parameters

        Parameters
        ----------
        args : Namespace
            The populated namespace from argument input

        Attributes
        ----------
        args : Namespace
            store args
        num_experiments : int
            number of experiments to run with each seed provided, default 1
        seeds : list
            a list of seeds to use (default [0])
        simulations : dict {seed : Simulation}
            a dictionary of simulation object, seed as the key
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
            # print(self.num_experiments)
            for _ in range(self.num_experiments):
                mapping, runtime = self.simulations[seed].map() # 
                delay = self.simulations[seed].evaluate(mapping)
                mappings[seed].append(mapping)
                runtimes[seed].append(runtime)
                delays[seed].append(delay)
        return mappings, runtimes, delays
