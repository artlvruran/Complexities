import random

import numpy as np


class Particle:
    def __init__(self, route, graph, cost=None):
        self.velocity = []

        self.route = route
        self.cost = cost

        self.pbest = route
        self.pbest_cost = cost

        self.graph = graph

    def update_cost(self):
        self.cost = self.eval_cost()
        if self.cost < self.pbest_cost:
            self.pbest_cost = self.cost
            self.pbest = self.route

    def eval_cost(self):
        return self.graph.path_cost(self.route)

    def update_route(self):
        for swap in self.velocity:
            
        


class PSO:
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.parameters = kwargs
        self.gbest = None

        self.particles = [Particle(route, graph) for route in self.initial_population()]

    def random_route(self):
        return np.random.permutation(range(self.graph.n))

    def initial_population(self):
        return [self.random_route() for _ in range(self.parameters['population_size'])]

    def run(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        for i in range(self.parameters['iterations']):
            for particle in self.particles:
