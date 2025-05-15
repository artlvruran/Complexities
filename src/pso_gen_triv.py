import time

import numpy as np
from typing import List, Tuple

from IPython.core.display_functions import clear_output
from src.graph import Graph
from src.output import Output

class Position:
    route: List[int]

    def __init__(self, route):
        self.route = route.copy()


class Particle:
    position: Position
    cost: float

    def __init__(self, route, graph):
        self.position = Position(route)
        self.graph = graph
        self.calc_cost()
        self.pbest = self.position
        self.pbest_cost = self.cost

    def calc_cost(self):
        self.cost = self.graph.path_cost(self.position.route + self.position.route[:1])
        return self.cost

    def update_best(self):
        if self.cost < self.pbest_cost:
            self.pbest = Position(self.position.route)
            self.pbest_cost = self.cost


class Swarm:
    def __init__(self, graph: Graph, particle_count: int, r=10, alpha=0.5, c1=0.2, c2=0.2):
        self.graph = graph
        self.n = self.graph.n
        self.particle_count = particle_count
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.r = r

        self.pcitylink, self.dcitylink = self.compute_city_links()
        self.pcitydistlink, self.pcitydistlinkcum = self.compute_probability_matrices()

        self.best_position = None
        self.best_cost = float('inf')
        self.particles: List[Particle] = []
        self.initialize_particles()

    def compute_city_links(self):
        distances = self.graph.distance_matrix
        pcitylink = np.zeros((self.n, self.r), dtype=int)
        dcitylink = np.zeros(self.n, dtype=int)

        for i in range(self.n):
            sorted_cities = np.argsort(distances[i])
            sorted_cities = sorted_cities[sorted_cities != i]

            pcitylink[i] = sorted_cities[:self.r]
            dcitylink[i] = sorted_cities[0]

        return pcitylink, dcitylink

    def compute_probability_matrices(self):
        distances = self.graph.distance_matrix
        pcitydistlink = np.zeros((self.n, self.r))
        pcitydistlinkcum = np.zeros((self.n, self.r))

        for i in range(self.n):
            nearest_cities = self.pcitylink[i]
            dists = distances[i, nearest_cities]

            inv_dists = 1 / np.where(dists > 0, dists, 0)
            total = inv_dists.sum()

            if total == 0:
                probs = np.ones(self.r) / self.r
            else:
                probs = inv_dists / total

            probs /= probs.sum()

            pcitydistlink[i] = probs
            pcitydistlinkcum[i] = np.cumsum(probs)

        return pcitydistlink, pcitydistlinkcum

    def probabilistic_crossover(self, aa: Position, pbest: Position):
        current_route = aa.route.copy()
        pbest_route = pbest.route

        c1, c2 = sorted(np.random.choice(len(pbest_route), 2, replace=False))
        cros = pbest_route[c1:c2 + 1]

        new_route = [x for x in current_route if x not in cros]

        if not cros:
            return Position(new_route)

        cc = cros[0]
        cum_probs = self.pcitydistlinkcum[cc]
        rand = np.random.rand()

        selected = np.where(cum_probs >= rand)[0]
        dd = self.pcitylink[cc, selected[0]]

        if dd not in cros:
            insert_idx = new_route.index(dd) + 1
            final_route = new_route[:insert_idx] + cros + new_route[insert_idx:]
        else:
            final_route = new_route + cros

        return Position(final_route)

    def deterministic_crossover(self, aa: Position, gbest: Position):
        current_route = aa.route.copy()
        gbest_route = gbest.route

        c3, c4 = sorted(np.random.choice(len(gbest_route), 2, replace=False))
        cros = gbest_route[c3:c4 + 1]

        new_route = [x for x in current_route if x not in cros]

        cc = cros[0]
        dd = self.dcitylink[cc]

        if dd not in cros:
            insert_idx = new_route.index(dd) + 1
            final_route = new_route[:insert_idx] + cros + new_route[insert_idx:]
        else:
            final_route = new_route + cros

        return Position(final_route)

    def directional_mutation(self, aa: Position):
        route = aa.route.copy()
        if len(route) < 2:
            return Position(route)

        b1 = np.random.randint(0, len(route) - 1)
        xx = route[b1]
        temp = route[b1 + 1]

        ck = self.dcitylink[xx]

        ck_idx = route.index(ck)
        route[b1 + 1] = ck
        route[ck_idx] = temp

        return Position(route)

    def greedy_initialization(self) -> List[List[int]]:
        distances = self.graph.distance_matrix
        population = []
        start_index = 0

        for _ in range(self.particle_count):
            if start_index < self.n:
                current = start_index
            else:
                current = np.random.randint(0, self.n)
            start_index += 1

            route = [current]
            unvisited = set(range(self.n))
            unvisited.remove(current)

            while unvisited:
                last = route[-1]
                next_city = min(unvisited, key=lambda city: distances[last, city])
                route.append(next_city)
                unvisited.remove(next_city)

            population.append(route)

        return population

    def initialize_particles(self):
        routes = self.greedy_initialization()
        #routes = [self.random_route() for _ in range(self.particle_count)]
        for route in routes:
            particle = Particle(route, self.graph)
            self.particles.append(particle)
            if particle.pbest_cost < self.best_cost:
                self.best_cost = particle.pbest_cost
                self.best_position = Position(particle.pbest.route)

    def random_route(self):
        route = list(range(self.graph.n))
        np.random.shuffle(route)
        return route

    def cross(self, first: Position, second: Position):
        first_route = first.route.copy()
        rnd = np.random.choice(range(self.graph.n), 2)
        x, y = min(rnd), max(rnd)
        cross_part = second.route[x:y]

        tmp = []
        for el in first_route:
            if el in cross_part:
                continue
            tmp += [el]
        son = Position(tmp + cross_part)
        daughter = Position(cross_part + tmp)
        son_cost = self.calc_cost(son)
        daughter_cost = self.calc_cost(daughter)
        if son_cost < daughter_cost:
            return son
        else:
            return daughter

    def mutate(self, position: Position):
        route = position.route.copy()

        rnd = np.random.choice(range(self.graph.n), 2)
        x, y = min(rnd), max(rnd)
        route[x], route[y] = route[y], route[x]
        pos = Position(route)
        return pos


    def calc_cost(self, position: Position):
        return self.graph.path_cost(position.route + position.route[:1])

    def two_opt(self, position: Position) -> Position:
        route = position.route.copy()
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if self.graph.path_cost(new_route + [new_route[0]]) < self.graph.path_cost(best + [best[0]]):
                        best = new_route
                        improved = True
            route = best
        return Position(best)

    def solve(self, iterations=500) -> Output:
        history = []
        times = []
        start = time.time()
        for _ in range(iterations):
            st = time.time()
            for i, particle in enumerate(self.particles):
                cur_position = particle.position
                cur_cost = particle.cost

                new_position = self.cross(cur_position, particle.pbest)
                new_cost = self.calc_cost(new_position)
                if new_cost < cur_cost or np.random.rand() < self.c1:
                    cur_position = new_position
                    cur_cost = new_cost

                new_position = self.cross(cur_position, self.best_position)
                new_cost = self.calc_cost(new_position)
                if new_cost < cur_cost or np.random.rand() < self.c2:
                    cur_position = new_position
                    cur_cost = new_cost

                new_position = self.mutate(cur_position)
                new_cost = self.calc_cost(new_position)
                if new_cost < cur_cost or np.random.rand() < self.alpha:
                    cur_position = new_position
                    cur_cost = new_cost

                self.particles[i].position = cur_position
                self.particles[i].cost = cur_cost
                self.particles[i].update_best()

            for i in range(self.particle_count):
                if self.particles[i].pbest_cost < self.best_cost:
                    self.best_cost = self.particles[i].pbest_cost
                    self.best_position = Position(self.particles[i].pbest.route)

            history.append(self.best_cost)
            times.append(time.time() - st)

            clear_output()
            #print(_, self.best_cost)
        return Output(self.best_position.route, self.best_cost, history, times, time.time() - start)

