import random
import time

import numpy as np
from typing import List, Tuple
from src.output import Output
from IPython.display import clear_output
class Velocity:
    swaps: List[Tuple[int, int]] = []

    def __init__(self, swaps):
        self.swaps = swaps

    def __neg__(self):
        return Velocity(reversed(self.swaps))

    def __iadd__(self, other):
        self.swaps.extend(other.swaps)
        return self

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __imul__(self, cf: float):
        if cf == 0:
            self.swaps = []
            return self
        if 0 <= cf:
            swaps = self.swaps
            k = int(cf * len(swaps))
            self.swaps = [swaps[i % len(swaps)] for i in range(k)]
            return self
        self.__imul__(-1)
        self.__imul__(-cf)
        return self

    def __mul__(self, cf: float):
        tmp = Velocity(self.swaps)
        tmp *= cf
        return tmp

class Position:
    route: List[int]

    def __init__(self, route):
        self.route = route.copy()

    def __iadd__(self, other: Velocity):
        for swap in other.swaps:
            self.route[swap[0]], self.route[swap[1]] = self.route[swap[1]], self.route[swap[0]]
        return self

    def __sub__(self, other) -> Velocity:
        idx = [0 for _ in range(len(self.route))]

        for i in range(len(self.route)):
            idx[self.route[i]] = i

        current = self.route.copy()
        result = Velocity([])
        for i in range(len(self.route)):
            j = idx[other.route[i]]
            if i != j:
                result.swaps.append((i, j))
                current[i], current[j] = current[j], current[i]
                idx[current[i]] = i
                idx[current[j]] = j
        return result

class Particle:
    velocity: Velocity
    position: Position
    cost: float

    def __init__(self, route, graph, self_trust, past_trust, global_trust):
        self.velocity = Velocity([])
        self.position = Position(route)
        self.graph = graph
        self.calc_cost()
        self.pbest = self.position
        self.pbest_cost = self.cost
        self.self_trust = self_trust
        self.past_trust = past_trust
        self.global_trust = global_trust

    def calc_cost(self):
        self.cost = self.graph.path_cost(self.position.route + self.position.route[:1])
        return self.cost

    def move(self):
        self.position += self.velocity
        self.calc_cost()
        if self.cost < self.pbest_cost:
            self.pbest = Position(self.position.route)
            self.pbest_cost = self.cost

    def update_velocity(self, gbest: Position):
        new_velocity = Velocity([])
        new_velocity += self.velocity * self.self_trust
        new_velocity += (self.pbest - self.position) * self.past_trust
        new_velocity += (gbest - self.position) * self.global_trust
        self.velocity = new_velocity

class Swarm:
    def __init__(self, graph, particle_count: int, self_trust: float, past_trust: float = random.uniform(0, 2), global_trust: float = random.uniform(0, 2)):
        self.graph = graph
        self.particle_count = particle_count
        self.self_trust = self_trust
        self.past_trust = past_trust
        self.global_trust = global_trust
        self.best_position = None
        self.best_cost = float('inf')
        self.particles = []
        self.initialize_particles()

    def greedy_initialization(self) -> List[List[int]]:
        distances = self.graph.distance_matrix
        population = []
        start_index = 0

        for _ in range(self.particle_count):
            if start_index < self.graph.n:
                current = start_index
            else:
                current = np.random.randint(0, self.graph.n)
            start_index += 1

            route = [current]
            unvisited = set(range(self.graph.n))
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
        for route in routes:
            particle = Particle(route, self.graph, self.self_trust, self.past_trust, self.global_trust)
            self.particles.append(particle)
            if particle.pbest_cost < self.best_cost:
                self.best_cost = particle.pbest_cost
                self.best_position = Position(particle.pbest.route)

    def random_route(self):
        route = list(range(self.graph.n))
        random.shuffle(route)
        return route

    def run(self, iterations=5000) -> Output:
        history = []
        times = []
        start = time.time()
        for i in range(iterations):
            st = time.time()
            self.solve()
            # clear_output()
            # print(i, self.solve())
            times.append(time.time() - st)
            history.append(self.best_cost)
        end = time.time()
        return Output(self.best_position.route, self.best_cost, history, times, end - start)


    def solve(self):
        self.past_trust = self.global_trust = random.uniform(0, 2)
        moves_since_best_changed = 0
        while moves_since_best_changed <= 4:
            best_changed = False
            if moves_since_best_changed < 2:
                best_changed = self.normal_search()
            else:
                if moves_since_best_changed < 4:
                    best_changed = self.lazy_descent()
                else:
                    best_changed = self.energetic_descent()
            if not best_changed:
                moves_since_best_changed += 1
            else:
                moves_since_best_changed = 0
        return self.best_cost

    def normal_search(self):
        best_changed = False
        for particle in self.particles:
            particle.update_velocity(self.best_position)
            particle.move()
            if particle.cost < self.best_cost:
                self.best_cost = particle.cost
                self.best_position = Position(particle.position.route)
                best_changed = True
        return best_changed

    def lazy_descent(self):
        best_changed = False
        self.particles_back_to_best()
        for _ in range(self.graph.n):
            if self.move_all_slowly():
                best_changed = True
                break
        return best_changed

    def energetic_descent(self):
        any_changed = False
        self.particles_back_to_best()
        while True:
            current_changed = False
            for _ in range(self.graph.n):
                current_changed = self.move_all_slowly() or current_changed
            if not current_changed:
                break
            any_changed = True
        return any_changed

    def particles_back_to_best(self):
        for particle in self.particles:
            particle.position = Position(particle.pbest.route)

    def move_all_slowly(self):
        best_changed = False
        for particle in self.particles:
            a, b = random.sample(range(self.graph.n), 2)
            particle.velocity = Velocity([(a, b)])
            particle.move()
            if particle.cost < self.best_cost:
                self.best_cost = particle.cost
                self.best_position = Position(particle.position.route)
                best_changed = True
        return best_changed
