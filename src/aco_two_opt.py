import logging
import random
import time

import matplotlib.pyplot as plt
from src.graph import Graph
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from IPython.display import display, clear_output
from src.output import Output


class Ant:
    def __init__(self, idx, colony, start):
        self.id = idx
        self.colony = colony
        self.graph = colony.graph

        self.not_visited = list(filter(lambda x: x != start, range(colony.graph.n)))
        self.path = [start]
        self.distance = 0

        self.q0 = colony.parameters['q0']
        self.phi = colony.parameters['phi']
        self.current_city = start

    def choose_next(self) -> Optional[int]:
        pheromone = self.colony.pheromone[self.current_city, self.not_visited] ** self.colony.parameters['alpha']
        heuristic = self.colony.heuristic[self.current_city, self.not_visited] ** self.colony.parameters['beta']
        probabilities = pheromone * heuristic
        probabilities /= np.sum(probabilities)
        return random.choices(self.not_visited, weights=probabilities)[0]

    def run(self) -> None:
        for _ in range(self.graph.n - 1):
            next_city = self.choose_next()
            self.not_visited.remove(next_city)
            self.distance += self.graph.distance(self.current_city, next_city)
            self.path += [next_city]
            self.current_city = next_city

        self.distance += self.graph.distance(self.path[-1], self.path[0])
        self.colony.solution_update(self.path, self.distance)

    def local_search(self):
        improved = True
        best_path = self.path
        best_distance = self.distance
        while improved:
            improved = False
            for i in range(1, len(best_path) - 1):
                for k in range(i + 1, len(best_path) - 1):
                    new_route = best_path[:i] + list(reversed(best_path[i:k + 1])) + best_path[k + 1:]
                    new_route_distance = self.graph.path_cost(new_route)
                    if new_route_distance < best_distance:
                        best_distance = new_route_distance
                        best_path = new_route
                        improved = True
                        break
                if improved:
                    break
        self.path = best_path
        self.distance = best_distance


class ACSAnt(Ant):
    def __init__(self, idx, colony, start):
        super().__init__(idx, colony, start)

    def choose_next(self) -> Optional[int]:
        if random.random() < self.colony.parameters['q0']:
            probs = (self.colony.pheromone[self.current_city, self.not_visited] ** self.colony.parameters['alpha'] *
                     self.colony.heuristic[self.current_city, self.not_visited] ** self.colony.parameters['beta'])
            next_city = self.not_visited[np.argmax(probs)]
        else:
            next_city = super().choose_next()
        return next_city

    def local_update_pheromone(self):
        self.colony.pheromone[self.path, self.path[1:] + self.path[:1]] = (
            (1 - self.phi) * self.colony.pheromone[self.path, self.path[1:] + self.path[:1]] +
            self.phi * self.colony.init_pheromone
        )

    def run(self) -> None:
        for _ in range(self.graph.n - 1):
            next_city = self.choose_next()
            self.not_visited.remove(next_city)
            self.distance += self.graph.distance(self.current_city, next_city)
            self.path += [next_city]
            self.current_city = next_city

        self.distance += self.graph.distance(self.path[-1], self.path[0])
        self.local_update_pheromone()
        self.colony.solution_update(self.path, self.distance)


class Colony:
    def __init__(self, graph: Graph, **kwargs):
        self.ants = []
        self.paths = []
        self.distances = []
        self.graph = graph
        self.parameters = kwargs


        match self.parameters['type']:
            case 'as':
                self.init_pheromone = float(self.parameters['ant_count']) / self.graph.nn_cost
            case 'eas':
                self.init_pheromone = float(2 * self.parameters['ant_count']) / (self.parameters['rho'] * self.graph.nn_cost)
            case 'as_rank':
                self.init_pheromone = float(2 * self.parameters['top'] * (self.parameters['top'] - 1)) / (self.parameters['rho'] * self.graph.nn_cost)
            case 'mmas':
                self.init_pheromone = float(1) / (self.parameters['rho'] * self.graph.nn_cost)
            case 'acs':
                self.init_pheromone = float(1) / (self.graph.n * self.graph.nn_cost)
            case _:
                raise Exception('Unrecognized parameter "type"')

        self.pheromone = np.full((self.graph.n, self.graph.n), self.init_pheromone)
        self.heuristic = np.zeros_like(self.pheromone)
        self.heuristic = np.divide(
            float(1),
            self.graph.distance_matrix,
            out=self.heuristic,
            where=self.graph.distance_matrix != 0
        )
        self.best_path = None
        self.best_distance = np.inf
        self.iter_best_path = None
        self.iter_best_distance = np.inf

    def update_best(self):
        if self.iter_best_distance < self.best_distance:
            self.best_distance = self.iter_best_distance
            self.best_path = self.iter_best_path

    def update_pheromone(self):
        self.update_best()
        self.pheromone *= (1 - self.parameters['rho'])

        if 'max_pheromone' in self.parameters:
            mx = self.parameters['max_pheromone']
        else:
            mx = 1. / (self.parameters['rho'] * self.best_distance)
        tmp = 0.05 ** (1. / self.graph.n)
        if 'min_pheromone' in self.parameters:
            mn = self.parameters['min_pheromone']
        else:
            mn = mx * (1 - tmp) / ((self.graph.n / 2 - 1) * tmp)

        match self.parameters['type']:
            case 'as':
                for path, distance in zip(self.paths, self.distances):
                    pheromone_value = 1 / distance
                    edges = list(zip(path, path[1:] + path[:1]))
                    rows, cols = zip(*edges)
                    self.pheromone[rows, cols] += pheromone_value
            case 'eas':
                for path, distance in zip(self.paths, self.distances):
                    pheromone_value = 1 / distance
                    edges = list(zip(path, path[1:] + path[:1]))
                    rows, cols = zip(*edges)
                    self.pheromone[rows, cols] += pheromone_value
                elitist_value = 1 / self.best_distance
                edges = list(zip(self.best_path, self.best_path[1:] + self.best_path[:1]))
                rows, cols = zip(*edges)
                self.pheromone[rows, cols] += elitist_value
            case 'as_rank':
                rank = np.argsort(self.distances)
                for j in range(0, self.parameters['top']):
                    i = rank[j]
                    path = self.paths[i]
                    distance = self.distances[i]
                    pheromone_value = (self.parameters['top'] - j) / distance
                    edges = list(zip(path, path[1:] + path[:1]))
                    rows, cols = zip(*edges)
                    self.pheromone[rows, cols] += pheromone_value
            case 'mmas':
                elitist_value = 1 / self.best_distance
                edges = list(zip(self.best_path, self.best_path[1:] + self.best_path[:1]))
                rows, cols = zip(*edges)
                self.pheromone[rows, cols] += elitist_value
                np.clip(self.pheromone, mn, mx, out=self.pheromone)
            case 'acs':
                self.pheromone[self.best_path, self.best_path[1:] + self.best_path[:1]] = (
                    (1 - self.parameters['rho']) * self.pheromone[self.best_path, self.best_path[1:] + self.best_path[:1]] +
                    self.parameters['rho'] / self.best_distance
                )
                np.clip(self.pheromone, mn, mx, out=self.pheromone)

    def create_ant(self, idx):
        if self.parameters['type'] == 'acs':
            return ACSAnt(idx, self, random.choice(range(self.graph.n)))
        return Ant(idx, self, random.choice(range(self.graph.n)))

    def generate_solutions(self) -> None:
        self.paths = []
        self.distances = []
        self.ants = []
        self.iter_best_path = None
        self.iter_best_distance = np.inf

        ibest = 0
        for i in range(self.parameters['ant_count']):
            self.ants.append(self.create_ant(i))
            self.ants[-1].run()
            if self.ants[i].distance < self.ants[ibest].distance:
                ibest = i
                self.iter_best_distance = self.ants[i].distance
                self.iter_best_path = self.ants[i].path

        self.ants[ibest].local_search()
        if self.parameters['type'] == 'acs':
            self.ants[ibest].local_update_pheromone()
            self.solution_update(self.ants[ibest].path, self.ants[ibest].distance)

    def solution_update(self, path, distance):
        self.paths.append(path)
        self.distances.append(distance)

        if distance < self.iter_best_distance:
            self.iter_best_path = path
            self.iter_best_distance = distance

    def run(self) -> Output:
        start = time.time()
        history = []
        times = []
        for _ in range(self.parameters['generations']):
            st = time.time()
            self.generate_solutions()
            self.update_pheromone()
            history.append(self.best_distance)
            times.append(time.time() - st)
            # print(_, self.best_distance)

        end = time.time()
        return Output(self.best_path, self.best_distance, history, times, end - start)


