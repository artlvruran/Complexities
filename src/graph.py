import random
import threading
from typing import List

from IPython.core.display_functions import clear_output
from sklearn.neighbors import KDTree

import numpy
import numpy as np

from tqdm.auto import tqdm


class Graph:
    def __init__(
            self,
            distance_matrix: numpy.ndarray,
            points,
            eval_nn: bool = False
    ):
        self.distance_matrix = distance_matrix
        self.points = points
        self.n = distance_matrix.shape[0]

        if eval_nn:
            self.nn_cost = self.get_nn_cost()
        else:
            self.nn_cost = self.get_random_cost()

        self.lock = threading.Lock()

    def distance(self, i, j):
        return self.distance_matrix[i][j]

    def get_nn_cost(self):
        nn_cost = float("inf")
        for start in range(0, self.n):
            clear_output()
            not_visited = list(range(self.n))
            not_visited.remove(start)

            current = start
            current_cost = 0
            for i in range(self.n - 1):
                nxt = int(np.argmin(self.distance_matrix[current, not_visited]))
                next_city = not_visited[nxt]

                not_visited.remove(next_city)
                current_cost += self.distance_matrix[current, next_city]
                current = next_city
            current_cost += self.distance_matrix[current, start]

            if current_cost < nn_cost:
                nn_cost = current_cost

        return nn_cost

    def get_random_cost(self):

        nn_cost = float("inf")
        start = random.randint(0, self.n - 1)
        not_visited = list(range(self.n))
        not_visited.remove(start)

        current = start
        current_cost = 0
        for i in range(self.n - 1):
            nxt = int(np.argmin(self.distance_matrix[current, not_visited]))
            next_city = not_visited[nxt]

            not_visited.remove(next_city)
            current_cost += self.distance_matrix[current, next_city]
            current = next_city
        current_cost += self.distance_matrix[current, start]

        if current_cost < nn_cost:
            nn_cost = current_cost

        return nn_cost

    def path_cost(self, route):
        return np.sum(self.distance_matrix[route, route[1:] + route[:1]])
