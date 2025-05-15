from typing import List


class Output:
    best_path: List[int]
    best_dist: float
    history: List[float]
    times: List[float]
    overall_time: float

    def __init__(self,
                 best_path: List[int],
                 best_dist: float,
                 history: List[float],
                 times: List[float],
                 overall_time: float):
        self.best_path = best_path
        self.best_dist = best_dist
        self.history = history
        self.times = times
        self.overall_time = overall_time

