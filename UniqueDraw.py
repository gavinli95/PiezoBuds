import itertools
import random


class UniqueDraw:
    def __init__(self):
        self.history = set()

    def draw_numbers(self, n, m):
        """
        Draw m numbers from n integers (0 to n-1), ensuring that the same combination is not returned twice.

        :param n: The range of possible numbers to draw from.
        :param m: The number of numbers to draw.
        :return: A list of m numbers drawn from the range 0 to n-1, or None if all combinations have been drawn.
        """
        # All combinations have been drawn
        if len(self.history) == len(list(itertools.combinations(range(n), m))):
            return None

        while True:
            draw = tuple(sorted(random.sample(range(n), m)))
            if draw not in self.history:
                self.history.add(draw)
                return list(draw)
