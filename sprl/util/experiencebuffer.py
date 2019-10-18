import numpy as np


class ExperienceBuffer:

    def __init__(self, size, n_elements=1):
        self.size = size
        self.elements = tuple([[] for __ in range(0, n_elements)])

    def insert(self, *args):
        for i in range(0, len(args)):
            self.elements[i].append(args[i])
            if len(self.elements[i]) > self.size:
                self.elements[i].pop(0)

    def get(self):
        return tuple([np.concatenate(self.elements[i], axis=0) for i in range(0, len(self.elements))])
