import numpy as np
import matplotlib.pyplot as plt
import random
import heapq

# makes outputs easier to read
ROUND = True


class Grid:
    def __init__(self, height=15, width=15,
                 distribution="uniform", distr_params=[0,10], seed=-1):
        self.height = height
        self.width = width
        # by default it doesn't set a seed
        if seed > 0:
            np.random.seed(seed)  # fix it so can compare more easily for tests
            random.seed(seed)
        self.distribution = distribution
        if distribution == "uniform":
            distribution = "randint"
        # this allows the user to pass an np function such as: randint, poisson,
        # gamma, normal, negative_binomial, etc with a list of parameters
        np_distr_function = getattr(np.random, distribution)
        self.cost_matrix = np_distr_function(*distr_params, size=(height, width))
        if ROUND:
            self.cost_matrix = np.round(self.cost_matrix,0)

    def plot_grid(self, show_values=False):
        # fig and ax simply show the grid untraversed.
        # https://stackoverflow.com/questions/40887753/display-matrix-values-and-colormap
        fig, ax = plt.subplots()
        ax.matshow(self.cost_matrix, cmap=plt.cm.Pastel1)
        if show_values:
            for i in range(self.height):
                for j in range(self.width):
                    ax.text(i, j, str(self.cost_matrix[j, i]), va='center', ha='center')
        fig.show()

    def __str__(self):
        return_str = f"Grid, size ({self.height},{self.width})\n"
        return_str += f"generated using a {self.distribution} distribution\n"
        return_str += str(self.cost_matrix)
        return return_str


class Agent:
    def __init__(self, location=(0,0),
                 legal_moves=[[0, 1], [1, 0], [-1, 0], [0, -1]]):
        self.legal_moves = legal_moves
        self.location = location


class Game:
    def __init__(self, game_mode=1, target_location=-1, init_location=(0,0),
                 grid=Grid(height=15, width=15, distribution="uniform",
                 distr_params=[0, 10]), seed=-1):
        self.game_mode = game_mode
        self.agent = Agent(location=init_location)
        self.grid = grid
        if target_location == -1:
            self.target_location = (self.grid.height-1, self.grid.width-1)
        else:
            self.target_location = target_location

    def __str__(self):
        return_str = "GAME\n"
        return_str += f"Game mode {self.game_mode}\n"
        return_str += f"Initial agent location is {self.agent.location}\n"
        return_str += f"Target agent location is {self.target_location}\n"
        return_str += f"Using:\n{self.grid}\n"
        return return_str




g = Grid(distribution="randint",distr_params=[0,10])
g = Grid(distribution="normal", distr_params=[150,50])
#print(g)
#g.plot_grid()
g = Game()
print(g)
