import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
import copy
import statistics

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
        self.distr_params = distr_params
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
        return_str += f"generated using a {self.distribution} distribution {self.distr_params}\n"
        return_str += str(self.cost_matrix)
        return return_str

class Agent:
    def __init__(self, location=(0,0),
                 legal_moves=[[0, 1], [1, 0], [-1, 0], [0, -1]]):
        self.legal_moves = legal_moves
        self.prev_location = location
        self.location = location


class Game:
    def __init__(self, game_mode=1, target_location=-1, init_location=(0,0),
                 grid=Grid(height=15, width=15, distribution="uniform",
                 distr_params=[0, 10]), seed=-1):
        self.game_mode = game_mode
        self.agent = Agent(location=init_location)
        self.grid = grid
        # this will be the actual path marked up grid
        self.grid_path = copy.deepcopy(self.grid)
        # These two will eventually become the shortest path and the lowest cost
        self.shortest_path = []
        self.lowest_cost = 1000000
        # if no target location given, assume far bottom right of grid.
        if target_location == -1:
            self.target_location = (self.grid.height-1, self.grid.width-1)
        else:
            self.target_location = target_location

    # this is used both by heuristic and dijkstra
    # it stops agent from moving off grid as well
    def get_possible_moves(self):
        # now strip out those that take it off the grid
        possible_moves = []
        for move in self.agent.legal_moves:
            new_agent_location = (self.agent.location[0] + move[0],
                                  self.agent.location[1] + move[1])
            # will this move keep it on the grid? Keep it if it does.
            if new_agent_location[0] >= 0 and new_agent_location[1] >= 0 \
                    and new_agent_location[0] < self.grid.height and \
                        new_agent_location[1] < self.grid.width:
                possible_moves.append((self.agent.location[0] + move[0],
                                       self.agent.location[1] + move[1]))
        # all possible moves returned
        return possible_moves

    # find_best_move
    # This is based on rewarding moves the go right and down
    # The value is the total reward
    # Value is "reward" for moving left or down but not both,
    # and "reward*2" for moving both.
    # other moves have no reward.
    # Note - that the value is initialised as -1 * the cost moving
    # to the location.
    # So a value of 10 moving to a square with time 7 will
    # lead ot the move;s value being 10-7 = 3
    def get_heuristic_move_values(self, moves):
        reward = 5
        values = []
        for move in moves:
            value = -1 * self.grid.cost_matrix[move[0]][move[1]]
            if self.game_mode == 2:
                value = abs(value + self.grid.cost_matrix[
                    self.agent.location[0]][self.agent.location[1]])
            if move[0] > self.agent.location[0]:  # down
                value += reward
            if move[1] > self.agent.location[1]:
                value += reward  # right
            # print("Matrix value: ",cost_matrix[move[0]][move[1]])
            # print("Value: ", value)
            values.append(value)
        return values

    # For each move to x,y the agent calculates C = TIME(x,y)-sum_of_move_costs
    # The agent then picks the x,y move that minimises C
    # if two versions of C are equal, the agent randomly picks one
    def find_best_heuristic_move(self):
        gpm = self.get_possible_moves()
        values = self.get_heuristic_move_values(gpm)
        max_move_index = values.index(max(values))
        return gpm[max_move_index], max(values)

    def random_move(self):
        gpm = self.get_possible_moves()
        # https://stackoverflow.com/questions/306400/how-can-i-randomly-select-an-item-from-a-list
        return random.choice(gpm)

    def generate_shortest_heuristic_path(self):
        if self.game_mode == 2:
            prev_agent_location = [0, 0]
        agent_goal = (self.grid.height-1, self.grid.width-1)
        loop_count = 0
        value_total = 0
        time_total = self.grid.cost_matrix[0, 0]
        prev_squares = []
        loop_abort = 100000
        self.shortest_path = [(0,0)]
        while self.agent.location != agent_goal and loop_count < loop_abort:
            next_best_move, value = self.find_best_heuristic_move()
            # note if the agent goes to a previous square, then
            # make a random move to prevent getting into a loop
            if next_best_move in prev_squares:
                next_best_move = self.random_move()
            prev_squares.append(next_best_move)
            # try to make it visible on plot
            self.grid_path.cost_matrix[next_best_move[0]][next_best_move[1]] = -50
            self.shortest_path.append(next_best_move)
            # print(next_best_move, value, agent_location)
            value_total += self.grid.cost_matrix[next_best_move[0], next_best_move[1]]
            loop_count += 1
            if self.game_mode == 2:
                self.agent.prev_location = self.agent.location
            self.agent.location = next_best_move
            if self.game_mode == 1:
                time_total += self.grid.cost_matrix[self.agent.location[0],
                                                    self.agent.location[1]]
            else:
                time_total += abs(self.grid.cost_matrix[self.agent.location[0],
                                                   self.agent.location[1]] -
                                                      self.grid.cost_matrix[self.agent.prev_location[0],
                                                        self.agent.prev_location[1]])

        self.lowest_cost = time_total
        # need to mark up first move
        self.grid_path.cost_matrix[0][0] = -50
        if loop_count == loop_abort:
            return False
        return True

    def cost_matrix_to_nodes_edges(self):
        # in order to run Dijkstra's we will put the
        # cost_matrix into the familiar nodes and edges form
        # for that algorithm
        # Each element of the matrix is a node
        nodes = []
        for i in range(self.grid.height):
            for j in range(self.grid.width):
                nodes.append((i, j))
        # make edges seperately to simplfy the coding
        edges = {}
        # agent in this method is not really moved,
        # but is just a holder for locations
        for node in nodes:
            # all adjacent nodes are also the possible moves
            # move agent to node and see possible moves
            self.agent.location = node
            adjacent_nodes = self.get_possible_moves()
            for adjacent_node in adjacent_nodes:
                # the cost (assuming time = mode 1) will then be simply the
                # value of the node / square being moved to
                # print(node, adjacent_node)
                if self.game_mode == 1:
                    edges[(node, tuple(adjacent_node))] = self.grid.cost_matrix[adjacent_node[0]][adjacent_node[1]]
                else:  # assume game mode 2
                    edges[(node, tuple(adjacent_node))] = abs(
                        self.grid.cost_matrix[adjacent_node[0]][
                            adjacent_node[1]] -
                                self.grid.cost_matrix[node[0]][node[1]])
        # print("edges: ", edges)
        return nodes, edges

    def generate_shortest_dijkstra_path(self):
        #nodes, edges are not stored as attributes,
        # but are just used to run Dijkstra.
        nodes, edges = self.cost_matrix_to_nodes_edges()
        self.path = []
        # print(nodes)
        # print(edges)
        # input("pak")
        # https://www.youtube.com/watch?v=VnTlW572Sc4
        # This builds a new dictionary squares_graph
        # that for each square, it lists the adjacent
        # squares that can be moved to, and the cost of
        # each move.
        # This is a dictionary comprehension taken from the
        # youtube video. It is similar to a list comprehension
        # and builds a bunch of keys labelled by the nodes
        # each with an empty dict.
        squares_graph = {s: {} for s in nodes}
        # print(adjacent_squares)
        # edges.items() provides an iterable over the dictionary
        # Recall an edge is a node->node (e.g. (0,0)-(1,0))
        # cost is the cost (time in mode 1) in the square benig
        # moved to.
        for edge, cost in edges.items():
            # print(edge, cost)
            # need to split out the values since the
            # keys were made into strings.
            # start_end = edge.split("-")
            start_square = edge[0]
            # print("start_square:", start_square)
            end_square = edge[1]
            # print("end_square:", end_square)
            # cost is the same in both directions.
            # This fills out the empty dictionaries from the
            # dictionary comprehension above.
            squares_graph[start_square][end_square] = self.grid.cost_matrix[end_square]
            squares_graph[end_square][start_square] = self.grid.cost_matrix[start_square]

        # print(squares_graph)
        # print("RUNNING DIJKSTRA")
        large_number = np.sum(self.grid.cost_matrix) * 1000000
        # this sets up the initial game_costs for each node
        # to a large number, so any path will be better
        costs = {c: large_number for c in nodes}
        # the above dict will be updated by the algorithm
        # But the cost of the starting square is 0 by definition, so set it
        # costs[(0, 0)] = 0

        # https://docs.python.org/3/library/heapq.html
        # https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/
        # this queue will be used to the run the algorithm
        # initialise it with the starting square (0,0) having top priority
        # In a priority queue, the first item in tuple is higher priority
        # if it is a lower number. Thus a heappop will pop the lowest numbered
        # item first. So this pushes the start square as the highest priority item.
        priority_queue = [(0, (0, 0))]
        counter = 0
        # priority queue last item will be the shortest distance from (0,0) to the
        # target square (n,n)
        prev_node = {}
        while len(priority_queue) > 0:
            # counter +=1
            # print(f"*************  {counter}  **********")
            # print("len(priority_queue): ",len(priority_queue))
            # Note - it's only current_time for game mode 1, but using time for
            # easy reading of code.
            # Get the highest priority (lowest time numbered), square from the
            # queue using heappop.
            # It will also give us the current destination square to work on.
            current_time, current_square = heapq.heappop(priority_queue)
            # print("priority_queue: ",priority_queue)
            # print("current_time, current_square = heappop(priority_queue)")
            # print("current_time, current_square: ",current_time, current_square)
            # Check if the best time on the priority queue is less than
            # the current estimate for the least cost route to the current
            # sub-destination:
            if current_time <= costs[current_square]:
                # print("current_time <= costs[current_square]:")
                # print("costs[current_square]: ", costs[current_square])
                # print("squares_graph[current_square].items(): ", squares_graph[current_square].items())
                # print("for neighbour, time in squares_graph[current_square].items():")"""
                # Now go through all neighbours and times for the current
                # sub-destination being looked at.
                # (This was the purpose of building the squares graph earlier)

                for neighbour, time in squares_graph[current_square].items():
                    """print("neighbour, time: ",neighbour, time)
                    print("total_time = current_time + time")"""
                    # The total time to get to this neighbour
                    # will be the current time to the sub-destination
                    # plus the time plus the time to get to the sub-destination
                    # neighbour.
                    total_time = current_time + time
                    # print("total_time: ", total_time)
                    # If the calculated time for this sub-route is less that the
                    # current estimate of best time to get from (0,0)
                    # to neighbour then replace the best time with the
                    # new best time, and push it onto the queue.
                    if total_time < costs[neighbour]:
                        """print("total_time < costs[neighbour]: ", total_time, costs[neighbour])
                        print("costs[neighbour] = total_time")
                        print("costs[neighbour] ", costs[neighbour])
                        print("total_time: ", total_time)"""
                        costs[neighbour] = total_time
                        # https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
                        prev_node[neighbour] = current_square
                        # print(neighbour, total_time)
                        # print("heappush(priority_queue, (total_time, neighbour))")
                        heapq.heappush(priority_queue, (total_time, neighbour))
                        # print("total_time,neighbour: ", total_time,neighbour)
                        # print("heap priority queue: ", priority_queue)
                    #print("priority_queue ", priority_queue)
        # print(cost_matrix)
        # print(costs)
        # convert to grid
        """print(costs)
        input("pak")
        pq2 = []"""
        # corner val in costs[] should be cost of shortest path to far left
        self.lowest_cost = costs[(self.grid.height - 1,
                                  self.grid.width - 1)]
        # now convert
        """grid = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]))
        for k in costs.keys():
            # print(costs[k])
            # print(k)
            # input("pak")

            # heapq.heappush(pq2,(costs[k],k))
            grid[k] = costs[k]"""
        # print(cost_matrix)
        # print(grid)
        """print("pq2 ", pq2)
        path = [(3,3)]
        goal = (cost_matrix.shape[0]-1,cost_matrix.shape[1]-1)
        for i in range(len(pq2)):
            m = heapq.heappop(pq2)[1]
            #print("m ",m)
            if m != path[0]:
                if list(m) in get_possible_moves(path[-1], cost_matrix):
                    path.append(m)
            if m == goal:
                break
        print("path ",path)
        input("pak")"""

        """
        fig, ax = plt.subplots()
        ax.matshow(grid, cmap=plt.cm.Pastel1)
        fig.show()
        input("pak")


        print(grid)
        #minimum_neighbour(grid,(cost_matrix.shape[0]-1,cost_matrix.shape[1]-1))"""
        # print(grid)
        # print("generate_dijkstra_path(grid) ", generate_dijkstra_path(grid))
        #print(grid)
        # Now generate the path
        # https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
        node = (self.grid.height - 1, self.grid.width - 1)
        while node != (0, 0):
            self.grid_path.cost_matrix[node[0]][node[1]] = -50
            self.shortest_path.append(node)
            node = prev_node[node]
        self.shortest_path.append((0, 0))
        #reverse path
        self.shortest_path = self.shortest_path[::-1]
        # need to mark up first move
        self.grid_path.cost_matrix[0][0] = -50
        #print("Path ", path)
        #print("prev_node ", prev_node)
        #input("pak")

        #return best_cost


    def run_game(self, path_algorithm="heuristic"):
        # path_algorithm is either heuristic or Dijkstra
        # assume agent object is reset each time this is run
        if path_algorithm == "heuristic":
            self.generate_shortest_heuristic_path()
        else:
            self.generate_shortest_dijkstra_path()


    def plot_game_result(self, show_values=False):
        self.grid_path.plot_grid(show_values=True)

    def __str__(self):
        return_str = "GAME\n"
        return_str += f"Game mode {self.game_mode}\n"
        return_str += f"Initial agent location is {self.agent.location}\n"
        return_str += f"Target agent location is {self.target_location}\n"
        return_str += f"Using:\n{self.grid}\n"
        return return_str


# This class runs a series of games N times.
# For each of the N runs the average and SD of the lowest_cost is stored.
class Experiment:
    def __init__(self, game_list=[], algorithm = "heuristic", repeats=30):
        # a list of games
        game_list = game_list
        self.repeats = repeats
        self.results = []
        self.algorithm = algorithm

    def generate_game_list(self, params_list):
        # params_list is of the form:
        # [(game modes), (grid sizes), (distribution type/param pairs)]
        # e.g. [(1,2), ((10,10),(50,50),(100,100)),
        #       (("normal",(10,3)), ("normal", (50,15)),
        #           ("normal", (100,30)), ("uniform",(0,10)),
        #           ("uniform",(0,50)), ("uniform",(0,100))) ]
        game_modes = params_list[0]
        grid_sizes = params_list[1]
        distributions = params_list[2]
        self.game_list = []
        for gm in game_modes:
            for gs in grid_sizes:
                for d, d_p in distributions:
                    for rep in range(self.repeats):
                        grid = Grid(height=gs[0], width=gs[1],
                                    distribution=d, distr_params=d_p)
                        self.game_list.append(Game(game_mode=gm,
                                              grid=grid))


    def run_experiments(self, verbose=False, export="results.csv"):
        if export != "":
            f = open(export, 'w')
            to_write = ["algorithm", "game_mode", "grid.height", "grid.width", "grid.distribution",
                        "grid.distr_params","grid.distr_params", "mean shortest path", "SD shortest path"]
            f.write(','.join(to_write)+"\n")
        # need to create new result element after each self.repeats repeats
        repeats_index = 0
        result = []
        for game in self.game_list:
            if verbose:
                print(game)
            game.run_game(path_algorithm=self.algorithm)
            result.append(game.lowest_cost)
            repeats_index += 1
            # need to create new result element after each self.repeats repeats
            if repeats_index % self.repeats == 0:
                #print(game)
                if ROUND:
                    stats = (round(statistics.mean(result),0), round(statistics.stdev(result),0))
                else:
                    stats = (statistics.mean(result), statistics.stdev(result))
                self.results.append(stats)
                if export != "":
                    if len(game.grid.distr_params) == 2:
                        to_write = [self.algorithm, game.game_mode,game.grid.height, game.grid.width, game.grid.distribution, *game.grid.distr_params, *stats]
                    else:
                        to_write = [self.algorithm, game.game_mode, game.grid.height, game.grid.width,
                                    game.grid.distribution, game.grid.distr_params[0],0, *stats]
                    to_write = [str(tw) for tw in to_write]
                    to_write = ','.join(to_write)+"\n"
                    f.write(to_write)
                result = []
        if export != "":
            f.close()

    def __str__(self):
        result_str = f"Experiment {self.algorithm}\n"
        for g in self.game_list:
            result_str += str(g)
        return result_str

#grd = Grid(width=15, height=15, distribution="randint",distr_params=[0,10])
#g = Grid(distribution="normal", distr_params=[150,50])
#print(g)
#g.plot_grid()
"""g = Game(grid=grd)
print(g)
g.grid.plot_grid(show_values=True)
g.run_game(path_algorithm="heuristic")
print(g.shortest_path)
print(g.lowest_cost)
g.plot_game_result(show_values=True)
g = Game(grid=grd)
g.run_game(path_algorithm="dijkstra")
print(g.shortest_path)
print(g.lowest_cost)
g.plot_game_result(show_values=True)"""

#e = Experiment(algorithm="heuristic")
param_list = [(1,2), ((5,5),(10,10)),
               (("normal",(10,3)), ("normal", (50,15)),
                   ("normal", (100,30)), ("uniform",(0,10)),
                   ("uniform",(0,50)), ("uniform",(0,100))) ]
#e.generate_game_list(param_list)
#print(e)
#e.run_experiments()
e2 = Experiment(algorithm="dijkstra")
e2.generate_game_list(param_list)
e2.run_experiments(export="results_d.csv")
