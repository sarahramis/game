import numpy as np
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush


def create_grid(height=15, width=15,
                distribution_type="uniform", distr_params = (0,9), seed=10):
    if seed > 0:
        np.random.seed(seed)   # fix it so can compare more easily for tests
        random.seed(seed)
    if distribution_type == "uniform":
        n = distr_params
        cost_matrix = np.random.randint(distr_params[0],
                                        distr_params[1]+1,
                                        size=(height, width))
    elif distribution_type == "normal":
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        # for this type, distr params is (mean, sd)
        cost_matrix = np.random.normal(distr_params[0],
                                       distr_params[1],
                                       size=(height, width)) \
            # this version is purely to display the path
    cost_matrix_path = cost_matrix.copy()
    cost_matrix[0,0] = 0        # no point in having a cost on the first square or last square
    cost_matrix[-1,-1] = 0
    cost_matrix_path[0, 0] = -50  # mark this so the start of the path shows up
    return cost_matrix, cost_matrix_path


def cost_matrix_to_nodes_edges(cost_matrix, game_mode):
    # in order to run Dijkstra's we will put the
    # cost_matrix into the familiar nodes and edges form
    # for that algorithm
    # Each element of the matrix is a node
    nodes = []
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            nodes.append((i,j))
    # make edges seperately to simplfy the coding
    edges = {}
    for node in nodes:
        # all adjacent nodes are also the possible moves
        adjacent_nodes = get_possible_moves(node, cost_matrix)
        for adjacent_node in adjacent_nodes:
            # the cost (assuming time = mode 1) will then be simply the
            # value of the node / square being moved to
            #edges[str(node) + "-" + str(tuple(adjacent_node))] = cost_matrix[adjacent_node[0]][adjacent_node[1]]
            #print(node, adjacent_node)
            if game_mode == 0:
                edges[(node, tuple(adjacent_node))] = cost_matrix[adjacent_node[0]][adjacent_node[1]]
            else:   #assume game mode 1
                edges[(node, tuple(adjacent_node))] = abs(cost_matrix[adjacent_node[0]][adjacent_node[1]] - cost_matrix[node[0]][node[1]])
    print("edges: ", edges)
    return nodes, edges


def plot_results(cost_matrix, cost_matrix_path):
    height = cost_matrix.shape[0]
    width = cost_matrix.shape[1]
    # fig and ax simply show the grid untraversed.
    # https://stackoverflow.com/questions/40887753/display-matrix-values-and-colormap
    fig, ax = plt.subplots()
    ax.matshow(cost_matrix, cmap=plt.cm.Pastel1)
    for i in range(height):
        for j in range(width):
            ax.text(i, j, str(cost_matrix[j,i]), va='center', ha='center')

    #fig2 and ax2 show the grid with the path chosen
    fig2, ax2 = plt.subplots()
    ax2.matshow(cost_matrix_path, cmap=plt.cm.Pastel1)
    for i in range(height):
        for j in range(width):
            ax2.text(i, j, str(cost_matrix_path[j,i]), va='center', ha='center')
    fig.show()
    fig2.show()


def get_possible_moves(agent_location, cost_matrix):
    # all possible moves, including diagonals
    all_moves = [(1,0),(0,1),
                      (-1,0),(0,-1)] #,(-1,1),(1,-1), (-1,-1), (1,1)]
    height = cost_matrix.shape[0]
    width =  cost_matrix.shape[1]
    # now strip out those that take it off the grid
    possible_moves = []
    for move in all_moves:
        new_agent_location = (agent_location[0]+move[0],
                              agent_location[1] + move[1])
        # will this move keep it on the grid? Keep it if it does.
        if new_agent_location[0] >= 0 and new_agent_location[1] >= 0 \
            and new_agent_location[0] < height and new_agent_location[1] < width:
            possible_moves.append([agent_location[0]+move[0],
                                  agent_location[1]+move[1]])
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
def get_values(agent_location, moves, game_mode, cost_matrix):
    reward = 5
    values = []
    for move in moves:
        value = -1 * cost_matrix[move[0]][move[1]]
        if game_mode == 1:
            value = abs(value + cost_matrix[agent_location[0]][agent_location[1]])
        if move[0] > agent_location[0]:  # down
            value += reward
        if move[1] > agent_location[1]:
                value += reward  # right
        #print("Matrix value: ",cost_matrix[move[0]][move[1]])
        #print("Value: ", value)
        values.append(value)
    return values


# For each move to x,y the agent calculates C = TIME(x,y)-sum_of_move_costs
# The agent then picks the x,y move that minimises C
# if two versions of C are equal, the agent randomly picks one
def find_best_move(agent_location, game_mode, cost_matrix):
    gpm = get_possible_moves(agent_location, cost_matrix)
    values = get_values(agent_location, gpm, game_mode, cost_matrix)
    max_move_index = values.index(max(values))
    return gpm[max_move_index], max(values)


def random_move(agent_location, cost_matrix):
    gpm = get_possible_moves(agent_location, cost_matrix)
    # https://stackoverflow.com/questions/306400/how-can-i-randomly-select-an-item-from-a-list
    return random.choice(gpm)


def run_game(cost_matrix, cost_matrix_path, game_mode=0):
    if game_mode == 1:
        prev_agent_location = [0,0]
    agent_location = [0,0]
    # far right hand corner
    agent_goal = [cost_matrix.shape[0]-1,cost_matrix.shape[1]-1]
    loop_count = 0
    value_total = 0
    time_total = cost_matrix[0, 0]
    prev_squares = []
    while agent_location != agent_goal and loop_count < 100000:
        next_best_move, value= find_best_move(agent_location, game_mode, cost_matrix)
        # note if the agent goes to a previous square, then
        # make a random move to prevent getting into a loop
        if next_best_move in prev_squares:
            next_best_move = random_move(agent_location, cost_matrix)
        prev_squares.append(next_best_move)
        # try to make it visible on plot
        cost_matrix_path[next_best_move[0],next_best_move[1]] = -50
        #print(next_best_move, value, agent_location)
        value_total += cost_matrix[next_best_move[0],next_best_move[1]]
        loop_count += 1
        if game_mode == 1:
            prev_agent_location = agent_location
        agent_location = next_best_move
        if game_mode == 0:
            time_total += cost_matrix[agent_location[0],agent_location[1]]
        else:
            time_total += abs(cost_matrix[agent_location[0],
                            agent_location[1]]-cost_matrix[prev_agent_location[0],
                                                           prev_agent_location[1]])

    return agent_location, time_total, loop_count


def dijkstra_on_matrix(cost_matrix, game_mode):
    nodes, edges = cost_matrix_to_nodes_edges(cost_matrix, game_mode)

    #print(nodes,edges)
    #input("pak")
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
    #print(adjacent_squares)
    # edges.items() provides an iterable over the dictionary
    # Recall an edge is a node->node (e.g. (0,0)-(1,0))
    # cost is the cost (time in mode 1) in the square benig
    # moved to.
    for edge, cost in edges.items():
        #print(edge, cost)
        # need to split out the values since the
        # keys were made into strings.
        #start_end = edge.split("-")
        start_square = edge[0]
        #print("start_square:", start_square)
        end_square = edge[1]
        #print("end_square:", end_square)
        # cost is the same in both directions.
        # This fills out the empty dictionaries from the
        # dictionary comprehension above.
        squares_graph[start_square][end_square] = cost_matrix[end_square]
        squares_graph[end_square][start_square] = cost_matrix[start_square]

    #print(squares_graph)
    #print("RUNNING DIJKSTRA")
    #input("pak")
    large_number = np.sum(cost_matrix) * 1000000
    # this sets up the initial game_costs for each node
    # to a large number, so any path will be better
    costs = {c: large_number for c in nodes}
    # the above dict will be updated by the algorithm
    # But the cost of the starting square is 0 by definition, so set it
    #costs[(0, 0)] = 0

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
    while len(priority_queue) > 0:
        #counter +=1
        #print(f"*************  {counter}  **********")
        #print("len(priority_queue): ",len(priority_queue))
        # Note - it's only current_time for game mode 1, but using time for
        # easy reading of code.
        # Get the highest priority (lowest time numbered), square from the
        # queue using heappop.
        # It will also give us the current destination square to work on.
        current_time, current_square = heappop(priority_queue)
        #print("priority_queue: ",priority_queue)
        #print("current_time, current_square = heappop(priority_queue)")
        #print("current_time, current_square: ",current_time, current_square)
        # Check if the best time on the priority queue is less than
        # the current estimate for the least cost route to the current
        # sub-destination:
        if current_time <= costs[current_square]:
            #print("current_time <= costs[current_square]:")
            #print("costs[current_square]: ", costs[current_square])
            #print("squares_graph[current_square].items(): ", squares_graph[current_square].items())
            #print("for neighbour, time in squares_graph[current_square].items():")"""
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
                #print("total_time: ", total_time)
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
                    #print("heappush(priority_queue, (total_time, neighbour))")
                    heappush(priority_queue, (total_time, neighbour))
                    #print("heap priority queue: ", priority_queue)

    # final val in dictionary should be cost of shortest path to far left
    return costs[(cost_matrix.shape[0]-1,cost_matrix.shape[1]-1)]


def run_experiments(list_of_params, plot_me=False):
    # list_of_params is list of dicts:
    # {'height': number, 'width': number,
    # 'distribution_type': "normal" or "uniform",
    # 'distribution_params': (number, number),
    # 'game_mode': 0 or 1, 'algorithm_type: "heuristic" or "dijkstra"}
    results = []
    for params in list_of_params:
        height = params['height']
        width = params['width']
        distribution_type = params['distribution_type']
        distribution_params = params['distribution_params']
        game_mode = params['game_mode']
        # This is ignored for now
        algorithm_type = params['algorithm_type']
        cost_matrix, cost_matrix_path = create_grid(height=height,
                                                    width=width,
                                                    distribution_type=distribution_type,
                                                    distr_params=distribution_params,
                                                    seed=10)
        print(cost_matrix)
        if algorithm_type == "heuristic":
            print("Heuristic")
            agent_location, time_total, loop_count = run_game(cost_matrix,
                                                              cost_matrix_path,
                                                              game_mode=game_mode)
        else:       # assume it is dijkstra:
            print("Dijkstra")
            min_cost = dijkstra_on_matrix(cost_matrix, game_mode)
            #agent_location = cost_matrix.shape  # it will always succeed
            time_total = min_cost
        # if it succeeded
        #if agent_location == cost_matrix.shape:
        results.append(time_total)
        #print("Final agent location: ",agent_location)
        #print("Total time: ", time_total)
        #print("Number of moves: ", loop_count)
        if plot_me and algorithm_type == "heuristic":  # for debug purposes
            plot_results(cost_matrix, cost_matrix_path)
    return results

#cost_matrix, cost_matrix_path = create_grid(width = 2, height = 2, seed = 21)
#cost_matrix[(0,0)] = 0  #since we start at the top left
#nodes, edges = cost_matrix_to_nodes_edges(cost_matrix)
#print(cost_matrix)
#print(nodes)
#print(edges)
#min_cost = dijkstra_on_matrix(cost_matrix)
#print(min_cost)
exp1 = {'height': 4, 'width': 4, 'distribution_type': "uniform",
        'distribution_params': (0, 9),
        'game_mode': 1, 'algorithm_type': "heuristic"}
"""exp2 = {'height': 15, 'width': 15, 'distribution_type': "uniform",
        'distribution_params': (0, 9),
        'game_mode': 1, 'algorithm_type': "heuristic"}"""
exp3 = {'height': 4, 'width': 4, 'distribution_type': "uniform",
        'distribution_params': (0, 9),
        'game_mode': 1, 'algorithm_type': "dijkstra"}
exp_list = [exp1, exp3]
results = run_experiments(exp_list, plot_me=True)
print(results)