import numpy as np
import matplotlib.pyplot as plt
import random


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
    cost_matrix_path[0, 0] = -50  # mark this so the start of the path shows up
    return cost_matrix, cost_matrix_path


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
        print(next_best_move, value, agent_location)
        value_total += value
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
        agent_location, time_total, loop_count = run_game(cost_matrix,
                                                          cost_matrix_path,
                                                          game_mode=game_mode)
        # if it succeeded
        if agent_location == cost_matrix.shape:
            results.append((time_total, loop_count))
        print("Final agent location: ",agent_location)
        print("Total time: ", time_total)
        print("Number of moves: ", loop_count)
        if plot_me:  # for debug purposes
            plot_results(cost_matrix, cost_matrix_path)
    return results


exp1 = {'height': 15, 'width': 15, 'distribution_type': "uniform",
        'distribution_params': (0, 9),
        'game_mode': 0, 'algorithm_type': "heuristic"}
exp2 = {'height': 15, 'width': 15, 'distribution_type': "uniform",
        'distribution_params': (0, 9),
        'game_mode': 1, 'algorithm_type': "heuristic"}
exp_list = [exp1, exp2]
results = run_experiments(exp_list, plot_me=True)