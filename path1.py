import numpy as np
import matplotlib.pyplot as plt
import random

height = 15
width = 15
n = 9
np.random.seed(10)   # fix it so can compare the two game modes
random.seed(10)
cost_matrix = np.random.randint(0, n+1, size=(height, width))

# https://stackoverflow.com/questions/40887753/display-matrix-values-and-colormap
fig, ax = plt.subplots()

ax.matshow(cost_matrix, cmap=plt.cm.Pastel1)

for i in range(height):
    for j in range(width):
        ax.text(i, j, str(cost_matrix[j,i]), va='center', ha='center')


def get_possible_moves(agent_location):
    # all possible moves, including diagonals
    all_moves = [(1,0),(0,1),
                      (-1,0),(0,-1),(-1,1),(1,-1), (-1,-1), (1,1)]

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

# this version is purely to display the path
cost_matrix_path = cost_matrix.copy()
cost_matrix_path[0, 0] = -50 # mark this so the start of the path shows up

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
def get_values(agent_location, moves, game_mode):
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
def find_best_move(agent_location, game_mode):
    gpm = get_possible_moves(agent_location)
    values = get_values(agent_location, gpm, game_mode)
    max_move_index = values.index(max(values))
    return gpm[max_move_index], max(values)

def random_move(agent_location):
    gpm = get_possible_moves(agent_location)
    # https://stackoverflow.com/questions/306400/how-can-i-randomly-select-an-item-from-a-list
    return random.choice(gpm)


game_mode = 0
if game_mode == 1:
    prev_agent_location = [0,0]
agent_location = [0,0]
agent_goal = [width-1, height-1]
loop_count = 0
value_total = 0
time_total = cost_matrix[0, 0]
prev_squares = []
while agent_location != agent_goal and loop_count < 100000:
    next_best_move, value= find_best_move(agent_location, game_mode)
    # note if the agent goes to a previous square, then
    # make a random move to prevent getting into a loop
    if next_best_move in prev_squares:
        next_best_move = random_move(agent_location)
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


print("Final agent location: ",agent_location)
print("Total time: ", time_total)
print("Number of movesL: ", loop_count)

fig2, ax2 = plt.subplots()

ax2.matshow(cost_matrix_path, cmap=plt.cm.Pastel1)

for i in range(height):
    for j in range(width):
        ax2.text(i, j, str(cost_matrix_path[j,i]), va='center', ha='center')

"""
gpm = get_possible_moves(agent_location)
print(gpm)
print(get_values(agent_location, gpm))
print(find_best_move(agent_location))"""
fig.show()
fig2.show()