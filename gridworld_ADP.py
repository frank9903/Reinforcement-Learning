import run_world
from run_world import *
import numpy as np

N_e = 30
R_plus = 2
actions = range(4)
N_sa_dic = {}
N_sas_dic = {}
grid = read_grid("lecture")
immediate_reward = 0
rewards = []
discount_factor = 1
utils = [[0] * 4 for i in range(3)]
conv = []
conv_cnt = 0
policy = []

def load_grid(name):
    global N_sa_dic
    global N_sas_dic
    global grid
    global immediate_reward
    global discount_factor
    global utils
    global rewards
    global conv
    global conv_cnt
    global policy

    grid = read_grid(name)
    immediate_reward = get_reward(name)
    discount_factor = get_gamma(name)
    utils = [[0] * 4 for i in range(3)]
    rewards = [[immediate_reward] * 4 for i in range(3)]
    conv = [[False] * 4 for i in range(3)]
    conv_cnt = 0
    policy = np.zeros(grid.shape, dtype='int32')
    for i in range(3):
        for j in range(4):
            for a in actions:
                N_sa_dic[((i,j), a)] = 0
                for s in get_next_states(grid, (i,j)):
                    N_sas_dic[((i,j), a, s)] = 0
            if is_goal(grid, (i,j)):
                utils[i][j] = int(grid[i][j])
                rewards[i][j] = int(grid[i][j])

def explore(u, n):
    if n < N_e:
        return R_plus
    else:
        return u
    
def get_prob(n_sa, n_sas):
    if n_sa == 0:
        return 0
    return n_sas / n_sa

def exp_utils(curr_state, action):
    result = 0
    next_states = get_next_states(grid, curr_state)
    next_states = set(next_states)
    for s_prime in next_states:
        result += get_prob(N_sa_dic[(curr_state, action)], N_sas_dic[(curr_state, action, s_prime)]) * utils[s_prime[0]][s_prime[1]]
    return result

def optimistic_exp_utils(curr_state):
    start = True
    optim = 0
    best_a = 0
    for a in actions:
        temp = explore(exp_utils(curr_state, a), N_sa_dic[(curr_state, a)])
        if start or temp > optim:
            optim = temp
            best_a = a
            start = False
    return best_a, optim

def update_utils(curr_state, prob_conv=False):
    global conv_cnt
    best_a, optim = optimistic_exp_utils(curr_state)
    U = rewards[curr_state[0]][curr_state[1]] + discount_factor * optim
    prev_U = utils[curr_state[0]][curr_state[1]]
    if prob_conv:
        if not conv[curr_state[0]][curr_state[1]]:
            if abs(prev_U - U) <= 0.001:
                conv[curr_state[0]][curr_state[1]] = True
                conv_cnt += 1
                # print(conv_cnt)
    utils[curr_state[0]][curr_state[1]] = U

# def update_all():
#     for i in range(3):
#         for j in range(4):
#             state = (i,j)
#             if not (is_goal(grid, state) or is_wall(grid, state)):
#                 update_utils(state)

def random_state():
    state = None
    while True:
        row = random.sample(range(3), k=1)[0]
        col = random.sample(range(4), k=1)[0]
        state = (row, col)
        if not_goal_and_wall(grid, state):
            break
    return state

cur = 12*4
def check_prob():
    global cur
    nums = N_sa_dic.values()
    lst = [x for x in nums if x < N_e]
    # temp = len(lst)
    # if (not temp == cur):
    #     cur = temp
    #     print(cur)
    return len(lst) == 0

def main(name):
    load_grid(name)
    curr_state = (0,0)
    i = 0
    while not check_prob() and i < 1000000:
        i += 1
        best_a, optim = optimistic_exp_utils(curr_state)
        next_state = make_move(grid, curr_state, best_a, name)
        N_sa_dic[(curr_state, best_a)] += 1
        N_sas_dic[(curr_state, best_a, next_state)] += 1
        update_utils(curr_state)
        policy[curr_state] = best_a
        if (is_goal(grid, next_state)):
            curr_state = random_state()
        else:
            curr_state = next_state

    i = 0
    while conv_cnt < 9 and i < 2000000:
        i += 1
        best_a, optim = optimistic_exp_utils(curr_state)
        next_state = make_move(grid, curr_state, best_a, name)
        N_sa_dic[(curr_state, best_a)] += 1
        N_sas_dic[(curr_state, best_a, next_state)] += 1
        update_utils(curr_state, True)
        policy[curr_state[0]][curr_state[1]] = best_a
        if (is_goal(grid, next_state)):
            curr_state = random_state()
        else:
            curr_state = next_state

print("=====================LECTURE GRID=======================")
main('lecture')
print(np.array(utils))
pretty_print_policy(grid, policy)
print("=====================A4 GRID=======================")
main('a4')
print(np.array(utils))
pretty_print_policy(grid, policy)

