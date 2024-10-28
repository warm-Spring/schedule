import numpy as np
import importlib
import config

module_name = config.dag_path
yt = importlib.import_module(module_name)
w_ave = np.sum(yt.w_array, axis=1) / len(yt.w_array[0])


def max_up(t_i, succ_t, rank_u):
    max_succ = yt.t_array[t_i][succ_t[0]] + rank_u[succ_t[0]]
    max_succ = np.max(max_succ)

    return max_succ


def cal_rank_u(w_array):
    rank_u = np.zeros(len(w_array))

    for i in range(len(rank_u) - 1, 0, -1):
        if i == (len(w_array) - 1):
            rank_u[i] = w_ave[i]
            rank_u[i] = rank_u[i]
        else:
            succ_t = yt.t_net[np.where(yt.t_net[:, 0] == i), 1]
            rank_u[i] = w_ave[i] + max_up(i, succ_t, rank_u)
            rank_u[i] = rank_u[i]

    for i in range(0, len(rank_u)):
        rank_u[i] = rank_u[i]
    return rank_u



if __name__ == "__main__":
    print(cal_rank_u(yt.w_array))