import numpy as np
import importlib
import config

module_name = config.dag_path
td = importlib.import_module(module_name)
w_ave = np.sum(td.w_array, axis=1) / len(td.w_array[0])

def max_up(t_i, pred_t, rank_d):
    max_pred = 0
    for j in pred_t[0]:
        temp = td.t_array[j][t_i] + rank_d[j] + w_ave[j]
        max_pred = max(temp,max_pred)
    return max_pred


def cal_rank_d(w_array):
    rank_d = np.zeros(len(w_array))
    for i in range(0, len(rank_d) , 1):
        if i == 0 or i == 1:
            rank_d[i] = 0
        else:
            pred_t = td.t_net[np.where(td.t_net[:, 1] == i), 0]
            rank_d[i] = max_up(i, pred_t, rank_d)
            rank_d[i] = round(rank_d[i], 4)
    return rank_d
