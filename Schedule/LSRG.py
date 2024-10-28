from Performance import performance as pm
from Priority import CalRanku
import numpy as np
import importlib
import config

module_name = config.dag_path
td = importlib.import_module(module_name)

def get_R_G_bound():
    R_min_G = 1
    R_max_G = 1
    for i in range(1, td.taskNum + 1):
        R_min_ni = 1
        R_max_ni = 0
        temp = 1
        for k in range(1, td.vmNum + 1):
            f_k_min = td.f_k_min_list[k]
            f_k_max = td.f_k_max_list[k]
            R_min_ni = min(R_min_ni, pm.get_R_ni(i, k, f_k_min))
            temp = temp * (1 - pm.get_R_ni(i, k, f_k_max))
        R_max_ni = 1 - temp
        R_min_G *= R_min_ni
        R_max_G *= R_max_ni
    return R_min_G, R_max_G


def LSRG(R_goal_G):
    R_min_G, R_max_G = get_R_G_bound()
    if R_goal_G < R_min_G or R_goal_G > R_max_G:
        return -1
    R_actual_x_set = []
    rank_u = CalRanku.cal_rank_u(td.w_array)
    t_pri = np.argsort(-rank_u[:])[0:-1]
    u_pri = []
    v_pri = []
    R_goal_z = R_goal_G ** (1 / td.taskNum)
    R_goal_ni_set = [round(R_goal_z, 6)]
    AST = []
    AFT = []
    FreeSlot = np.zeros(td.vmNum)

    for i in range(0, len(t_pri)):
        R_actual_y = 0
        u_k = 0
        f_v = 0
        min_EFT = 10000000
        # AFT
        AFT_i = 0
        AST_i = 0
        for u in range(1, td.vmNum + 1):
            for f in range(int(td.f_k_min_list[u] * 100), int((td.f_k_max_list[u]) * 100 + 1), 1):
                current_R_ni = pm.get_R_ni(t_pri[i], u, f / 100)
                if current_R_ni >= R_goal_ni_set[i]:
                    current_T_ni = pm.get_w_i_k_v(t_pri[i], u, f / 100)
                    if i == 0:
                        if min_EFT > current_T_ni:
                            min_EFT = current_T_ni
                            u_k = u
                            f_v = f
                            R_actual_y = current_R_ni
                    else:
                        pred_t = np.ravel(td.t_net[np.where(td.t_net[:, 1] == t_pri[i]), 0])
                        max_pred_AFT_i = 0
                        for pred in pred_t:
                            indices = int(np.where(t_pri == pred)[0])
                            pred_AFT = AFT[indices]
                            if u == u_pri[indices]:
                                c_i_j = 0
                            else:
                                c_i_j = td.t_array[t_pri[indices]][t_pri[i]]
                            max_pred_AFT_i = max(max_pred_AFT_i, c_i_j + pred_AFT)
                        AST_i = max(max_pred_AFT_i, FreeSlot[u-1])
                        AFT_i = AST_i + current_T_ni
                        if min_EFT > AFT_i:
                            min_EFT = AFT_i
                            u_k = u
                            f_v = f
                            R_actual_y = current_R_ni
        AST.append(AST_i)
        AFT.append(min_EFT)
        u_pri.append(u_k)
        v_pri.append(f_v/100)
        R_actual_x_set.append(R_actual_y)
        FreeSlot[u_k-1] = min_EFT
        R_goal_z_prod = 1
        R_actual_x = np.prod(R_actual_x_set)
        for j in range(1, td.taskNum - len(R_actual_x_set)):
            R_goal_z_prod *= R_goal_z
        R_goal_y = R_goal_G / (R_actual_x * R_goal_z_prod)
        if i != len(t_pri) - 1:
            R_goal_ni_set.append(R_goal_y)
    R_goal_ni_set = [round(num, 6) for num in R_goal_ni_set]
    return t_pri, u_pri, v_pri, R_goal_ni_set, AST, AFT


