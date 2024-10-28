import mpmath
import numpy as np
from Performance import performance as pm
from Priority import CalRanku
import importlib
import config

module_name = config.dag_path
yt = importlib.import_module(module_name)

def get_R_G_bound():
    R_min_G = 1
    R_max_G = 1
    for i in range(1, yt.taskNum + 1):
        R_min_ni = 1
        R_max_ni = 0
        temp = 1
        for k in range(1, yt.vmNum + 1):
            f_k_min = yt.f_k_min_list[k]
            f_k_max = yt.f_k_max_list[k]
            R_min_ni = min(R_min_ni, pm.get_R_ni(i, k, f_k_min))
            temp = temp * (1 - pm.get_R_ni(i, k, f_k_max))
        R_max_ni = 1 - temp
        R_min_G *= R_min_ni
        R_max_G *= R_max_ni
    return R_min_G, R_max_G



def ESRG(R_goal_G):
    R_min_G, R_max_G = get_R_G_bound()
    if R_goal_G<R_min_G or R_goal_G>R_max_G:
        return -1
    R_actual_x_set = []
    rank_u = CalRanku.cal_rank_u(yt.w_array)
    t_pri = np.argsort(-rank_u[:])[0:-1]
    u_pri = []
    v_pri = []
    R_goal_z = R_goal_G ** (1 / yt.taskNum)
    R_goal_ni_set = [R_goal_z]
    for i in range(0, len(t_pri)):
        R_actual_y = 0
        E_ni_min = 1000000
        u_k = 0
        f_v = 0
        flag = 0
        for u in range(1, yt.vmNum+1):
            for f in range(int(yt.f_k_min_list[u]*100), int((yt.f_k_max_list[u])*100+1), 1):
                current_R_ni = pm.get_R_ni(t_pri[i], u, f/100)
                if current_R_ni > R_goal_ni_set[i]:
                    flag = 1
                    current_E_ni = pm.get_E_ni(t_pri[i], u, f/100)
                    if current_E_ni < E_ni_min:
                        E_ni_min = current_E_ni
                        R_actual_y = current_R_ni
                        u_k = u
                        f_v = f/100

        if flag == 0:
            break
        R_actual_x_set.append(R_actual_y)
        u_pri.append(u_k)
        v_pri.append(f_v)

        R_goal_z_prod = 1
        R_actual_x = np.prod(R_actual_x_set)
        mpmath.mp.dps = 50
        R_goal_G = mpmath.mpf(R_goal_G)
        R_actual_x = mpmath.mpf(R_actual_x)
        R_goal_z = mpmath.mpf(R_goal_z)
        num = mpmath.mpf(yt.taskNum - len(R_actual_x_set) - 1)
        R_goal_y = R_goal_G / (R_actual_x * (mpmath.power(R_goal_z,num)))
        if i != len(t_pri) - 1:
            R_goal_ni_set.append(R_goal_y)
    return t_pri, u_pri, v_pri, R_goal_ni_set





