import numpy as np
from Priority import CalRanku
import importlib
import config

module_name = config.dag_path
td = importlib.import_module(module_name)

def get_outter_set(threshold):
    rank_u = CalRanku.cal_rank_u(td.w_array)
    t_pri = np.argsort(-rank_u[:])[0:-1]
    origin_outdegree_set = np.zeros(td.taskNum)
    outdegree_large_count = np.zeros(td.taskNum)
    large_outdegree_set =[]
    for single_relation in td.t_net:
        pred = single_relation[0]
        succ = single_relation[1]
        origin_outdegree_set[pred-1]+=1
    for i in range(0, len(origin_outdegree_set)):
        for j in range(0, len(origin_outdegree_set)):
            if origin_outdegree_set[i]>origin_outdegree_set[j]:
                outdegree_large_count[i]+=1
    outdegree_large_percent = outdegree_large_count/td.taskNum
    for i in range(0, len(origin_outdegree_set)):
        if outdegree_large_percent[i] >= threshold:
            large_outdegree_set.append(i+1)
    return large_outdegree_set


