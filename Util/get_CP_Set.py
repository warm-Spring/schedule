from Priority import CalRankd as cd, CalRanku as cu
import importlib
import config

module_name = config.dag_path
td = importlib.import_module(module_name)

def get_cp_set():
    rank_u = cu.cal_rank_u(td.w_array)
    rank_d = cd.cal_rank_d(td.w_array)
    rank_exit = rank_u[td.exit] + rank_d[td.exit]
    rank_sum = rank_u+rank_d
    cp_set = []
    for i in range(0,len(rank_u)):
        if abs(rank_sum[i]-rank_exit) <= 0.05:
            cp_set.append(i)
    return  cp_set


def get_min_cp():
    min_cp = 0
    cp_set = get_cp_set();
    for i in cp_set:
        min_cp += min(td.w_array[i])
    return min_cp

