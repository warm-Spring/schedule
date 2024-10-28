import math
import importlib
import config

module_name = config.dag_path
yt = importlib.import_module(module_name)

def get_w_i_k_v(n_i, u_k, f_k_v):
    f_k_max = yt.f_k_max_list[u_k]
    w_i_k = yt.w_array[n_i][u_k - 1]
    w_i_k_v = w_i_k * f_k_max / f_k_v
    return w_i_k_v


def get_R_ni(n_i, u_k, f_k_v):
    lambda_k_max = yt.lambda_uk_max_list[u_k]
    f_k_max = yt.f_k_max_list[u_k]
    f_k_min = yt.f_k_min_list[u_k]
    lambda_k_v = lambda_k_max * 10 ** (yt.d * (f_k_max - f_k_v) / (f_k_max - f_k_min))
    w_i_k_v = get_w_i_k_v(n_i, u_k, f_k_v)
    R_ni = math.exp(-1 * lambda_k_v * w_i_k_v)
    return R_ni


def get_E_ni(n_i, u_k, f_k_v):
    P_k_v = yt.P_k_ind[u_k-1] + yt.C_k_ef[u_k-1] * (f_k_v**yt.m_k[u_k-1])
    w_i_k_v = get_w_i_k_v(n_i, u_k, f_k_v)
    return P_k_v*w_i_k_v

