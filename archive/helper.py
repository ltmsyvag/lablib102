#%%
import numpy as np
from scipy.stats import norm
def velo_dist_discretization(
        arr_velo_positive: np.ndarray, 
        mu: float=0, 
        sigma: float=174.8769038)->tuple:
    """
    +-+ <-- 该 bar 的面积为速度为 0 的原子的概率占比
    | |
    | +--+
    | |  |
    | |  +----+
    | |  |    |
    • •  •    •
    0 v1 v2 … v_final <-- v_vinal 在输入速度列表中有, 在返回速度列表中会被删除

    输入 [0, v1, v2, …, v_final] 一组速度(m/s), 不必等间距. 
    然后函数根据用户输入的多普勒速度分布的 µ 和 sigma, 
    给出包含负速度的一位速度数组, 以及每个速度的原子所对应的概率占比
    返回 arr_velo, arr_probs 和总概率 prob_tot (作为离散化精细度的一个衡量)
    """
    #v 这是上面的 bar 的面积的列表, 是概率, 不是概率密度(bar 的高度)
    arr_prob_positive = (norm.pdf(arr_velo_positive, mu, sigma)[:-1] # velo prob density val
              * (arr_velo_positive[1:] - arr_velo_positive[:-1])) # velo bin size
    arr_prob = np.hstack((np.flip(arr_prob_positive), arr_prob_positive))
    arr_velo = np.hstack((-np.flip(arr_velo_positive), arr_velo_positive))[1:-1]
    prob_tot = arr_prob.sum()
    return arr_prob, arr_velo, prob_tot
    
    
if __name__ == "__main__":
    arr_velo_p = np.arange(0,1000,0.01)
    _, _, prob_tot = velo_dist_discretization(arr_velo_p)
    print(prob_tot) # very close to unity!