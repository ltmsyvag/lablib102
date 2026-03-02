#%%
from scipy.signal import find_peaks
import numpy as np
from matplotlib.axes import Axes
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.stats import norm
from scipy.constants import k as kB, atomic_mass
from collections.abc import Sequence
from typing import Optional, Tuple

def peaks2binary(nWinPnts, analogData, height=1):
    """
    convert analog data series (peaks) into binary data (1001020...), which is returned.
    bin size is nWinPnts, the tailing points less than nWinPnts is thrown away
    """
    idPeaks, _ = find_peaks(analogData, height=height)
    peakPredicates = np.array([False]*len(analogData))
    for id in idPeaks: peakPredicates[id] = True
    remainder = len(analogData)%nWinPnts
    binary = peakPredicates[:-remainder if remainder else None].reshape((-1, nWinPnts))
    binary = binary.sum(axis=1)
    return binary
def peaks2binary2(nWinPnts, analogData, height=1):
    """
    same as peaks2binary, but the binary has the same length as the analogData
    note that nWinPnts does nothing, it's just for the sake of compatibility with peaks2binary, 
    so that fast switching between the two functions is possible
    """
    idPeaks, _ = find_peaks(analogData, height=height)
    peakPredicates = np.array([False]*len(analogData))
    for id in idPeaks: peakPredicates[id] = True
    binary = peakPredicates
    return binary


def extend_Axes_methods(c: type[Axes])-> type[Axes]: # 所有的类的 type 都是 `type`, 但是由于下面的函数 type annotation 中要写具体的 class `Axes`, 因此, 反正都要 import 这个 `Axes` 对象, 就在 decorator 输入的 type annotation 中也 explicitly 写出 `type[Axes]` 吧. 否则直接写 `type` 也无不可
    """
    为 matplotlib Axes 实例(e.g. ax) 猴子添加方法
    """
    assert c is Axes, "这个类装饰器仅用于装饰 plt.Axes 类"
    def color_right_yax(self: Axes, color: str)->None:
        """
        decorator 专用函数, 将 Axes 对象右侧 yax 涂成颜色 color
        """
        self.tick_params(which = "both", colors = color) # tick color, both major and minor ticks
        self.spines["right"].set_color(color) # edge color
        self.yaxis.label.set_color(color) # label color
    c.color_right_yax = color_right_yax # 猴子添加一个实例方法
    return c
    
def _fdata_keep_n_lowfreq_pnts(fdata: Sequence, nPositive_freq_pnts_kept: int)->np.ndarray:
    """
    一个简单的频域高频成分截断 filter, 可以用于对任何数据序列的 smoothing (不需要是时域数据)
    fft(data) 后得到的 fdata 频率序列有两种情况:
    0 1 2 3 -4 -3 -2 -1
    0 1 2 3 -3 -2 -1
    fftshift(fdata) 后有两种情况:
    -4 -3 -2 -1 0 1 2 3
       -3 -2 -1 0 1 2 3
    其中 [1 2 3] 被称为正频率成分, 不包含 0 (DC), 以上两种情况中, fdata_keep_n_lowfreq_pnts 均为 3
    如果保留两个正频率成分, 那么上述两种情况下均得到 (其他成分设为 0):
    -2 -1 0 1 2
    如果保留 0 个正频率成分, 那么得到:
    0 
    如果保留所有的点那么, 两种情况下最终均会返回原始成分 (那奎斯特频率 -4 如果有, 会保留):
    -4 -3 -2 -1 0 1 2 3
       -3 -2 -1 0 1 2 3
    """
    assert isinstance(nPositive_freq_pnts_kept, int) and (nPositive_freq_pnts_kept >=0), "需要保留的点数是非负整数"
    nPositive_freq_pnts = int(len(fdata)/2-0.1) # 永远返回正确的 positive frequency components 数量, 见 docstring
    assert nPositive_freq_pnts_kept<=nPositive_freq_pnts, "正频率成分数量上限 ≈ signal 点数的一半!"
    sfdata = fftshift(fdata)
    nPositive_freq_pnts_thrown = nPositive_freq_pnts - nPositive_freq_pnts_kept
    if nPositive_freq_pnts_thrown: # when this number is 0, do nothing
        sfdata[-nPositive_freq_pnts_thrown:] = 0 # set positive high freq components to zero
    if nPositive_freq_pnts_thrown: # 如果根本没有丢弃正频率成分, 那么也不必要动负频率成分
        nNegative_freq_pnts_thrown = nPositive_freq_pnts_thrown
        if len(fdata)%2 == 0: 
            nNegative_freq_pnts_thrown += 1
        sfdata[:nNegative_freq_pnts_thrown] = 0 # set negative high freq components to zero
    fdata_filtered = ifftshift(sfdata)
    return fdata_filtered

def data_keep_n_fft_pnts(data: Sequence, nPnts: int)->np.ndarray:
    fdata = fft(data)
    fdata_filtered = _fdata_keep_n_lowfreq_pnts(fdata=fdata, nPositive_freq_pnts_kept=nPnts)
    return ifft(fdata_filtered).real # 注意只返回 real 部分, 这要求 data 本身是 real 的 (当然一般都是), 如果有复信号的特殊需求, 可以用 fdata_keep_n_lowfreq_pnts 再构造新的函数

def normalize_to_01(dataset: np.ndarray, user_min_max: Optional[Tuple[float, float]] = None):
    """
    自动用 dataset 的 min-max 将 dataset 归一到 0-1. 
    如果用户指定了 min-max tuple, 就用用户 min-max 做归一化 (不一定归一到 0-1)
    """
    if user_min_max:
        themin, themax = user_min_max
    else:
        themin, themax = dataset.min(), dataset.max()
    data_normed = (dataset - themin)/(themax - themin)
    return data_normed, (dataset.min(), dataset.max())

def _velo_dist_discretization(
        arr_velo_positive: np.ndarray, 
        mu: float=0, 
        sigma: float=174.8769038)->tuple: # 
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
    
    sigma 只不过是 sqrt(kB T/m), see Downes 2023
    320 K (Borowka) : 174.8769038
    417 K (A102 EIT) : 199.6298561
    388 K (A102 产生光) : 192.5632154
    """
    #v 这是上面的 bar 的面积的列表, 是概率, 不是概率密度(bar 的高度)
    arr_prob_positive = (norm.pdf(arr_velo_positive, mu, sigma)[:-1] # velo prob density val
              * (arr_velo_positive[1:] - arr_velo_positive[:-1])) # velo bin size
    arr_prob = np.hstack((np.flip(arr_prob_positive), arr_prob_positive))
    arr_velo = np.hstack((-np.flip(arr_velo_positive), arr_velo_positive))[1:-1]
    prob_tot = arr_prob.sum()
    return arr_prob, arr_velo, prob_tot

def velo_dist_discretization(
        arr_velo_positive: Sequence, 
        T: float,
        use_Rb85 = False
        )->tuple:
    """
    输入温度 T, 给出一维多普勒的 sigma
    """
    arr_velo_positive = np.array(arr_velo_positive)
    sigma = np.sqrt(kB*T/atomic_mass/(85 if use_Rb85 else 87))
    return _velo_dist_discretization(arr_velo_positive, sigma=sigma)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    data = np.random.randn(101)
    datafil = data_keep_n_fft_pnts(data, 20)
    plt.plot(data)
    plt.plot(datafil)
