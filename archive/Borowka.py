#%%
from lablib102.qfuns import steadystateMWM, make_decayRateDict
import numpy as np
import matplotlib.pyplot as plt
from arc import Rubidium85
from p_tqdm import p_map
from archive.helper import velo_dist_discretization
from functools import partial


atom = Rubidium85()
lstJmanifs = [
    (5,0,1/2),
    (5,1,3/2),
    (55,2,5/2),
    (54,3,7/2),
    (5,2,5/2),
    ]
for jmanif1, jmanif2 in zip(lstJmanifs[0:-1], lstJmanifs[1:]):
    print(atom.getTransitionWavelength(*jmanif1, *jmanif2)*1e9)

#%%

arr_velo_p = np.hstack((np.arange(0,10,0.5),
                     np.arange(10,50,5),
                     np.arange(50,200,20),
                     np.arange(200,600,100),
                     ))

# arr_velo_p = np.arange(0,1000,0.01)
arr_prob, arr_velo, prob_tot = velo_dist_discretization(arr_velo_p)
# print(prob_tot)

#%%
### params needed for outer_decay_rate_dict
term_list = ["5S1/2", "5P3/2", "55D5/2", "54F7/2", "5D5/2"]
nonNNchannels = ["41"]
transit_rate = 50
unphysicalChannels = [("10", transit_rate), ("20", transit_rate), ("30", transit_rate),("40", transit_rate)]
temperature = 320

def get_rhos_given_deltaps(
        velo: float, 
        arr_deltap: np.ndarray
        )->list:
    couple_list = [8,22,8,17]
    rho_list = []
    for detp in arr_deltap:
        det_list = [detp - (velo/7.802414762021273e-07)*1e-6, # (velo... 前的正负号表示 beam 的方向
                   -16   + (velo/4.799353650243717e-07)*1e-6, # 正是红失谐
                    16   + (velo/0.021539232660859896)*1e-6,
                    0    + (velo/1.2579169675550843e-06)*1e-6]
        # rho_list.append("blah")
        rho_list.append(steadystateMWM(
            term_list, couple_list, det_list, nonNNchannels,
            unphysicalChannels, temperature=temperature, outer_decay_rate_dict=outer_decay_rate_dict))
    return rho_list

outer_decay_rate_dict = make_decayRateDict(term_list, nonNNchannels,unphysicalChannels, temperature=temperature)

arr_deltap = np.linspace(-40,40,101)
rho_llist = p_map(
    partial(get_rhos_given_deltaps, arr_deltap = arr_deltap), arr_velo)



#%%
arrRho01imag = np.array([[rho[1,0].imag for rho in rhoList] for rhoList in rho_llist])
arrrho41sq = np.array([[abs(rho[4,1])**2 for rho in rhoList] for rhoList in rho_llist])
EITdoppler = (arrRho01imag*(arr_prob.reshape((-1,1)))).sum(axis=0)
gendoppler = (arrrho41sq*(arr_prob.reshape((-1,1)))).sum(axis=0)


#%%
arr_deltap = np.linspace(-40,40,101)
fig, ax = plt.subplots()
axx = ax.twinx()
ax.plot(-arr_deltap, EITdoppler)
ax.axvline(5,c="r", ls = ":")
# ax.set_ylim(0,0.02)
axx.plot(-arr_deltap, gendoppler, "r")
axx.tick_params(colors="r")
