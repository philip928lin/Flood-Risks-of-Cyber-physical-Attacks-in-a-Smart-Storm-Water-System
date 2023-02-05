import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from util import gen_ref, set_sim
from lqg_controller import  gen_Wn
from scipy.stats import chi2



# Gen sig_W_list
seed = 3
rngen = np.random.default_rng(seed)
with open(os.path.join(model_path, "Inputs.pickle"), "rb") as f:
    ss_model, cali_data = pickle.load(f)
rho_q = [1, 1, 1,  100, 100, 900, 250,     1,  10]
rho_r = [1, 3,   50, 30, 15, 120,  50, 20000, 10]
gw = 1

yr_list = [1, 2, 5, 10, 25, 50, 100, 200]
intival = 0.05
sig_v_list = list(np.arange(0.15, 0.9 + intival, intival).round(4))  # sig_V = 0.25    # [cm] from WRR paper 2.5 mm

df_L = pd.DataFrame()
sig_dict = {}
for yr in tqdm(yr_list):
    runoffs = cali_data[(yr, 24)][["Sb"+str(i+1) for i in range(9)]].to_numpy().T
    W = runoffs
    _, sig_W = gen_Wn(W=W, s=0.9, m=1, alpha=1, dist="lognorm", size=1000, rngen=rngen)
    for sig_v in sig_v_list:
        ss_uc, lqg, lqe_uc = set_sim(ss_model, sig_v, sig_W, rho_q, rho_r, gw=gw)

        L = lqg.L.todense()
        df_L[(yr, sig_v)] = np.array(np.max(L, axis=0)).flatten()
        sig_dict[(yr, sig_v)] = (sig_v, sig_W)

df_L = df_L.T

# std_W_dict = {}
# for yr in yr_list:
#     runoffs = cali_data[(yr, 24)][["Sb"+str(i+1) for i in range(9)]].to_numpy().T
#     W = runoffs
#     _, std_w = gen_Wn(W=W, s=0.9, m=1, alpha=1, dist="lognorm", size=1000, rngen=rngen)
#     std_W_dict[yr] = std_w

#%%
# =============================================================================
# Plot Kalman gain
# =============================================================================
fig, ax = plt.subplots(figsize=(5,5))
ax.set_ylim([0,1])
ax.set_xlim([0,1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
upperLimit = 100
lowerLimit = 0.3
max_value = 1
w_scale = 0.85
slope = (max_value - lowerLimit) / max_value

# Conduits
locs = [(0.8,0.15), (0.85,0.4), (0.65,0.65), (0.7,0.85), (0.5,0.4), (0.4,0.25),
        (0.25,0.5), (0.15,0.8), (0.2,0.2)]
locs = [(l[0]*0.85 , l[1]*0.85) for l in locs]
locss = [(l[0]/0.8-0.033 , l[1]/0.8-0.06) for l in locs]
links = [(1,0),(2,0),(4,0),(5,0),(8,0),(7,6),(6,5),(3,2)]
for link in links:
    ax.plot([locss[link[0]][0], locss[link[1]][0]],
            [locss[link[0]][1], locss[link[1]][1]], color="grey")

# Gain L
my_cmap = plt.get_cmap("summer_r")
#my_cmap = plt.get_cmap("Wistia")
rescale = lambda y: y / len(sig_v_list)
for n in range(9):
    axx = fig.add_axes([locs[n][0] , locs[n][1] , 0.15, 0.15], polar=True)
    axx.set_axis_off()
    axx.set_ylim([0,max_value+lowerLimit])

    axx.annotate(str(n+1),
                 xy=(0.5, 0.5),  # theta, radius
                 xytext=(0.5, 0.5),    # fraction, fraction
                 textcoords='axes fraction',
                 horizontalalignment='center',
                 verticalalignment='center')

    for i, sig_v in enumerate(sig_v_list):
        mask = [(yr, sig_v) for yr in yr_list]
    # for i, yr in enumerate(yr_list):
    #     mask = [(yr, sig_v) for sig_v in sig_v_list]
        data = df_L.loc[mask, n].values

        heights = slope * data + lowerLimit
        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2*np.pi / len(data)

        # Compute the angle each bar is centered on:
        indexes = list(range(1, len(data)+1))
        angles = [(element+1) * width + width * (1-w_scale**i)/2*0.8 for element in indexes]

        if i == 0:
            bars = axx.bar(
                x=angles,
                height=[max_value]*len(data),
                width=width,
                bottom=0,
                linewidth=0.5,
                facecolor="lightgrey",
                edgecolor=(0,0,0,0))
            bars = axx.bar(
                x=angles,
                height=[max_value]*len(data),
                width=width,
                bottom=lowerLimit,
                linewidth=0.5,
                facecolor="lightgrey",
                edgecolor=(0,0,0,0))

        # Draw bars
        if sig_v == 0.25: # what we used in the model
            edgecolor=(0,0,0,1)
        else:
            edgecolor=(0,0,0,0)
        bars = axx.bar(
            x=angles,
            height=heights,
            width=width * 0.8 * w_scale**i,
            bottom=lowerLimit,
            linewidth=0.5,
            facecolor=my_cmap(rescale(i)),
            edgecolor=edgecolor) # (R,G,B alpha)
