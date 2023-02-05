import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

folder = "OptFDI-F"
with open(os.path.join(model_path, folder, "Prob.pickle"), "rb") as f:
    (df_p_Wn, df_p_fdi, df_p_ce, df_p_c) = pickle.load(f)

folder = "OptFDI-F_noAttack"
with open(os.path.join(model_path, folder, "Prob.pickle"), "rb") as f:
    (_, _, _, df_p_noA) = pickle.load(f)

#%%
yr_list = [200,100,50,25,10,5,2,1] #[1, 2, 5, 10, 25, 50, 100, 200]
ta_list = [5,10,15,20,25,30,35,40,45,50,55,60]

fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True,
                         figsize=(5.5,4))
plt.tight_layout(h_pad=1, w_pad=0)
axes = axes.flatten()
cbar_ax = fig.add_axes([0.97, 0.1, 0.02, 0.8])
cmap = sns.color_palette("mako_r", as_cmap=True)

ponds = ['Pond 1', 'Pond 2', 'Pond 3', 'Pond 4', 'Pond 5', 'Pond 6', 'Pond 7', 'Pond 8', 'Pond 9']

df_proportion = df_p_noA.copy()
df_proportion = df_p_noA.loc[:, ponds]/df_p_c.loc[:, ponds]

for k, v in enumerate([8,4,2,7,5,3,9,6,1]):
    prob = np.fliplr(df_p_c["Pond {}".format(v)].values.reshape((12, 8), order="F"))
    prob[prob==0] = np.nan
    prob = prob.T
    ax = sns.heatmap(prob, xticklabels=ta_list, yticklabels=yr_list, square=True,
                      cmap=cmap, vmin=0.00001, vmax=1, ax=axes[k], cbar=v ==3,
                      cbar_ax=cbar_ax)
    ax.set_facecolor('lightgrey')
    ax.set_title("Pond "+str(v), fontsize=9)

    proportion = np.fliplr(df_proportion["Pond {}".format(v)].values.reshape((12, 8), order="F"))
    locs = np.where(proportion >= 0.9)
    for xi, yi in zip(locs[0], locs[1]):
        ax.add_patch(Rectangle((xi, yi), 1, 1, ec='red', fc='none', lw=0.8, hatch='//', alpha=0.5))
axes[5].annotate("Flooding risks given FDI", (1.4, -0.3), rotation=270, xycoords='axes fraction', fontsize=11)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.xlabel("\n$T^a$ (minute)", fontsize=11)
plt.ylabel("Return period (year)", fontsize=11, labelpad=10)
plt.show()


# cbar = fig.colorbar(ax, cax=cb_ax)
# cbar.set_ticks([0,0.5,1])
# cbar.set_ticklabels([0,0.5,1])




