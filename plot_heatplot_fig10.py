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
yr_list = [1, 2, 5, 10, 25, 50, 100, 200]

ta = 35
df_p_c_noTa = df_p_c[df_p_c["ta"]==ta]
df_p_noA_noTa = df_p_noA[df_p_noA["ta"]==ta]


fig, axes = plt.subplots(figsize=(3,3))
cbar_ax = fig.add_axes([0.97, 0.1, 0.02, 0.8])
cmap = sns.color_palette("mako_r", as_cmap=True)
cmap = sns.color_palette("PuBu", as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
cmap = sns.color_palette("GnBu", as_cmap=True)

ponds = ['Pond 1', 'Pond 2', 'Pond 3', 'Pond 4', 'Pond 5', 'Pond 6', 'Pond 7', 'Pond 8', 'Pond 9']

df_proportion = df_p_noA_noTa.copy()
df_proportion = df_p_noA_noTa.loc[:, ponds]/df_p_c_noTa.loc[:, ponds]

pondud = ['Pond 8', 'Pond 2', 'Pond 4', 'Pond 5', 'Pond 7', 'Pond 3', 'Pond 6', 'Pond 9', 'Pond 1']
pondud = ['Pond 1', 'Pond 3', 'Pond 8', 'Pond 7', 'Pond 2', 'Pond 9', 'Pond 6', 'Pond 4', 'Pond 5']
cpr = df_p_c_noTa.loc[:, pondud].values.T
cpr[cpr==0] = np.nan
ax = sns.heatmap(cpr, xticklabels=yr_list, yticklabels=pondud, square=False,
                  cmap=cmap, vmin=0.00001, vmax=1, ax=axes, cbar=True,
                  cbar_ax=cbar_ax)
ax.set_facecolor('lightgrey')

proportion = df_proportion.loc[:, pondud].values.T
locs = np.where(proportion >= 0.9)
for xi, yi in zip(locs[1], locs[0]):
    print(xi)
    ax.add_patch(Rectangle((xi, yi), 1, 1, ec='red', fc='none', lw=0.8, hatch='//', alpha=0.5))

ax.add_patch(Rectangle((0, 0), 8, 3, ec='blue', fc='none', lw=1.5, ls="--"))

ax.annotate("Conditional flood risks given FDI", (1.26, -0.02), rotation=270, xycoords='axes fraction', fontsize=11)
ax.set_xlabel("Return period (year)", fontsize=11)




