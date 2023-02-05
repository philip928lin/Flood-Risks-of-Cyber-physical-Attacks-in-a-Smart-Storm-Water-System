import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lqg_controller import cal_prob_fdi
folder = "OptFDI-F"

df_p_Wn = pd.DataFrame()
df_p_fdi = pd.DataFrame()
df_p_ce = pd.DataFrame()
df_p_c = pd.DataFrame()

yr_list = [1, 2, 5, 10, 25, 50, 100, 200]
ci_list = [0.99] 
ta_list = [5,10,15,20,25,30,35,40,45,50,55,60]
combinations = [(yr, ci, ta) for yr in yr_list for ci in ci_list for ta in ta_list]
for com in combinations:
    (yr, ci, ta) = com
    strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)

    with open(os.path.join(model_path, folder, "MC_{}.pickle".format(strategy)), "rb") as f:
        sim_y_list, sim_o_list, flood_list, err_list = pickle.load(f)
    with open(os.path.join(model_path, folder, "FDIN1_{}.pickle".format(strategy)), "rb") as f:
        all_Y_d, dY, Err, all_Fh_d, sim_y, sim_y_uc = pickle.load(f)

    prob_Wn = np.array(flood_list).mean(axis=0)
    prob_fdi = np.array([cal_prob_fdi(err, q_o=0.95, dof_o=9, dof_a=8) for err in Err])
    prob_ce = [p_f * p_w for p_f, p_w in zip(prob_fdi, prob_Wn)]
    prob_c = [p_f * p_w / yr for p_f, p_w in zip(prob_fdi, prob_Wn)]

    df_p_Wn[strategy] = prob_Wn
    df_p_fdi[strategy] = prob_fdi
    df_p_ce[strategy] = prob_ce
    df_p_c[strategy] = prob_c

def convert_df(df):
    df.index = ["Pond {}".format(i+1) for i in range(9)]
    df = df.T
    df["yr"] = [i[0] for i in combinations]
    df["ci"] = [i[1] for i in combinations]
    df["ta"] = [i[2] for i in combinations]
    return df
df_p_Wn, df_p_fdi, df_p_ce, df_p_c = [convert_df(df) for df in [df_p_Wn, df_p_fdi, df_p_ce, df_p_c]]


with open(os.path.join(model_path, folder, "Prob.pickle"), "wb") as f:
    pickle.dump((df_p_Wn, df_p_fdi, df_p_ce, df_p_c), f)

# #%%
# fig, ax = plt.subplots()
# df = df_p_c[df_p_c["yr"] == 25]
# x = list(df["ta"])
# for i in range(9):
#     s = "Pond {}".format(i+1)
#     ax.plot(x, df[s], label=s)
# ax.axhline(1/25, color="k", ls="--", lw=0.5)
# ax.set_ylabel("$P_c$ [25yr, $p$=0.95, $p^a=0.99$]")
# ax.set_xlabel("$T^a$")
# ax.legend(loc="lower right", fontsize=8)

# #%%
# fig, ax = plt.subplots()
# df = df_p_c[df_p_c["ta"] == 30]
# x = [0,1,2,3,4,5,6,7]#list(df["yr"])
# for i in range(9):
#     s = "Pond {}".format(i+1)
#     ax.plot(x, df[s], label=s)
# #ax.axhline(1/25, color="k", ls="--", lw=0.5)
# ax.set_xticks(x)
# ax.set_xticklabels(list(df["yr"]))
# ax.set_ylabel("$P_c$ [$T^a$=30, $p$=0.95, $p^a=0.99$]")
# ax.set_xlabel("Return period of 24 hour storm")
# ax.legend(loc="upper right", fontsize=8)

# #%%
# fig, ax = plt.subplots()
# df = df_p_ce[df_p_ce["ta"] == 30]
# x = [0,1,2,3,4,5,6,7]#list(df["yr"])
# for i in range(9):
#     s = "Pond {}".format(i+1)
#     ax.plot(x, df[s], label=s)
# #ax.axhline(1/25, color="k", ls="--", lw=0.5)
# ax.set_xticks(x)
# ax.set_xticklabels(list(df["yr"]))
# ax.set_ylabel("$P_{FDI} * P_W$ [$T^a$=30, $p$=0.95, $p^a=0.99$]")
# ax.set_xlabel("Return period of 24 hour storm")
# ax.legend(loc="upper right", fontsize=8)