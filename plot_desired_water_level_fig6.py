# =============================================================================
# Plot figure desired water level (figure 4)
# =============================================================================
import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import gen_ref, set_sim, run_sim
from lqg_controller import gen_noises
from scipy.stats import chi2

seed = 3
rngen = np.random.default_rng(seed)
with open(os.path.join(model_path, "Inputs.pickle"), "rb") as f:
    ss_model, cali_data = pickle.load(f)
uc = [0.043063, 1.018975, 0.093868, 1.805496, 1.925863, 4.126848, 0.766258, 0.046447, 1.444397]
ymax = [240.0, 60.0, 180.0, 40.0, 40.0, 91.44, 120.0, 180.0, 110.0]  # [cm]

sig_V = 0.25    # [cm] from WRR paper 2.5 mm
num = 1000
rho_q = [1, 1, 1,  100, 100, 900, 250,     1, 10]
rho_r = [1, 3,   50, 30, 15, 120,  50, 20000, 10]
gw = 1
sw = 0.9
wc = 0.8
uya = float('inf')


(yr, ci, ta) = (25,0.99,30)
strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)
eps = chi2.ppf(q=0.95**(1/1440), df=9, scale=sig_V**2) \
      - chi2.ppf(q=ci**(1/ta), df=8, scale=sig_V**2)
      
runoffs = cali_data[(yr, 24)][["Sb"+str(i+1) for i in range(9)]].to_numpy().T
M0 = np.zeros(runoffs.shape)
W = runoffs
V_dict, Wn_dict, std_w = gen_noises(W, sv=sig_V, sw=sw, m=1, alpha=1,
                                    dist=("normal","lognorm"), size=num, rngen=rngen)
sig_W = std_w
V = M0; Wn = M0

ss_uc, lqg, lqe_uc = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss_uc, lqe_uc = run_sim(ss_uc, lqg, lqe_uc, W, Wn=M0, V=M0, Ya=None, R=M0,
                        ymax=ymax, uc=uc, control=False, s1_o=None)
sim_y_uc, sim_o_uc = ss_uc.get_ss_record()

Ref = gen_ref(sim_y_uc, ymax, wc=wc)
tc_q10, tc_max = gen_ref(sim_y_uc, ymax, return_tc=True)

Ref=Ref.T
#%%
s=5
fig, ax = plt.subplots(figsize=(4,3))

ax.axvline(tc_q10[s], ls="--", lw=0.8, color="lightgrey", zorder=0)
ax.axvline(tc_max[s], ls="--", lw=0.8, color="lightgrey", zorder=0)
ax.axhline(80, ls="--", lw=0.8, color="lightgrey", zorder=0)
ax.plot(sim_y_uc[:,s]/ymax[s]*100, label="Uncontrolled", color="grey", ls=":", zorder=1)
ax.plot(Ref[:,s]/ymax[s]*100, label="$\mathscr{R}$", color="k", zorder=2)
ax.scatter(tc_q10[s], sim_y_uc[tc_q10[s],s]/ymax[s]*100, label="$t_{5\%}$",
           marker="x", color="k", zorder=3)
ax.scatter(tc_max[s], sim_y_uc[tc_max[s],s]/ymax[s]*100, label="$t_{p}$",
           marker="*", color="k", zorder=4)
ax.set_ylim([0,100])
ax.set_xlim([0,1440])
ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.02, 1.15), ncol=4, handletextpad=0.2 )

l = 60*5
ax.arrow(x=tc_max[s], y=85, dx=l, dy=0,
         length_includes_head=True, head_width=0.03, zorder=10)
ax.arrow(x=tc_max[s]+l, y=85, dx=-l, dy=0,
         length_includes_head=True, head_width=0.03, zorder=10)
ax.text(0.52, 0.88, '5 hours', transform=ax.transAxes, fontsize=10)
ax.set_ylabel("Water level (%)", fontsize=12)
ax.set_xlabel("Time (minute)", fontsize=12)









