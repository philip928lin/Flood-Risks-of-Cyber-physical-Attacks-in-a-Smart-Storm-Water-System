import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import plot_ss, gen_ref, set_sim, run_sim
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
rho_q = [1, 1, 1,  100, 100, 900, 250,     1,  10]
rho_r = [1, 3,   50, 30, 15, 120,  50, 20000, 10]
gw = 1
sw = 0.9
wc = 0.8
uya = float('inf')

strategy = "1N{}M{}CI{}yr".format(30, 990, 25)
eps = chi2.ppf(q=0.95**(1/1440), df=9, scale=sig_V**2) \
      - chi2.ppf(q=0.99**(1/30), df=8, scale=sig_V**2)

runoffs = cali_data[(25, 24)][["Sb"+str(i+1) for i in range(9)]].to_numpy().T
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
tc_q05, tc_p = gen_ref(sim_y_uc, ymax, return_tc=True)

ss, lqg, lqe = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss, lqe = run_sim(ss, lqg, lqe, W, Wn=M0, V=M0, Ya=None, R=Ref, ymax=ymax,
                  uc=uc, control=True, s1_o=None)
sim_y, sim_o = ss.get_ss_record()
# plot_ss(c=sim_y, unc=sim_y_uc, FDI_labels=[], wc=wc*100,
#         ylabel="Water level (%)", ty="Y%", ymax=ymax, sharey=True, seg=None)
# plot_ss(c=sim_o, unc=sim_o_uc, FDI_labels=[], wc=wc*100,
#         ylabel="Outflow", ty="O", ymax=ymax, sharey=True, seg=None)


do_max = (sim_o_uc.max(axis=0) - sim_o.max(axis=0))/100
area = np.array([19517.92, 630.0, 6838.89, 800.0, 750.0, 350.0, 1885.0, 3848.33, 1000.0])
Q_cms = do_max * area / 60

(yr, ci, ta) = (25, 0.99, 30)
strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)
with open(os.path.join(model_path, "OptFDI", "FDIN1_{}.pickle".format(strategy)), "rb") as f:
    all_Y_d, dY, Err, all_Fh_d, sim_y, sim_y_uc = pickle.load(f)
with open(os.path.join(model_path, "OptFDI", "MC_{}.pickle".format(strategy)), "rb") as f:
    sim_y_list, sim_o_list, flood_list, err_list = pickle.load(f)
# plot_ss(FDI_list=[all_Y_d], dY_list=dY, c=sim_y, unc=sim_y_uc, FDI_labels=[strategy],
#         ylabel="Water level (%) {}".format(strategy), ty="Y%", ymax=ymax, sharey=True, Fh=all_Fh_d,
#         wc=100)

#%%

##### Single attack
(yr, ci, ta) = (25, 0.99, 30)
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
tc_q5, tc_p = gen_ref(sim_y_uc, ymax, return_tc=True)

ss, lqg, lqe = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss, lqe = run_sim(ss, lqg, lqe, W, Wn=M0, V=M0, Ya=None, R=Ref, ymax=ymax,
                  uc=uc, control=True, s1_o=None)
sim_y, sim_o = ss.get_ss_record()
# plot_ss(c=sim_y, unc=sim_y_uc, FDI_labels=[], wc=wc*100,
#         ylabel="Water level (%)", ty="Y%", ymax=ymax, sharey=True, seg=None)
with open(os.path.join(model_path, "OptFDI", "Ya_{}.pickle".format(strategy)), "rb") as f:
    Ya_dict = pickle.load(f)

nT = ta
seg=None#[0, 980]
dY = []
Err = []
all_Y_d = np.empty(sim_y.shape)
all_Fh_d = np.empty(sim_y.shape)
for i, s in enumerate(["S{}".format(i+1) for i in range(9)]):
    Ya_d = Ya_dict[s]
    ss_a_d, lqg, lqe_a_d = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
    ss_a_d, lqe_a_d = run_sim(ss_a_d, lqg, lqe_a_d, W, Wn=M0, V=M0, Ya=Ya_d, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
    sim_y_a_d, sim_o_a_d = ss_a_d.get_ss_record()
    all_Y_d[:, i] = sim_y_a_d[:, i]
    Fh = ss_a_d.cal_Fh(ymax)
    all_Fh_d[:, i] = Fh[:, i]

    kd = np.abs(sim_y-sim_y_a_d).argmax(axis=0)
    dY.append([[(sim_y_a_d-sim_y)[kd[j],j] for j in range(9)]])
    err = lqe_a_d.eps[tc_p[i]+1:tc_p[i]+nT+1]
    Err.append(np.array(err).round(3))


##### Hybrid attack

dY_h = []
Err_h = []
all_Y_h = np.empty(sim_y.shape) * np.nan
all_Fh_h = np.empty(sim_y.shape) * np.nan
for i, s in enumerate(["S{}".format(i+1) for i in range(9)]):
    if s in ["S1", "S3", "S6", "S7"]:
        print(s)
        strategy = "{}U{}M{}CI{}yr".format(s, ta, int(ci*1000), yr)
        with open(os.path.join(model_path, "MOptFDI", "Ya_{}.pickle".format(strategy)), "rb") as f:
            Ya_dict = pickle.load(f)
        Ya_h = Ya_dict[s]
        ss_a_h, lqg, lqe_a_h = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
        ss_a_h, lqe_a_h = run_sim(ss_a_h, lqg, lqe_a_h, W, Wn=M0, V=M0, Ya=Ya_h, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
        sim_y_a_h, sim_o_a_h = ss_a_h.get_ss_record()
        all_Y_h[:, i] = sim_y_a_h[:, i]
        Fh = ss_a_h.cal_Fh(ymax)
        all_Fh_h[:, i] = Fh[:, i]
    
        kd = np.abs(sim_y-sim_y_a_h).argmax(axis=0)
        dY_h.append([[(sim_y_a_h-sim_y)[kd[j],j] for j in range(9)]])
        dY[i].append([(sim_y_a_h-sim_y)[kd[j],j] for j in range(9)])
        err = lqe_a_h.eps[tc_p[i]+1:tc_p[i]+nT+1]
        Err_h.append(np.array(err).round(3))
    else:
        dY_h.append(None)
        
#%% Fig 7
from util import plot_ss
ref = Ref.T/ymax*100
plot_ss(FDI_list=[all_Y_d, all_Y_h], dY_list=dY, c=sim_y, unc=sim_y_uc,
        FDI_labels=["$FDI_S$","$FDI_M$"],
        ylabel="Water level (%)", ty="Y%", ymax=ymax, sharey=True,
        seg=seg, Fh=all_Fh_d, wc=100, t_tuple=(tc_p, nT), ref=ref)

#%% Fig 8
from util import plot_ss
ref = Ref.T/ymax*100
prob_Wn = np.array(flood_list).mean(axis=0)
plot_ss(FDI_list=[all_Y_d], dY_list=[], c=sim_y, unc=sim_y_uc,
        FDI_labels=["$FDI_S$"],
        ylabel="Water level (%) / $P^s_W$", ty="Y%", ymax=ymax, sharey=True,
        seg=seg, Fh=all_Fh_d, wc=100, t_tuple=(tc_p, nT),
        stoch_data_list=sim_y_list, prob=prob_Wn, ref=ref)






