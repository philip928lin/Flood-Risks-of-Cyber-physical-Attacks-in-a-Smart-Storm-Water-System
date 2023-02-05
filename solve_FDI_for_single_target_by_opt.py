import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import pickle
import time
import numpy as np
from util import plot_ss, gen_ref, set_sim, run_sim
from gp_model import GPModel, collect_res
from lqg_controller import gen_noises
from scipy.stats import chi2


folder = "OptFDI-F"
err_log = []

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

yr_list = [1, 2, 5, 10, 25, 50, 100, 200]
ci_list = [0.99]
ta_list = [5,10,15,20,25,30,35,40,45,50,55,60]

# yr = 25
# ci = 0.99
# ta = 30

try:
    with open(os.path.join(model_path, folder, "All_Ya.pickle"), "rb") as f:
        all_Ya_dict = pickle.load(f)
except:
    all_Ya_dict = {}
# try:
#     with open(os.path.join(model_path, folder, "All_Pkl.pickle"), "rb") as f:
#         all_pkl_dict = pickle.load(f)
# except:
#     all_pkl_dict = {}

combinations = [(yr, ci, ta) for yr in yr_list for ci in ci_list for ta in ta_list]
for count, com in enumerate(combinations):
    (yr, ci, ta) = com
    strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)
    print("Attack strategy: ", strategy)

    # if strategy in list(all_Ya_dict.keys()):
    #     continue
    if os.path.exists(os.path.join(model_path, folder, "Ya_{}.pickle".format(strategy))):
        print(strategy, " exists.")
        continue

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
    tc_q05, tc_p = gen_ref(sim_y_uc, ymax, return_tc=True)

    ss, lqg, lqe = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
    ss, lqe = run_sim(ss, lqg, lqe, W, Wn=M0, V=M0, Ya=None, R=Ref, ymax=ymax,
                      uc=uc, control=True, s1_o=None)
    sim_y, sim_o = ss.get_ss_record()

    # plot_ss(c=sim_y, unc=sim_y_uc, FDI_labels=[], wc=wc*100,
    #         ylabel="Water level (%)", ty="Y%", ymax=ymax, sharey=True, seg=None)

    ##%%
    ### Setting
    nT = ta
    dT = 5
    oT = 1 # must >= 1
    maxTime = 20*60 # 20 min
    segs = int(nT / dT) + (nT % dT > 0)
    tic = time.perf_counter()

    all_gpm_dict = {}
    pkl_dict = {}
    Ya_dict = {}
    gpm_dict = {}
    for s in range(9):
        target = "S{}".format(s+1)
        print("{}-{}".format(strategy, target))
        uYa = [0]*9
        uYa[s] = uya
        noF = ["S{}".format(i+1) for i in range(9) if i != s]
        Ya = np.zeros(W.shape) * np.nan
        pkl_dict[target] = []
        gpm_dict[target] = []
        for i in range(segs):
            name = "{}_{}_{}-{}".format(strategy, target, segs, i+1)
            iniT = tc_p[s] + i*dT
            try:
                if i == 0:
                    ss_a = ss; lqe_a = lqe
                gpm = GPModel(runoffs, lqg, lqe_a, ss_a, Ref, uc=uc, name=name)
                gpm.set_gp(iniT, dT+oT, uYa=uYa)
                gpm.addObj_maxY(target)
                gpm.addC_Err(eps=eps)
                #gpm.addC_noflood(noF, tol=1e-5)
                gpm.optimize_gp(wd=os.path.join(model_path, folder),
                                TimeLimit=maxTime, log=False)
                if gpm.status == 4: raise ValueError("Infeasible or unbounded")
            except:
                if i == 0:
                    ss_a = ss; lqe_a = lqe
                gpm = GPModel(runoffs, lqg, lqe_a, ss_a, Ref, uc=uc, name=name)
                gpm.set_gp(iniT, dT+oT, uYa=uYa)
                gpm.addObj_maxY(target)
                gpm.addC_Err(eps=eps)
                #gpm.addC_noflood(noF, tol=1e-5)
                gpm.optimize_gp(wd=os.path.join(model_path, folder),
                                TimeLimit=maxTime, DualReductions=0, log=False)
            Ya_ = gpm.records["Ya"].T
            Ya[:, iniT:iniT+dT] = Ya_[:, iniT:iniT+dT] # Avoid the overlap part
            ss_a, lqg, lqe_a = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
            ss_a, lqe_a = run_sim(ss_a, lqg, lqe_a, W, Wn=M0, V=M0, Ya=Ya, R=Ref,
                                  ymax=ymax, uc=uc, control=True, s1_o=None)
            gpm_dict[target].append(gpm)
            pkl_dict[target].append(gpm.get_pickable_dict())
            #gpm.pickle()
        Ya_dict[target] = Ya
    #all_gpm_dict[strategy] = gpm_dict
    #all_pkl_dict[strategy] = pkl_dict
    all_Ya_dict[strategy] = Ya_dict
    with open(os.path.join(model_path, folder, "Ya_{}.pickle".format(strategy)), "wb") as f:
        pickle.dump(Ya_dict, f)
    with open(os.path.join(model_path, folder, "Pkl_{}.pickle".format(strategy)), "wb") as f:
        pickle.dump(pkl_dict, f)
    # with open(os.path.join(model_path, folder, "All_Ya.pickle"), "wb") as f:
    #     pickle.dump(all_Ya_dict, f)
    # with open(os.path.join(model_path, folder, "All_Pkl.pickle"), "wb") as f:
    #     pickle.dump(all_pkl_dict, f)

    toc = time.perf_counter()
    print(strategy, f" done in {toc - tic:0.4f} seconds")
with open(os.path.join(model_path, folder, "All_Ya.pickle"), "wb") as f:
    pickle.dump(all_Ya_dict, f)
# with open(os.path.join(model_path, folder, "All_Pkl.pickle"), "wb") as f:
#     pickle.dump(all_pkl_dict, f)
#%%

yr_list = [1, 2, 5, 10, 25, 50, 100, 200] # [25]#
ci_list = [0.99]
ta_list = [5,10,15,20,25,30,35,40,45,50,55,60]
combinations = [(yr, ci, ta) for yr in yr_list for ci in ci_list for ta in ta_list]

N = []; strategies = []
with open(os.path.join(model_path, folder, "1N99_OptRes.txt"), "w") as file:
    for com in combinations:
        (yr, ci, ta) = com
        strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)
        with open(os.path.join(model_path, folder, "Pkl_{}.pickle".format(strategy)), "rb") as f:
            pkl_dict = pickle.load(f)
        print(com)
        file.write(str(com))
        file.write("\n")
        collect_res(pkl_dict, f=file, n=N, strategies=strategies, show=True)
#%%
from util import plot_ss
seg=[0, 980]
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
    Err.append(np.array(err).round(2))

plot_ss(FDI_list=[all_Y_d], dY_list=dY, c=sim_y, unc=sim_y_uc, FDI_labels=[strategy],
        ylabel="Water level (%)", ty="Y%", ymax=ymax, sharey=True, seg=seg, Fh=all_Fh_d,
        wc=100)

#%%
def print_res(pkl_dict):
    for k, v in pkl_dict.items():
        print("{}: {} [{}] \t Gap: {}%  [{}]".format(
            k, [vv["status"] for vv in v],
            round(sum([vv["bestObj"] if vv["status"] != 3 else 0 for vv in v]), 2),
            [round(vv["gap"],4) if vv["status"] != 3 else vv["gap"] for vv in v],
            [round(vv["runtime"], 2) if vv["status"] != 3 else vv["runtime"] for vv in v]))
    print("\n")

def write_res(f, pkl_dict):
    for k, v in pkl_dict.items():
        f.write("{}: {} \t Gap: {}%".format(
            k, [vv["status"] for vv in v],
            [round(vv["gap"],4) if vv["status"] != 3 else vv["gap"] for vv in v]))
        f.write("\n")
    f.write("\n")

def find_infeasible_or_time_out(pkl_dict, n=None):
    if n is None:
        n = []
    for k, v in pkl_dict.items():
        if None in [round(vv["gap"],4) if vv["status"] != 3 else vv["gap"] for vv in v]:
            n.append("{}: {} \t Gap: {}%".format(
                k, [vv["status"] for vv in v],
                [round(vv["gap"],4) if vv["status"] != 3 else vv["gap"] for vv in v]))
    return n