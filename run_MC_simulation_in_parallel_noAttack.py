import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code", "Final")
os.chdir(module_path)
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from util import plot_ss, gen_ref, set_sim, run_sim
from lqg_controller import gen_noises, cal_prob_fdi
from scipy.stats import chi2
import time

folder = "OptFDI-F_noAttack"

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

yr_list = [25, 10, 50, 5, 100, 200, 1, 2]
ci_list = [0.99]
ta_list = [5,10,15,20,25,30,35,40, 45, 50, 55, 60]
combinations = [(yr, ci, ta) for yr in yr_list for ci in ci_list for ta in ta_list]
ss = rngen.bit_generator._seed_seq
rng_seeds = ss.spawn(len(combinations))
rngen_list = [np.random.default_rng(seed) for seed in rng_seeds]

def simFunc(com, rngen):
    (yr, ci, ta) = com
    strategy = "1N{}M{}CI{}yr".format(ta, int(ci*1000), yr)
    eps = chi2.ppf(q=0.95**(1/1440), df=9, scale=sig_V**2) \
          - chi2.ppf(q=ci**(1/ta), df=8, scale=sig_V**2)

    if os.path.exists(os.path.join(model_path, folder, "MC_{}.pickle".format(strategy))):
        print(strategy, " exists.")
        return None

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
    # with open(os.path.join(model_path, folder, "Ya_{}.pickle".format(strategy)), "rb") as f:
    #     Ya_dict = pickle.load(f)

    nT = ta
    dY = []
    Err = []
    all_Y_d = np.empty(sim_y.shape)
    all_Fh_d = np.empty(sim_y.shape)
    for i, s in enumerate(["S{}".format(i+1) for i in range(9)]):
        Ya_d = None#Ya_dict[s]
        ss_a_d, lqg, lqe_a_d = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
        ss_a_d, lqe_a_d = run_sim(ss_a_d, lqg, lqe_a_d, W, Wn=M0, V=M0, Ya=Ya_d,
                                  R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
        sim_y_a_d, sim_o_a_d = ss_a_d.get_ss_record()
        all_Y_d[:, i] = sim_y_a_d[:, i]
        Fh = ss_a_d.cal_Fh(ymax)
        all_Fh_d[:, i] = Fh[:, i]

        kd = np.abs(sim_y-sim_y_a_d).argmax(axis=0)
        dY.append([[(sim_y_a_d-sim_y)[kd[j],j] for j in range(9)]])
        err = lqe_a_d.eps[tc_p[i]+1:tc_p[i]+nT+1]
        Err.append(np.array(err).round(3))

    with open(os.path.join(model_path, folder, "FDIN1_{}.pickle".format(strategy)), "wb") as f:
        pickle.dump([all_Y_d, dY, Err, all_Fh_d, sim_y, sim_y_uc], f)

    sim_y_list = []
    sim_o_list = []
    flood_list = []
    err_list = []
    tic = time.perf_counter()
    for n in tqdm(range(num), desc=strategy):
        Err_n = []
        all_Y_n = np.empty(sim_y.shape)
        all_O_n = np.empty(sim_y.shape)
        # all_flood_n = []
        flood_n = []
        for i, s in enumerate(["S{}".format(i+1) for i in range(9)]):
            Ya_d = None#Ya_dict[s]
            ss_n, lqg, lqe_n = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
            ss_n, lqe_n = run_sim(ss_n, lqg, lqe_n, W, Wn=Wn_dict[n], V=V_dict[n],
                                  Ya=Ya_d, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
            sim_y_n, sim_o_n = ss_n.get_ss_record()
            all_Y_n[:, i] = sim_y_n[:, i]
            all_O_n[:, i] = sim_o_n[:, i]

            flood = [1 if y > ymax[j] else 0 for j, y in enumerate(sim_y_n.max(axis=0))]
            err = lqe_n.eps[tc_p[i]+1:tc_p[i]+nT+1]
            #all_flood_n.append(flood)
            flood_n.append(flood[i])
            Err_n.append(np.array(err).round(3))

        sim_y_list.append(all_Y_n)
        sim_o_list.append(all_O_n)
        err_list.append(Err_n)
        flood_list.append(flood_n)
    toc = time.perf_counter()
    print(f"Done in {toc - tic:0.4f} seconds")

    with open(os.path.join(model_path, folder, "MC_{}.pickle".format(strategy)), "wb") as f:
        pickle.dump([sim_y_list, sim_o_list, flood_list, err_list], f)


res = Parallel(n_jobs=-1, verbose=10)( delayed(simFunc)(com, rngen) for com, rngen in zip(combinations, rngen_list) )