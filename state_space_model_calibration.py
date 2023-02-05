import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = "1" # conda fix for pyswmm
prj_path = r""
module_path = os.path.join(prj_path, "Code")
swmm_model_path = os.path.join(prj_path, "Model", "Cyber_NancyRun_1min.inp")
cali_data_path = os.path.join(prj_path, "Model", "Cali_data")
os.chdir(module_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyswmm import Simulation
from util import (swmm_info, swmm_get_object, swmm_get_links,
                  swmm_get_inp_sections)
from lqg_controller import SS, SWMM2SSModel
from local_min_search import local_min_search
    
##### Form SWMM inp file
swmm_setting = swmm_get_inp_sections(swmm_model_path)
###### Load SWMM setting
with Simulation(swmm_model_path) as sim:
    swmm_info(sim)
    storages, junctions, orifices, conduits, subcatchs, raingauges = swmm_get_object(sim)
    links, links_sb = swmm_get_links(sim)
##### SS model settings
r2 = list(swmm_setting["XSECTIONS"]["Geom2"])[-9:]
nodes = ["S2", "J2", "S4", "J4", "S3", "J3", "S5", "J5", "S8", "J8", "S7",
         "J7", "S9", "J9", "S6", "J6", "S1"]
# Area [m2]
storage_areas = {"S"+str(k+1): v for k, v in enumerate(swmm_setting["STORAGE"]["CNPc"])}
areas = np.array(swmm_setting["STORAGE"]["CNPc"]).reshape((-1,1))
Y0 = np.array([0]*9).reshape((-1,1))
##### Load cali data
cali_data = {}
for dur_hr in [24]:
    for yr in [2, 25]:
        file_path = os.path.join(cali_data_path,
                                 "{}yr_{}hr_{}mins.csv".format(yr, dur_hr, 1))
        cali_data[(yr, dur_hr)] = pd.read_csv(file_path)
        
#%% =============================================================================
# Calibrate delay_n using mean water level of 2yr and 25yr storm as criteria
# =============================================================================
def func_delay(targets, delay_n, yr, dur_hr=24, plot=False):
    mins = 1
    T = mins * 60
    ##### Create the state-space model builder
    ss_model = SWMM2SSModel()
    ss_model.create_ss_model(storages, junctions, orifices, conduits, subcatchs,
                             delay_n, links, links_sb, storage_areas, T, nodes)
    X0 = ss_model.convert_Y0toX0(Y0) # Get initial value.
    ##### Create ss for stormwater system
    ss = SS(ss_model.A, ss_model.Bu, ss_model.Bw, ss_model.C, X0)
    
    df_swmm = cali_data[(yr, dur_hr)]
    obv_y = df_swmm[storages].values
    U = -df_swmm[orifices].to_numpy().T
    W = df_swmm[subcatchs].to_numpy().T
    for i in range(W.shape[1]):
        ss.run_step(U[:,i:i+1], W[:,i:i+1])
    
    inds = [int(t[-1])-1 for t in targets]
    rmses = []
    for ind in inds:
        sim_y = ss.get_record(ss.Y_act).T[:-1,:]
        rmse = np.nanmean((obv_y[:,ind] - sim_y[:,ind])**2, axis=0)**0.5
        rmses.append(rmse)
    if plot:
        fig, ax = plt.subplots()
        x = np.arange(0,obv_y.shape[0])#*mins
        ax.plot(x, obv_y[:,ind], label="obv")
        ax.plot(x, sim_y[:,ind], label="sim")
        ax.set_title("n{}, yr{},dur{}".format(delay_n, yr, dur_hr))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.5, "RMSE={}".format([round(r,5) for r in rmses]),
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=props)
        plt.legend()
        plt.show()
    return np.mean(rmses)

cali_res_delay = {}
n_list = [1,2,3,4,5]#,6,7,8,9,10]
# [2,4,4,3,1,1,2,2]
objs = []
for n in n_list:
    # ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
    delay_n = [1]*8
    delay_n[2] = n
    objs.append(np.mean([func_delay(["S3"], delay_n, yr=2, dur_hr=24, plot=True),
                         func_delay(["S3"], delay_n, yr=25, dur_hr=24, plot=True)]))
    cali_res_delay["J4"] = objs
print("J4: {} {}".format(np.argmin(objs)+1, [round(i, 5) for i in objs]))
# J4: 5 [0.00227, 0.00202, 0.00182, 0.00171, 0.00168] => 4
#%%
objs = []
for n in n_list:
    # ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
    delay_n = [1]*8
    delay_n[2] = 4
    delay_n[6] = n
    objs.append(np.mean([func_delay(["S7"], delay_n, yr=2, dur_hr=24, plot=False),
                         func_delay(["S7"], delay_n, yr=25, dur_hr=24, plot=False)]))
    cali_res_delay["J8"] = objs
print("J8: {} {}".format(np.argmin(objs)+1, [round(i, 5) for i in objs]))
# J8: 3 [0.00258, 0.00171, 0.00127, 0.0016, 0.00243] => 3
#%%
objs = []
for n in n_list:
    # ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
    delay_n = [1]*8
    delay_n[2] = 4
    delay_n[6] = 3
    delay_n[5] = n #3
    objs.append(np.mean([func_delay(["S6"], delay_n, yr=2, dur_hr=24, plot=False),
                         func_delay(["S6"], delay_n, yr=25, dur_hr=24, plot=False)]))
    cali_res_delay["J7"] = objs
print("J7: {} {}".format(np.argmin(objs)+1, [round(i, 5) for i in objs]))
# J7: 1 [0.01281, 0.03236, 0.05997, 0.08835, 0.11692] => 1
#%%
n_list = [1,2,3,4]
nd_list = []
for i in n_list:
    for ii in n_list:
        for iii in n_list:
            for iiii in n_list:
                for iiiii in n_list:
                    nd_list.append([i,ii,4,iii,iiii,1,3,iiiii])
mins = 1
objs = []
for delay_n in tqdm(nd_list):
    # ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
    objs.append(np.mean([func_delay(["S1"], delay_n, yr=2, dur_hr=24, plot=False),
                         func_delay(["S1"], delay_n, yr=25, dur_hr=24, plot=False)]))
    cali_res_delay["J"] = objs
print("J: {} {}".format(np.argmin(objs)+1, [round(i, 5) for i in objs]))
print("Final delay_n: ", nd_list[np.argmin(objs)]) # =>[1, 3, 4, 1, 1, 1, 3, 1]
#%%
np.mean(cali_res_delay["J"][128]
        + cali_res_delay["J4"][3]
        + cali_res_delay["J7"][0]
        + cali_res_delay["J8"][2])
#rmse = 0.017713077689980757 [m]
#%% 
mins = 1
T = mins * 60
delay_n = [1, 3, 4, 1, 1, 1, 3, 1]
ss_model = SWMM2SSModel()
ss_model.create_ss_model(storages, junctions, orifices, conduits, subcatchs,
                         delay_n, links, links_sb, storage_areas, T, nodes)
X0 = ss_model.convert_Y0toX0(Y0) # Get initial value.

def U_func(Y, W, unctrl_gates, orifices, areas, T):
    # ymaxs!!!
    def cal_unctrl_outflow(yh, Cg, Bg, mu, Ug, threshold=None):
        """Free flow"""        
        g = 9.8067 #"SI"
        return - Cg * mu * Ug * Bg * (2 * g * yh)**0.5
    U = np.array(
        [cal_unctrl_outflow(y, *unctrl_gates[o]) for y, o in zip(Y, orifices)]         
        ).reshape((-1, 1))
    
    # Add physical constrains due to discrete calculation.
    def bound(u, y, area):
        # We will need to design the max outflow
        u = max( - y * area / T, min(0, u))
        return u
    bound_U = np.vectorize(bound)
    
    return bound_U(U, Y, areas)

def func(x, orifice, sols):
    rmse_yr = []
    for yr in [2, 25]:
        df_swmm = cali_data[(yr, 24)]
        obv_y = df_swmm[storages].values
        obv_o = df_swmm[orifices].values
        W = df_swmm[subcatchs].to_numpy().T
        unctrl_gates = {}
        for i, o in enumerate(orifices):
            unctrl_gates[o] = [sols[i], r2[i], 0.65, r2[i]]
        unctrl_gates[orifice][0] = x 
        ss = SS(ss_model.A, ss_model.Bu, ss_model.Bw, ss_model.C, X0)
        for i in range(W.shape[1]):
            U = U_func(ss.Y_act[-1], W[:,i:i+1], unctrl_gates, orifices, areas, T)
            ss.run_step(U, W[:,i:i+1])
        
        ind = int(orifice[-1])-1
        sim_y = ss.get_record(ss.Y_act).T[:-1,:]
        rmse_y = np.nanmean((obv_y[:,ind] - sim_y[:,ind])**2, axis=0)**0.5
        sim_o = -ss.get_record(ss.U).T[:-1,:]
        rmse_o = np.nanmean((obv_o[:,ind] - sim_o[:,ind])**2, axis=0)**0.5
        # rmse_yr.append(rmse_o)
        rmse_yr.append(rmse_o+rmse_y)
    return np.mean(rmse_yr)


cali_order = ["O4","O8","O7","O2","O3","O5","O6","O9","O1"]
rmse_o_objs = []
sols = [0,0,0,0,0,0,0,0,0]
for cali_o in cali_order:
    sol_x, obj = local_min_search(
        func, max_iter=5, size=100, bound=[0,1], tol=10e-4,
        inputs=[cali_o, sols])
    sols[int(cali_o[-1])-1] = sol_x
    rmse_o_objs.append(obj)
    #sim_y, sim_o = sim(sol_x, cali_o, sols)
r"""
[0.98989898989899,
 0.9998979695949393,
 0.9998979695949393,
 0.9998979695949393,
 0.9998979695949393,
 0.9997959391898786,
 0.9996939087848179,
 0.494949494949495,
 0.9997959391898786]
"""
#%%
mins = 1
T = mins * 60
delay_n = [1, 3, 4, 1, 1, 1, 3, 1]
ss_model = SWMM2SSModel()
ss_model.create_ss_model(storages, junctions, orifices, conduits, subcatchs,
                         delay_n, links, links_sb, storage_areas, T, nodes)
X0 = ss_model.convert_Y0toX0(Y0) # Get initial value.

def sim_sol(sols, yr, roun=None):
    if roun is not None:
        sols = [round(sol, roun) for sol in sols]
    df_swmm = cali_data[(yr, 24)]
    obv_y = df_swmm[storages].values
    obv_o = df_swmm[orifices].values
    W = df_swmm[subcatchs].to_numpy().T
    unctrl_gates = {}
    for i, o in enumerate(orifices):
        unctrl_gates[o] = [sols[i], r2[i], 0.65, r2[i]]
    ss = SS(ss_model.A, ss_model.Bu, ss_model.Bw, ss_model.C, X0, c_scale=1)
    for i in range(W.shape[1]):
        U = U_func(ss.Y_act[-1], W[:,i:i+1], unctrl_gates, orifices, areas, T)
        ss.run_step(U, W[:,i:i+1])

    sim_y = ss.get_record(ss.Y_act).T[:-1,:]
    sim_o = -ss.get_record(ss.U).T[:-1,:]
    return sim_y, sim_o, obv_y, obv_o

def plot_cali(obv, sim, interval=1, ty="Outflow", note=""):
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True,
                             figsize=(6,6))
    plt.tight_layout(h_pad=2, w_pad=0)
    axes = axes.flatten()
    x = np.arange(0,obv.shape[0])*interval
    areas = [19517.91667, 630, 6838, 800, 750, 350, 1885, 3848.333333, 1000] # m2
    RMSE = []
    for i, v in enumerate([8,4,2,7,5,3,9,6,1]):
        ax = axes[i]
        ind = v-1
        # if ty != "Outflow":
        #     o = "S"+o[-1]
        o = "Pond " + str(v)
        ax.plot(x, obv[:,ind], label="SWMM",lw=1.5)
        ax.plot(x, sim[:,ind], label="State-space model",lw=1)
        rmse = np.nanmean((obv[:,ind] - sim[:,ind])**2, axis=0)**0.5
        RMSE.append(rmse)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, "RMSE={}".format(round(rmse,3)),
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=props)
        ax.set_title(o, fontsize=10)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel("Time (minute)")
    if ty == "Outflow":
        plt.ylabel("Outflow ($m^3/sec$) {}".format(note))
        RMSE = [RMSE[i]*60/areas[v-1] for i, v in enumerate([8,4,2,7,5,3,9,6,1])]
    elif ty == "Water level":
        plt.ylabel("Water level (m) {}\n".format(note))
    elif ty == "Water level%":
        plt.ylabel("Water level (%) {}".format(note))
    axes[2].legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.05, 1.35), ncol=2)
    #plt.tight_layout()
    print(sum(RMSE)/len(RMSE))
    plt.show()

sols = [0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.495, 1.0]
sim_y, sim_o, obv_y, obv_o = sim_sol(sols, yr=2, roun=3)
plot_cali(obv_o, sim_o, ty="Outflow",interval=1,note="[2yr-24hr storm]")
plot_cali(obv_y, sim_y, ty="Water level",interval=1,note="[2yr-24hr storm]")
sim_y, sim_o, obv_y, obv_o = sim_sol(sols, yr=25, roun=3)
plot_cali(obv_o, sim_o, ty="Outflow",interval=1,note="[25yr-24hr storm]")
plot_cali(obv_y, sim_y, ty="Water level",interval=1,note="[25yr-24hr storm]")
r"""
RMSE
2yrO:   0.018    [cms] = 0.0006489 [m]
2yrWL:  0.039    [m]
25yrO:  0.030    [cms] = 0.0014944 [m]
25yrWL: 0.051    [m]
"""
RMSE_Cg = np.mean([0.0006489507108482905,0.039003073101443055,
                  0.0014944368749057687,0.05080974446299849])*100 # cm
RMSE_nc = np.mean([0.039003073101443055,0.05080974446299849])*100 # cm







