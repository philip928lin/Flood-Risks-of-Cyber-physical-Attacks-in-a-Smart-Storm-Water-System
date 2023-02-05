import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = "1" # conda fix for pyswmm
prj_path, this_filename = os.path.split(__file__)
os.chdir(prj_path)
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyswmm import Simulation, Links, Nodes, Subcatchments, RainGages
from lqg_controller import LQG, LQE, SS

def find_interval(x):
    """
    array([[0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 0, 0, 1, 1],
       [0, 0, 0, 1, 1, 1, 0]])
    [array([[4, 6]], dtype=int64),
     array([[0, 2],
            [5, 6]], dtype=int64),
     array([[3, 5]], dtype=int64)]
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    # find switches 0->1 and 1->0
    d = np.empty((np.arange(2) + x.shape), bool)
    d[:, 0] = x[:, 0]   # a 1 in the first
    d[:, -1] = x[:, -1] # or last column counts as a switch
    d[:, 1:-1] = x[:, 1:] != x[:, :-1]

    # find switch indices (of flattened array)
    b = np.flatnonzero(d)
    # create helper array of row offsets
    o = np.arange(0, d.size, d.shape[1])
    # split into rows, subtract row offsets and reshape into start, end pairs
    result = [(x-y).reshape(-1, 2) - np.arange(2) for x, y in zip(np.split(b, b.searchsorted(o[1:])), o)]
    return result
#%%
def plot_ss(FDI_list=[], dY_list=[], c=None, unc=None, Fh=None, FDI_labels=[],
            ylabel="", ty="O", ymax=None, sharey=True, wc=80, stoch_data_list=[],
            ylim=None, seg=None, prob=None, t_tuple=None, ref=None):
    mins=1
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=sharey,
                             figsize=(6,4))
    plt.tight_layout(h_pad=1, w_pad=0)
    axes = axes.flatten()

    data_list = [unc, c] + FDI_list
    labels = ["Uncontrolled", "Controlled"] + FDI_labels
    if len(FDI_labels) != len(FDI_list):
        labels += ["FDIs-{}".format(i+1) for i in range(len(FDI_list))]

    if ymax is not None and ty == "Y%":
        data_list = [d/ymax*100 if d is not None else None for d in data_list]
        stoch_data_list = [d/ymax*100 if d is not None else None for d in stoch_data_list]
    if Fh is not None:
        Fh = find_interval(Fh.T)

    # Adjust seg
    if seg is not None:
        data_list = [d[seg[0]:seg[1], :] if d is not None else None for d in data_list]
        stoch_data_list = [d[seg[0]:seg[1], :] if d is not None else None for d in stoch_data_list]
    else:
        seg = [0]

    ls_list = [":", "--", "-", "-", "-.", "-", "-", "-", "-"]
    c_list = [9, 5, 1, 2, 3, 6, 8]
    for k, v in enumerate([8,4,2,7,5,3,9,6,1]):
        ax = axes[k]
        
        if ref is not None:
            x = np.arange(seg[0],seg[0]+ref.shape[0]) * mins
            ax.plot(x, ref[:,v-1], label="$\mathscr{r}$", lw=2.5, alpha=0.5,
                    ls="-", color="red", zorder=-10)
        
        #ax.set_yscale('log')
        # if Fh is not None:
        #     fh = Fh[v-1]
        #     for fhh in fh:
        #         if fhh != [] and prob is None:
        #             #ax.axvspan(fhh[0], fhh[1], alpha=0.5, color='grey')
        #             ax.patch.set_facecolor('lavender')
        # Add stochastic simulations.
        for i, d in tqdm(enumerate(stoch_data_list)):
            x = np.arange(seg[0],seg[0]+d.shape[0]) * mins
            if i == 0:
                ax.plot(x, d[:,v-1], lw=1, color="darkgray", alpha=0.8,
                        zorder=-1, label=r"{$V,\widetilde{W}$} realization")
            ax.plot(x, d[:,v-1], lw=0.2, color="darkgray", alpha=0.3, zorder=-1)
            # if prob is not None:
            #     ax.fill_between(x, prob[v-1]*100, color="lavender")

        for i, d in enumerate(data_list):
            if d is None:
                continue
            x = np.arange(seg[0],seg[0]+d.shape[0]) * mins
            if i == 0 or i == 1:
                lw = 1
            else:
                lw = 1.2+0.8*max(i-2,0) #2*0.8**max(i-2,0)
            if i >= 2 and t_tuple is not None:
                tp, nT = t_tuple
                ax.plot(x[:tp[v-1]+nT-seg[0]+1], d[:tp[v-1]+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                        ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
            else:
                ax.plot(x, d[:,v-1], label=labels[i], lw=lw,
                        ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
            ax.set_title("Pond "+str(v), fontsize=9)
            if prob is not None:
                ax.set_title("Pond {} ({}%)".format(v, round(prob[v-1]*100,1)), fontsize=9)
            elif Fh is not None:
                fh = Fh[v-1]
                for fhh in fh:
                    if fhh != [] and prob is None:
                        ax.set_title("Pond "+str(v)+"*", fontsize=9)

        if ty == "Y%":
            ax.axhline(wc, color="black", ls="dashed", lw=0.5)
            ax.set_ylim([0,110])
        if ty == "Y":
            ax.axhline(ymax[v-1], color="black", ls="dashed", lw=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim([x[0],x[-1]])

    # Add magnified plot
    if stoch_data_list != [] and t_tuple is not None:
        for k, v in enumerate([8,4,2,7,5,3,9,6,1]):
            ax = axes[k]
            locs = ax.get_position()
            x_, y_, w_, h_ = locs.x0, locs.y0, locs.width, locs.height
            axx = fig.add_axes([x_+w_/15 , y_+h_/3 , w_/3, h_*1/2])

            for i, d in tqdm(enumerate(stoch_data_list)):
                x = np.arange(seg[0],seg[0]+d.shape[0]) * mins
                x = x[tp[v-1]+1:tp[v-1]+nT-seg[0]+1]
                axx.plot(x, d[tp[v-1]+1:tp[v-1]+nT-seg[0]+1,v-1], lw=0.2,
                        color="darkgray", alpha=0.3, zorder=-1)
            for i, d in enumerate(data_list):
                if d is None:
                    continue
                x = np.arange(seg[0],seg[0]+d.shape[0]) * mins
                x = x[tp[v-1]+1:tp[v-1]+nT-seg[0]+1]
                if i == 0 or i == 1:
                    lw = 1
                else:
                    lw = 1.2+0.8*max(i-2,0) #2*0.8**max(i-2,0)
                if i >= 2 and t_tuple is not None:
                    tp, nT = t_tuple
                    axx.plot(x, d[tp[v-1]+1:tp[v-1]+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                            ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
                else:
                    axx.plot(x, d[tp[v-1]+1:tp[v-1]+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                            ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
                axx.axhline(wc, color="black", ls="dashed", lw=0.5)
                axx.set_yticks([])
                axx.set_ylim([75,125])
                axx.set_xlim([x[0],x[-1]])
                axx.set_xticks([x[0], x[14], x[-1]])
                axx.set_xticklabels([x[0], x[14], x[-1]], rotation=45, fontsize=6)
                axx.set_yticks([105])
                axx.set_yticklabels(["100"], rotation=90, fontsize=6)
                axx.tick_params(axis="y", length=0, pad=0.3)
                axx.tick_params(axis="x", pad=1)
    # Add stem plots.
    marker_list = ["o","d"]
    if dY_list != []:
        for k, v in enumerate([8,4,2,7,5,3,9,6,1]):
            ax = axes[k]
            locs = ax.get_position()
            x_, y_, w_, h_ = locs.x0, locs.y0, locs.width, locs.height
            axx = fig.add_axes([x_+w_/15 , y_+h_/3 , w_/3, h_/3])
            for i, dy_list in enumerate(dY_list[v-1]):
                (markers, stemlines, baseline) = axx.stem(dy_list)
                dif = np.max(dY_list[v-1]) - np.min(dY_list[v-1])
                axx.set_ylim([np.min(dY_list[v-1])-dif*0.1, np.max(dY_list[v-1])+dif*0.1])

                #print([np.min(dY_list[v-1]), np.max(dY_list[v-1])], "  ", dif)
                plt.setp(markers, markersize=3-0.5*i, color="C{}".format(c_list[i+2]), marker=marker_list[i])
                #plt.setp(markers[v-1], marker='D')
                plt.setp(baseline, color="k", lw=0.8)
                plt.setp(stemlines, color="grey", lw=0.7)
            axx.set_axis_off()
            axx.text(0, -0.5, '123456789', transform=axx.transAxes, fontsize=6.7)
    # plt setting
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel("Time (minute)")
    plt.ylabel(ylabel)
    axes[2].legend(ncol=4, fontsize=9, loc="upper right", bbox_to_anchor=(1.05, 1.55),
                   handletextpad=0.5, columnspacing=1.2)
    plt.show()
#%%
def gen_ref(sim_y, ymax, wc=0.85, tl=60*5, return_tc=False):
    """
    Get reference setpoints.

    Parameters
    ----------
    sim_y : 2darray
        (nT, nS).
    ymax : list
        Maximum height. Ref has the same unit as ymax.
    wc : float, optional
        % of capacity used for detention after peak. The default is 0.85.
    tl : int, optional
        Time length for detention [sec]. The default is 60*5.
    return_tc : bool, optional
        return [tc_q10, tc_max] if True. The default is False.

    Returns
    -------
    2darry
        (nS, nT).

    """
    tc_q10 = []
    tc_max = []
    numS = len(ymax)
    for i in range(numS):
        tc_q10.append(np.where(sim_y[:,i] > ymax[i]*0.05)[0][0]-1)
        tc_max.append(np.where(sim_y[:,i] == max(sim_y[:,i]))[0][-1])
    tc = [tc_q10, tc_max]
    print("tc_q10: {}".format(tc_q10))
    print("tc_max: {}".format(tc_max))
    if return_tc:
        return tc
    R = np.zeros(sim_y.T.shape)
    if wc != 0:
        for i in range(1,numS):
            R[i, tc_max[i]:tc_max[i]+tl] = ymax[i] * wc

            l1 = R[i, tc_q10[i]:tc_max[i]].shape[0]
            intep1 = np.arange(0, ymax[i] * wc, ymax[i] * wc / l1)
            try:
                R[i, tc_q10[i]:tc_max[i]] = intep1
            except:
                R[i, tc_q10[i]:tc_max[i]] = intep1[:-1]

            l2 = R[i, tc_max[i]+tl:].shape[0] - 1
            intep2 = np.arange(ymax[i] * wc, 0, -ymax[i] * wc / l2)
            try:
                R[i, tc_max[i]+tl:-1] = intep2
            except:
                R[i, tc_max[i]+tl:-1] = intep2[:-1]
    return R

def set_sim(ss_model, sig_V=0, sig_W=[0.1]*9, rho_q=None, rho_r=None, roun=5, gw=10000):
    ##### Create LQG controller.
    eye = np.eye(ss_model.C.shape[0])
    Cov_V = eye * (sig_V)**2
    Cov_W = np.diag(sig_W) @ np.diag(sig_W)
    lqg = LQG(ss_model.A, ss_model.Bu, ss_model.Bw, ss_model.C, Cov_V, Cov_W,
              rho_q, rho_r, roun, Q=None, R=None, Ain_dict=ss_model.Ain_dict)
    ##### Create ss for stormwater system
    X0 = ss_model.convert_Y0toX0(np.array([0]*ss_model.C.shape[0]).reshape((-1,1)))
    ss = SS(lqg.A, lqg.Bu, lqg.Bw, lqg.C, X0, Ain_dict=lqg.Ain_dict)
    lqe = LQE(lqg.A, lqg.Bu, lqg.Bw, lqg.C, lqg.L, X0, g=None, gw=gw, Ain_dict=lqg.Ain_dict)
    return ss, lqg, lqe

def run_sim(ss, lqg, lqe, W, Wn, V, Ya, R, ymax, uc, control=True, s1_o=None, Err=None, output_YaY=False):
    V = V.copy()
    ##### Uhmax & Bound
    def Uhmax(Y, ymax, uc):
        Uuc = [-uc[i] * (min(max(float(yh),0), ymax[i]))**0.5 for i, yh in enumerate(Y)]
        Uaw = [-yh[0] for yh in Y]
        Uh_max = [max(uuc, uaw) for uuc, uaw in zip(Uuc, Uaw)]
        return np.array(Uh_max).reshape((-1, 1))

    def bound(u, uhmax):
        u = max(uhmax, min(0, u))
        return u
    bound_U = np.vectorize(bound)

    # If the sensor is attacked, no noise for that sensor as it is control by the attacker entirely.
    if Ya is not None:
        V[np.isnan(Ya) == False] = Ya[np.isnan(Ya) == False]

    ##### Run simulation
    YaY = np.zeros(V.shape)
    YaY[:] = np.nan
    for i in range(W.shape[1]):
        Uh_max = Uhmax(ss.Y_act[-1], ymax, uc)
        if control:
            kk = -lqg.Kk @ lqe.X_hat[-1]
            kr = lqg.Kr @ R[:,i:i+1]
            kw = lqg.Kw @ (W[:,i:i+1] + Wn[:,i:i+1])
            U_dc = kk + kr + kw
            U = bound_U(U_dc, Uh_max)    # U_dc is updated based on mins_dc
            if s1_o is not None:        # Control S1 max outflow
                U[0,0] = max(s1_o, U[0,0])
        else:
            U = Uh_max
            if s1_o is not None:
                U[0,0] = max(s1_o, U[0,0])
        # Here we assume the W is a perfect measured value and will not be attacked.
        if output_YaY:
            YaY[:,i:i+1] = ss.Y[-1] + V[:,i:i+1]
        lqe.run_step(ss.Y[-1] + V[:,i:i+1],
                     U,
                     W[:,i:i+1])
        ss.run_step(U, W[:,i:i+1])
    
    if output_YaY:
        YaY[np.isnan(Ya) == True] = np.nan
        return ss, lqe, YaY
    return ss, lqe

# =============================================================================
# SWMM stepwise simulation (powered by pyswmm)
# =============================================================================
def swmm_info(swmm):
    """
    Show the basic information about the given swmm model.

    Parameters
    ----------
    swmm : str or pyswmm Simulation object

    Returns
    -------
    None.

    """
    def info(sim):
        print("Unit:", "\n\t", sim.system_units, "\n\t", sim.flow_units)
        print("Start:", "\n\t", sim.start_time)
        print("End:", "\n\t", sim.end_time)
        #print("Step:", "\n\t")
        print("Nodes:")
        nodes = Nodes(sim)
        print("\t", [node.nodeid for node in nodes])
        print("Links:")
        links = Links(sim)
        print("\t", [link.linkid for link in links])
        print("Subcatchments:")
        subs = Subcatchments(sim)
        print("\t", [sub.subcatchmentid for sub in subs])
        print("RainGages:")
        rgs = RainGages(sim)
        print("\t", [rg.raingageid for rg in rgs])
    if isinstance(swmm, str):
        with Simulation(swmm) as sim:
            info(sim)
    else:
        info(swmm)

def swmm_init_storage_depth(swmm):
    """
    Get initial depth for the storage nodes.

    Parameters
    ----------
    swmm : str or pyswmm Simulation object

    Returns
    -------
    list

    """
    def initial_depth(sim):
        return [i.initial_depth for i in Nodes(sim) if i.is_storage()]
    if isinstance(swmm, str):
        with Simulation(swmm) as sim:
            X0 = initial_depth(sim)
    else:
        X0 = initial_depth(swmm)
    return X0

def swmm_get_object(sim):
    """
    Extract the id for storages, junctions, orifices, conduits, subcatchs,
    raingauges in a swmm model.

    Parameters
    ----------
    sim : pyswmm Simulation object

    Returns
    -------
    storages : list
    junctions : list
    orifices : list
    conduits : list
    subcatchs : list
    raingauges : list
    """
    storages = [i.nodeid for i in Nodes(sim) if i.is_storage()]
    junctions = [i.nodeid for i in Nodes(sim) if i.is_junction()]
    orifices = [i.linkid for i in Links(sim) if i.is_orifice()]
    conduits = [i.linkid for i in Links(sim) if i.is_conduit()]
    subcatchs = [i.subcatchmentid for i in Subcatchments(sim)]
    raingauges = [i.raingageid for i in RainGages(sim)]
    return storages, junctions, orifices, conduits, subcatchs, raingauges

def swmm_run(sim, T=None, full_depth=None, display=True):
    """
    Run stepwise swmm simulation.

    Note: If got ERROR 317, please make sure the external file has absolute
    path in .inp (not just filename).

    Parameters
    ----------
    sim : pyswmm Simulation object

    T : int, optional
        Recording time interval [sec]. This will not affect the routing time
        step in the swmm model. The default is None.
    full_depth : dict or int, optional
        Change the max depth or storage nodes. If given an integer, the value
        will be apply to all storage nodes. Otherwise, a dictionary with key =
        storage node id and value = new max depth need to be given. The depth
        unit is corresponding to swmm setting. The default is None.

    Returns
    -------
    swmm_output : Dataframe

    """
    if display:
        swmm_info(sim)
    storages, junctions, orifices, conduits, subcatchs, raingauges = swmm_get_object(sim)
    swmm_object = {item: [] for item in ["storages", "junctions", "orifices",
                                         "conduits", "subcatchs",
                                         "raingauges"]}
    swmm_output = {item: [] for item in storages + junctions + orifices
                   + conduits + subcatchs + raingauges + ["time"]}
    for s in storages:
        swmm_output["F"+s] = []

    ##### Start Simulation
    swmm_object["storages"] = [Nodes(sim)[s] for s in storages]
    swmm_object["junctions"] = [Nodes(sim)[j] for j in junctions]
    swmm_object["orifices"] = [Links(sim)[o] for o in orifices]
    swmm_object["conduits"] = [Links(sim)[c] for c in conduits]
    swmm_object["subcatchs"] = [Subcatchments(sim)[sb] for sb in subcatchs]
    swmm_object["raingauges"] = [RainGages(sim)[r] for r in raingauges]
    if T is not None:
        sim.step_advance(T)
    if full_depth is not None:
        if isinstance(full_depth, dict):
            for i, v in full_depth:
                Nodes(sim)[i].full_depth = v
        else:   # If given a number.
            for s in storages:
                Nodes(sim)[s].full_depth = full_depth

    for step in sim:
        for i, v in enumerate(storages):    # Depth [m]
            swmm_output[v].append(swmm_object["storages"][i].depth)
            swmm_output["F"+v].append(swmm_object["storages"][i].total_outflow)

        for i, v in enumerate(junctions):   # Flow rate [cms]
            swmm_output[v].append(swmm_object["junctions"][i].flooding)
        for i, v in enumerate(orifices):    # Flow rate [cms]
            swmm_output[v].append(swmm_object["orifices"][i].flow)
        for i, v in enumerate(conduits):    # Flow rate [cms]
            swmm_output[v].append(swmm_object["conduits"][i].flow)
        for i, v in enumerate(subcatchs):   # Flow rate [cms]
            swmm_output[v].append(swmm_object["subcatchs"][i].runoff)
        for i, v in enumerate(raingauges):  # Intensity [mm/hr]
            swmm_output[v].append(swmm_object["raingauges"][i].rainfall)
        swmm_output["time"].append(sim.current_time)
    for i, v in enumerate(raingauges):
        swmm_output[v] = swmm_output[v][1:] + [swmm_output[v][-1]] #pyswmm cannot output last one.
    df_swmm = pd.DataFrame(swmm_output)
    df_swmm = df_swmm.set_index("time")
    # Calculate flood hight
    for i, v in enumerate(storages):
        df_swmm["Fh_"+v] = [max(h - Nodes(sim)[v].full_depth, 0) for h in df_swmm[v]]
        df_swmm[v+"%"] = df_swmm[v]/Nodes(sim)[v].full_depth*100
        print(v,": ", round(max(df_swmm[v+"%"]), 2))
    if display:
        print("SWMM simulation done!")
    return df_swmm

def swmm_set_sim_period(sim, start_time, end_time):
    """
    Set simulation period.

    Parameters
    ----------
    sim : pyswmm Simulation object
    start_time : datetime object
    end_time : datetime object

    Returns
    -------
    None.

    """
    # Has to be a datetime object.
    sim.start_time = start_time
    sim.end_time = end_time

def swmm_gen_rainfall_file(sim, filename, depth, T="5min", stn_id="stn1"):
    """
    Generate rainfall file .txt.

    Note that the filename has to be the same as swmm .inp setting. Please
    double check swmm's .inp has been setup corespondingly.

    Parameters
    ----------
    sim : pyswmm Simulation object
    filename : str
    depth : array
        Rainfall data. Unit is determined in swmm .inp setting.
    T : time interval, optional
        The resolution of the rainfall data. The default is "5min".
    stn_id : str, optional
        Station id. The default is "stn1".

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    depth = np.array(depth)
    start_time = sim.start_time
    end_time = sim.end_time
    rng = pd.date_range(start=start_time, end=end_time, freq=T)
    if len(depth) != len(rng):
        print("Data (depth) length is not corresponding to datetime range.")
    df = pd.DataFrame(index=rng,
                      columns=["stn_id", "year", "month", "day", "hour",
                               "minute", "depth"])
    df["stn_id"] = stn_id
    df["year"] = rng.year
    df["month"] = rng.month
    df["day"] = rng.day
    df["hour"] = rng.hour
    df["minute"] = rng.minute
    df["depth"] = depth
    df.to_csv(filename, index=None, header=None, sep=" ")
    return df

def swmm_get_links(sim):
    """
    Extract links information.

    Parameters
    ----------
    sim : pyswmm Simulation object

    Returns
    -------
    links : dict
        Linkages between nodes.
    links_sb : dict
        Linkages between subcatchments and nodes.

    """
    links = {i.linkid: i.connections for i in Links(sim)}
    links_sb = {}
    for i in Subcatchments(sim):
        c = i.connection
        # Currently, we only allow linkage between subcatch to storages or
        # junctions. The linkage between subcatch to subcatch is forbiden.
        #if c[0] == "2":
        try:
            links_sb[c].append(i.subcatchmentid)
        except:
            links_sb[c] = [i.subcatchmentid]
    return links, links_sb

#%%
# =============================================================================
# SWMM inp file modification
# =============================================================================
def swmm_get_inp_section(inp_path, section):
    """
    Get swmm .inp file section to dataframe.

    Parameters
    ----------
    inp_path : str
        .inp filename.
    section : str
        .inp file section name (e.g., "ORIFICES").

    Returns
    -------
    df : DataFrame

    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    try:
        section_idx = lines.index("[{}]\n".format(section))
    except ValueError:
        section_idx = [i for i, v in enumerate(lines) if "[{}]".format(section) in v][0]

    col = re.split(r'\s{2,}', lines[section_idx+1][2:])[:-1]
    if section == "RAINGAGES":
        col = ["Name", "Format", "Interval", "SCF", "Source"]
    if section == "STORAGE":
        col = ["Name", "Elev.", "MaxDepth", "InitDepth",
               "Shape", "CNPa", "CNPb", "CNPc",
               "Surchage_depth", "Fevap", "Psi", "Ksat",
               "IMD"]
    skip = 3
    l = section_idx + skip
    content = []
    while lines[l][:-2] != "":
        if lines[l][0] == ";":      # comment line.
            l += 1
        else:
            c = lines[l][:-2].split()
            c = c + [np.nan]*(len(col)-len(c))
            content.append(c)
            l += 1
    if len(col) != len(content[0]):
        slack = ["slack_"+str(i) for i in range(len(content[0]) - len(col))]
        col += slack
        warnings.warn("The column name may not match but it is ok.")
    df = pd.DataFrame(content, columns=col)
    df = df.apply(pd.to_numeric, errors='ignore')
    try:
        df = df.sort_values("Name")
    except:
        pass
    return df

def swmm_write_inp_section(inp_path, section, df, save_as=None):
    """
    Write df to replace original section in given .inp file.

    Parameters
    ----------
    inp_path : str
        .inp filename.
    section : str
        .inp file section name (e.g., "ORIFICES").
    df : DataFrame
        Section dataframe.
    save_as : str, optional
        New filename if want to keep the original file. The default is None.

    Returns
    -------
    None.

    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    try:
        section_idx = lines.index("[{}]\n".format(section))
    except ValueError:
        section_idx = [i for i, v in enumerate(lines) if "[{}]".format(section) in v][0]
    skip = 3
    l = section_idx + skip
    while lines[l][:-2] != "":
        if lines[l][0] == ";":      # comment line.
            skip += 1
        l += 1
    del lines[section_idx + skip:l]

    df = df.fillna(" ")
    content = df.to_string(header=None, index=None)
    lines = lines[:section_idx + skip] + [content, "\n"] + lines[section_idx + skip:]

    if save_as is None:
        inp_path_out = inp_path
    else:
        inp_path_out = save_as
    with open(inp_path_out, 'w') as f:
        f.writelines(lines)

def swmm_get_inp_sections(inp_path, sections=None):
    if sections is None:
        sections = ["RAINGAGES", "SUBCATCHMENTS", "SUBAREAS",
                    "INFILTRATION", "JUNCTIONS", "OUTFALLS",
                    "STORAGE", "CONDUITS", "ORIFICES",
                    "XSECTIONS"]
    dict = {}
    for s in sections:
        dict[s] = swmm_get_inp_section(inp_path, s)
    return dict

def swmm_write_inp_sections(inp_path, dict, save_as=None):
    path = inp_path
    for s in dict:
        swmm_write_inp_section(path, s, dict[s], save_as)
        if save_as is not None:
            path = save_as
#%%

def cal_unctrl_outflow(yh, Cg, Bg, mu, Ug, ymax=np.inf, area=None, T_dc=1, scale=1):
    """
    Calculate uncontrallable gate's outflow.

    Assume the orifice is rectangular and located at the bottom of the storage.
    Q = Cg * mu * Ug * Bg * (2 * 9.81 * yh)^0.5

    Parameters
    ----------
    yh : float
        Water level.
    Cg : float
        Calibrated parameter.
    Bg : float
        Gate's width.
    mu : float
        The coefficient of contraction.
    Ug : float
        Gate opening.
    ymax: float
        Maximun height for free flow calculation

    Returns
    -------
    float
        Outflow.

    """

    """Free flow"""
    g = 9.81 #"SI"
    # [0, ymax]
    yh = min( max(float(yh),0), ymax)
    if area is None: #[cms]
        return - Cg * mu * Ug * Bg * (2 * g * yh)**0.5
    else: # [m] *scale
        #print(- (Cg * mu * Ug * Bg * (2 * g * yh/scale)**0.5) * T_dc /area *scale)
        return - (Cg * mu * Ug * Bg * (2 * g * yh/scale)**0.5) * T_dc /area *scale

def cal_aw(yh, area=None, T_dc=1):
    """
    Calculation available water for outflow.

    Parameters
    ----------
    yh : float
        Water level.
    area : float
        Pond's surface area.
    T_dc : float
        Decision-making timestep.

    Returns
    -------
    float
        flow rate [cms].

    """
    yh = max(float(yh),0)
    if area is None: # [m]
        #print(- yh)
        return - yh
    else: # [cms]
        return - yh * area / T_dc

def rmse(x_obv, y_sim):
    """
    Root mean square error

    Parameters
    ----------
    x_obv : array
        x or obv.
    y_sim : array
        y or sim.

    Returns
    -------
    float

    """
    return np.nanmean((x_obv - y_sim)**2)**0.5