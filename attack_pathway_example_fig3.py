import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
module_path = os.path.join(prj_path, "Code")
os.chdir(module_path)
import numpy as np
import matplotlib.pyplot as plt
from gp_model import GPModel
from lqg_controller import SWMM2SSModel
from util import gen_ref, set_sim, run_sim

storages = ['S1', 'S2', 'S3']
junctions = ['J2', 'J3']
orifices = ['O1', 'O2', 'O3']
conduits = ['C2', 'C3']
subcatchs = ['Sb1', 'Sb2', 'Sb3']
delay_n = [1,1]
links = {'C2': ('J2', 'S1'),
         'C3': ('J3', 'S1'),
         'O1': ('S1', 'OUTLET'),
         'O2': ('S2', 'J2'),
         'O3': ('S3', 'J3')}
links_sb = {'S1': ['Sb1'],
            'S2': ['Sb2'],
            'S3': ['Sb3']}
storage_areas = {'S1': 2000,
                 'S2': 1000,
                 'S3': 1000}
nodes = ["S2", "J2", "S3", "J3", "S1"]


ss_model = SWMM2SSModel()
ss_model.create_ss_model(storages, junctions, orifices, conduits, subcatchs,
                         delay_n, links, links_sb, storage_areas, nodes=nodes, ratio=True)

#%%
def gen_w(ini=0, tp=20, l=60, peak=10, roun=2, start=None, end=None):
    if end is not None:
        num0e = l-end
        l = end
    else:
        num0e=0

    if start is not None:
        num0s = start
        l = end-start
        tp = tp-start
    else:
        num0s=0

    dup =  (peak-ini)/(tp-1)
    ddown = (0-peak)/(l-tp-1)
    w = [0]*num0s \
        + [round(ini+dup*i, roun) for i in range(tp)] \
        + [round(ini++dup*(tp-1)+ddown*i, roun) for i in range(1, l-tp)] \
        + [0]*num0e
    return np.array(w)
uc = []; r2 = [0.9144, 0.9144/2, 0.9144/2]; m2cm=100; areas = list(storage_areas.values())
for i in range(3):
    uc.append(round(1 * 0.65 * r2[i]**2 * (2*9.81*m2cm)**0.5 * 60 / areas[i], 6))
ymax = np.array([110, 110, 110])

sig_V = 0.25
sig_W = [2, 2, 2]
rho_q = [100, 100, 100]
rho_r = [1, 1, 1]
gw = 1

w1 = gen_w(ini=0, tp=25, l=60, peak=5, roun=2, start=15, end=35)
w2 = gen_w(ini=0, tp=10, l=60, peak=7, roun=2, start=8, end=30)
w3 = gen_w(ini=0, tp=10, l=60, peak=9, roun=2, start=5, end=20)
W = np.array([w1, w2, w3])*2

M0 = np.zeros(W.shape)
Y0 = np.array([0]*3).reshape((-1,1))

ss_uc, lqg, lqe_uc = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss_uc, lqe_uc = run_sim(ss_uc, lqg, lqe_uc, W, Wn=M0, V=M0, Ya=M0, R=M0, ymax=ymax, uc=uc, control=False, s1_o=None)
sim_y_uc, sim_o_uc = ss_uc.get_ss_record()

Ref = gen_ref(sim_y_uc, ymax, wc=0.85, tl=5, return_tc=False)
tc_q10, tc_max = gen_ref(sim_y_uc, ymax, tl=5, return_tc=True)
ss, lqg, lqe = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss, lqe = run_sim(ss, lqg, lqe, W, Wn=M0, V=M0, Ya=M0, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
sim_y, sim_o = ss.get_ss_record()

print(sim_o_uc.max(axis=0).round(2))
print(sim_o.max(axis=0).round(2))
#%% MaxInflow
iniT = 10
dT = 20
uya = float('inf')
uYa = [0, uya, uya]
s = 1
noF = ["S{}".format(i+1) for i in range(3) if i+1 != s]

gpm = GPModel(W, lqg, lqe, ss, Ref, uc, name="MaxInflow", demo=True)
gpm.set_gp(iniT, dT, uYa=uYa)
gpm.addObj_maxInflow("S{}".format(s))
gpm.addC_Err(eps=20)
gpm.addC_noflood(noF, tol=1e-5)
gpm.optimize_gp(wd=os.path.join(model_path, "OptExp"), MIPFocus=1, log=True, FeasibilityTol=1e-5)

Ya_i = gpm.records["Ya"].T
ss_a_i, lqg, lqe_a_i = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss_a_i, lqe_a_i = run_sim(ss_a_i, lqg, lqe_a_i, W, Wn=M0, V=M0, Ya=Ya_i, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
sim_y_a_i, sim_o_a_i = ss_a_i.get_ss_record()
#%%
in_uc = ss_uc.get_record(ss_uc.Sin_dict["S1"]).sum(axis=0)[0:-1] #+ w1
in_c = ss.get_record(ss.Sin_dict["S1"]).sum(axis=0)[0:-1] #+ w1
in_a_i = ss_a_i.get_record(ss_a_i.Sin_dict["S1"]).sum(axis=0)[0:-1] #+ w1
fig, ax = plt.subplots(figsize=(2.8,3))
ax.bar(x=0.3, height=in_a_i.max(), tick_label="FDIs", 
       width=0.5, color="darkorange")
ax.set_xlim([0,1.8])
ax.set_ylabel("Pond 1 peak inflow")
ax.set_yticks([])
ax.scatter(x=0.3, y=in_uc.max(), zorder=10, color="k", s=10)
ax.scatter(x=0.3, y=in_c.max(), zorder=10, color="k", s=10, marker="x")
ax.arrow(x=0.3, y=in_uc.max(), dx=0.5, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Uncontrolled", xy=(0.9, in_uc.max()-0.03), xytext=(0, 0),
          textcoords='offset points', fontsize=8)
ax.arrow(x=0.3, y=in_c.max(), dx=0.8, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Controlled", xy=(1.2, in_c.max()-0.07), xytext=(0, 0),
          textcoords='offset points', fontsize=8)
# for inflow in [in_uc, in_c, in_a_i]:
#     ax.plot(inflow)
# ax.plot(w1)

#%% MinOutflow
iniT = 10
dT = 20
uya = float('inf')
#uYa = [0, uya, uya]
uYa = [uya, 0, 0]
s = 1
noF = ["S{}".format(i+1) for i in range(3) if i+1 != s]
gpm = GPModel(W, lqg, lqe, ss, Ref, uc, name="MaxInflow", demo=True)
gpm.set_gp(iniT, dT, uYa=uYa)
gpm.addObj_maxY("S{}".format(s))
gpm.addC_Err(eps=20)
gpm.addC_noflood(noF, tol=1e-5)
gpm.optimize_gp(wd=os.path.join(model_path, "OptExp"), MIPFocus=1, log=True, FeasibilityTol=1e-5)

Ya_o = gpm.records["Ya"].T
ss_a_o, lqg, lqe_a_o = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss_a_o, lqe_a_o = run_sim(ss_a_o, lqg, lqe_a_o, W, Wn=M0, V=M0, Ya=Ya_o, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
sim_y_a_o, sim_o_a_o = ss_a_o.get_ss_record()

#%%
o1_uc = sim_o_uc[iniT:iniT+dT,0].mean()
o1_c = sim_o[iniT:iniT+dT,0].mean()
o1_a = sim_o_a_o[iniT:iniT+dT,0].mean()
fig, ax = plt.subplots(figsize=(2.8,3))
ax.bar(x=0.3, height=o1_a, tick_label="FDIs", 
       width=0.5, color="gold")
ax.set_xlim([0,1.8])
ax.set_yticks([])
ax.set_ylabel("Pond 1 mean outflow during attack")
ax.scatter(x=0.3, y=o1_uc, zorder=10, color="k", s=6)
ax.scatter(x=0.3, y=o1_c, zorder=10, color="k", s=6, marker="*")
ax.arrow(x=0.3, y=o1_uc, dx=0.5, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Uncontrolled", xy=(0.9, o1_uc-0.05), xytext=(0, 0),
          textcoords='offset points', fontsize=8)
ax.arrow(x=0.3, y=o1_c, dx=0.8, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Controlled", xy=(1.2, o1_c-0.05), xytext=(0, 0),
          textcoords='offset points', fontsize=8)

#%%
fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(6,3))
axes = ax.flatten()
ax = axes[0]
ax.bar(x=0.3, height=in_a_i.max(), tick_label="FDI", 
       width=0.5, color="darkorange")
ax.set_xlim([0,1.4])
ax.set_ylabel("Pond 1 peak inflow\nshown in water level change")
ax.set_yticks([])
ax.scatter(x=0.3, y=in_uc.max(), zorder=10, color="k", s=12)
ax.scatter(x=0.3, y=in_c.max(), zorder=10, color="k", s=12, marker="x")
ax.arrow(x=0.3, y=in_uc.max(), dx=0.35, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Uncontrolled", xy=(0.7, in_uc.max()-0.03), xytext=(0, 0),
          textcoords='offset points', fontsize=10)
ax.arrow(x=0.3, y=in_c.max(), dx=0.5, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Controlled", xy=(0.85, in_c.max()-0.12), xytext=(0, 0),
          textcoords='offset points', fontsize=10)
ax = axes[1]
ax.bar(x=0.3, height=o1_a, tick_label="FDI", 
       width=0.5, color="gold")
ax.set_xlim([0,1.4])
ax.set_yticks([])
ax.set_ylabel("Pond 1 mean outflow during\nattack shown in water level change")
ax.scatter(x=0.3, y=o1_uc, zorder=10, color="k", s=12)
ax.scatter(x=0.3, y=o1_c, zorder=10, color="k", s=12, marker="*")
ax.arrow(x=0.3, y=o1_uc, dx=0.35, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Uncontrolled", xy=(0.7, o1_uc-0.05), xytext=(0, 0),
          textcoords='offset points', fontsize=10)
ax.arrow(x=0.3, y=o1_c, dx=0.5, dy=0, 
         length_includes_head=True, head_width=0.04, zorder=10)
ax.annotate("Controlled", xy=(0.85, o1_c-0.05), xytext=(0, 0),
          textcoords='offset points', fontsize=10)


#%%
# fig, ax = plt.subplots(figsize=(4,3))
# d = sim_y_a_o#/ymax
# ax.plot(d[:,0], label="S1-FDIs", color="cornflowerblue", ls="--")
# d = sim_y_uc#/ymax
# ax.plot(d[:,0], label="S1-uncontrolled", color="k", ls=":", lw=1)
# d = sim_y#/ymax
# ax.plot(d[:,0], label="S1-controlled", color="royalblue")
# ax.plot(d[:,1], label="S2-controlled", color="lightgrey", lw=1)
# ax.plot(d[:,2], label="S3-controlled", color="lightgrey", lw=1)
# ax.set_ylabel("Water level (cm)")
# ax.set_xticks([])
# ax.legend(fontsize=8, loc="lower right", ncol=2)
# plt.show()

#%% MaxWaterLevel
#Too long to solve
iniT = 10
dT = 20
#uya = 100#float('inf')
uYa = [100, 10, 10]
s = 1
noF = ["S{}".format(i+1) for i in range(3) if i+1 != s]
gpm = GPModel(W, lqg, lqe, ss, Ref, uc, name="Hybrid", demo=True)
gpm.set_gp(iniT, dT, uYa=uYa)
gpm.addObj_maxY("S{}".format(s))
gpm.addC_Err(eps=20)
gpm.addC_noflood(noF, tol=1e-5)
gpm.optimize_gp(wd=os.path.join(model_path, "OptExp"), MIPFocus=1, log=False, FeasibilityTol=1e-5, TimeLimit=5*60)

Ya_o = gpm.records["Ya"].T
ss_a, lqg, lqe_a = set_sim(ss_model, sig_V, sig_W, rho_q, rho_r, gw=gw)
ss_a, lqe_a = run_sim(ss_a, lqg, lqe_a, W, Wn=M0, V=M0, Ya=Ya_o, R=Ref, ymax=ymax, uc=uc, control=True, s1_o=None)
sim_y_a, sim_o_a = ss_a.get_ss_record()

#%%
fig, ax = plt.subplots(figsize=(3.5,2))
d = sim_y_a#/ymax
ax.plot(d[:,0], label="FDI", color="cornflowerblue", ls="--")
d = sim_y_uc#/ymax
ax.plot(d[:,0], label="Uncontrolled", color="k", ls=":", lw=1)
d = sim_y#/ymax
ax.plot(d[:,0], label="Controlled", color="royalblue")
ax.axhline(80, lw=0.5, color="red")
ax.set_ylabel("Water level of Pond 1")
ax.set_xlabel("Time")
ax.set_yticks([])
ax.set_xticks([])
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.017, 1.2), ncol=3)
plt.show()
#%%
fig, ax = plt.subplots(figsize=(3.5,2))
d = sim_y_a#/ymax
ax.plot(d[:,0], label="S1-FDIs", color="cornflowerblue", ls="--")
d = sim_y_uc#/ymax
ax.plot(d[:,0], label="S1-uncontrolled", color="k", ls=":", lw=1)
d = sim_y#/ymax
ax.plot(d[:,0], label="S1-controlled", color="royalblue")
ax.plot(d[:,1], label="S2-controlled", color="lightgrey", lw=1)
ax.plot(d[:,2], label="S3-controlled", color="lightgrey", lw=1)

ax.set_ylabel("Water level")
ax.set_xlabel("Time")
ax.set_yticks([])
ax.set_xticks([])
ax.legend(fontsize=8, loc="lower right", ncol=2)
plt.show()

#%%

# # inflow = ss_uc.get_record(ss_uc.Sin_dict["S1"])
# # plt.plot(inflow.T)
# # plt.plot(w1)
# # plt.plot(inflow.T.sum(axis=1))

# # fig, ax = plt.subplots()
# # d = sim_y_uc/ymax
# # for i, dd in enumerate(d.T):
# #     ax.plot(dd, label=i+1)
# # ax.legend()
# #plt.plot(sim_o_uc)


# ref = Ref.T

# #plt.plot(Ref.T)


# # fig, ax = plt.subplots()
# # d = sim_y/ymax
# # for i, dd in enumerate(d.T):
# #     ax.plot(dd, label=i+1)
# # ax.legend()
# # plt.show()

# # #plt.plot(sim_o)
# # inflow = ss.get_record(ss.Sin_dict["S1"])
# # plt.plot(inflow.T)
# # plt.plot(w1)
# # plt.plot(inflow.T.sum(axis=1))




# # fig, ax = plt.subplots()
# # d = sim_o_a_i #sim_y_a_i/ymax
# # for i, dd in enumerate(d.T):
# #     ax.plot(dd, label="Att"+str(i+1))
# # d = sim_o #sim_y/ymax
# # for i, dd in enumerate(d.T):
# #     ax.plot(dd, label=i+1)
# # ax.legend()
# # plt.show()

# # fig, ax = plt.subplots()
# # d = sim_o_a_i #sim_y_a_i/ymax
# # ax.plot(d[:,1]+d[:,2]+w1, label="Attacked")
# # d = sim_o #sim_y/ymax
# # ax.plot(d[:,1]+d[:,2]+w1, label="Healthy")
# # d = sim_o_uc #sim_y/ymax
# # ax.plot(d[:,1]+d[:,2]+w1, label="Uncontrolled")
# # ax.legend()
# # plt.show()




# fig, ax = plt.subplots()
# d = sim_y_a_o/ymax
# for i, dd in enumerate(d.T):
#     ax.plot(dd, label="Att"+str(i+1))
# d = sim_y/ymax
# for i, dd in enumerate(d.T):
#     ax.plot(dd, label=i+1)
# ax.legend()
# plt.show()











