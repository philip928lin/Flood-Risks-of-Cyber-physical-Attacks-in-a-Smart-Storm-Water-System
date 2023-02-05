import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# Err gap plot
dof_o = 9
dof_a = 8
s = 0.25**2
q_o = 0.95
q_a = 0.99
q_a2 = 0.85

fig, ax = plt.subplots()

rv = chi2(df=dof_o, scale=s)
x = np.linspace(chi2.ppf(0.000001, df=dof_o, scale=s),
                chi2.ppf(0.9999999, df=dof_o, scale=s), 100000)
ax.plot(x, rv.pdf(x), color="orange", lw=2, label="Operator $\epsilon$ ($n$="+str(dof_o)+", $p$="+str(q_o)+")")

rv = chi2(df=dof_a, scale=s)
x = np.linspace(chi2.ppf(0.000001, df=dof_a, scale=s),
                chi2.ppf(0.9999999, df=dof_a, scale=s), 100000)
y = rv.pdf(x)
y[y>2] = np.nan
ax.plot(x, y, color="royalblue", lw=2, label="Attacker $\epsilon^a$ ($n^a$="+str(dof_o-dof_a)+", $p^a$="+str(q_a)+")")

# for legend
rv = chi2(df=dof_a, scale=s)
x = np.linspace(chi2.ppf(0.000001, df=dof_a, scale=s),
                chi2.ppf(0.9999999, df=dof_a, scale=s), 100000)
y = rv.pdf(x)
y[y>2] = np.nan
ax.plot(x, y, color="lightblue", ls="--", lw=2, label="Attacker $\epsilon^a$ ($n^a$="+str(dof_o-dof_a)+", $p^a$="+str(q_a2)+")")
ax.plot(x, y, color="royalblue", lw=2)


thres_o = chi2.ppf(q=q_o**(1/1440), df=dof_o, scale=s)
ax.axvline(thres_o, color='orange', lw=2)
ax.axhline(0, color='k', ls="--", lw=0.5)
nT_list = [i for i in range(5, 65, 5)]
nT_list = [5, 10, None, 20, None, 30, None, None, 45, None, None, 60]
nT_list = [60, None, None, 45, None, None, 30, None, 20, None, 10, 5]
for i, nT in enumerate(nT_list):
    if nT is None:
        continue
    b = chi2.ppf(q=q_a2**(1/nT), df=dof_a, scale=s)
    #ax.axvline(b, color='limegreen', lw=1, alpha=0.3+0.7/len(nT_list)*i, zorder=-1)
    h = 0.5+1.5/len(nT_list)*i
    ax.vlines(b, ymax=h, ymin=-0.1, color='lightblue', lw=1, alpha=0.3+0.7/len(nT_list)*i, zorder=-1, ls="--")
    ax.arrow(x=b, y=h, dx=(thres_o-b), dy=0,
              length_includes_head=True, head_width=0.03, zorder=-10, color="lightgrey")
    ax.arrow(x=thres_o, y=h, dx=-(thres_o-b), dy=0,
              length_includes_head=True, head_width=0.03, zorder=-10, color="lightgrey")

    # ax.annotate(round(thres_o-b, 2), xy=(thres_o, h), xytext=(-30, 1),
    #           textcoords='offset points', fontsize=7)
    # ax.annotate("$T^a$ = {} mins".format(nT), xy=(thres_o, h), xytext=(10, 0),
    #           textcoords='offset points', fontsize=8)

for i, nT in enumerate(nT_list):
    if nT is None:
        continue
    b = chi2.ppf(q=q_a**(1/nT), df=dof_a, scale=s)
    #ax.axvline(b, color='royalblue', lw=1, alpha=0.3+0.7/len(nT_list)*i, zorder=-1)
    h = 0.5+1.5/len(nT_list)*i
    ax.vlines(b, ymax=h, ymin=-0.1, color='royalblue', lw=1, alpha=0.3+0.7/len(nT_list)*i, zorder=-1)
    ax.arrow(x=b, y=h, dx=(thres_o-b), dy=0,
             length_includes_head=True, head_width=0.03, zorder=10, color="k")
    ax.arrow(x=thres_o, y=h, dx=-(thres_o-b), dy=0,
             length_includes_head=True, head_width=0.03, zorder=10, color="k")

    ax.annotate(round(thres_o-b, 2), xy=(thres_o-(thres_o-b)/2, h+0.02), xytext=(thres_o-(thres_o-b)/2, h+0.02),
              textcoords='data', fontsize=9, ha='center')
    
    ax.annotate("$T^a$ = {} mins".format(nT), xy=(thres_o, h), xytext=(10, 0),
              textcoords='offset points', fontsize=9)
    
ax.text(0.7, 0.2, '$Err^a$', transform=ax.transAxes, fontsize=12, 
        bbox={'facecolor':'white', 'alpha':0.8, 'pad':3, 'edgecolor':'none'})

ax.set_ylabel("PDF of $\chi^2_{dof}$ distribution", fontsize=12)
ax.set_xlabel("$Err_t=z^T_t\cdot\mathscr{G}^{-1}\cdot z_t$", fontsize=12)
ax.set_ylim([-0.1, 2])
ax.set_xlim([0, 2.9])

# for legend
(lines, labels) = plt.gca().get_legend_handles_labels()
#it's safer to use linestyle='none' and marker='none' that setting the color to white
#should be invisible whatever is the background
lines.insert(1, plt.Line2D(x, x, linestyle=None, marker=None, alpha=0))
labels.insert(1,'')
ax.legend(lines,labels, ncol=2, fontsize=9, loc="upper right", bbox_to_anchor=(1.01, 1.18))
#%%
import os
prj_path = r""
model_path = os.path.join(prj_path, "Model")
os.chdir(r"")
import pickle
from design_storm import ddf, return_year, duration
#[Duration, Return period]
x = [0, 7, 12, 17, 22, 30, 34, 42, 50,
     58,66,71,75,81,84,90,94,98,101]
fig, ax = plt.subplots(figsize=(3,2.5))
plt.set_cmap("Paired")
ax.plot(x,ddf, label=return_year)
ax.set_xlim([0,101])
ax.set_xticks(x)
ax.set_xticklabels(duration, rotation=90, fontsize=7)
ax.set_xlabel("Duration", fontsize=10)
ax.set_ylabel("Depth (cm)")
ax.legend(title="Return period (years)", fontsize=7, ncol=2)

#%%
strategy = "1N30M990CI25yr"
with open(os.path.join(model_path, "OptFDI", "MC_{}.pickle".format(strategy)), "rb") as f:
    sim_y_list, sim_o_list, flood_list, err_list = pickle.load(f)
with open(os.path.join(model_path, "OptFDI", "FDIN1_{}.pickle".format(strategy)), "rb") as f:
    all_Y_d, dY, Err, all_Fh_d, sim_y, sim_y_uc = pickle.load(f)

fig, ax = plt.subplots(figsize=(3,2.5))
ymax = np.array([240.0, 60.0, 180.0, 40.0, 40.0, 91.44, 120.0, 180.0, 110.0])
ls_list = ["--", "-"] #[":", "--", "-", "-"]
c_list = [5, 1]#[9, 5, 1, 2, 3, 6, 8]
data_list = [sim_y, all_Y_d]#[sim_y_uc, sim_y, all_Y_d]
stoch_data_list = sim_y_list
labels = ["Controlled", "FDI"]#["Uncontrolled", "Controlled", "FDI"]
data_list = [d/ymax*100 if d is not None else None for d in data_list]
stoch_data_list = [d/ymax*100 if d is not None else None for d in stoch_data_list]
seg = [0, 759]
v = 9
# Add stochastic simulations.
for i, d in enumerate(stoch_data_list):
    x = np.arange(seg[0],seg[0]+d.shape[0])
    if i == 0:
        ax.plot(x[seg[0]:seg[1]], d[seg[0]:seg[1],v-1], lw=1, color="darkgray", alpha=0.8,
                zorder=-1)
    ax.plot(x[seg[0]:seg[1]], d[seg[0]:seg[1],v-1], lw=0.2, color="darkgray", alpha=0.3, zorder=-1)

for i, d in enumerate(data_list):
    if d is None:
        continue
    x = np.arange(seg[0],seg[0]+d.shape[0])
    lw = 1    
    if i >= 1:
        tp, nT = (728,30)
        ax.plot(x[:tp+nT-seg[0]+1], d[:tp+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
    else:
        ax.plot(x, d[:,v-1], label=labels[i], lw=lw,
                ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)

locs = ax.get_position()
x_, y_, w_, h_ = locs.x0, locs.y0, locs.width, locs.height
axx = fig.add_axes([x_+w_/15 , y_+h_/3 , w_/3, h_*1/2])

for i, d in enumerate(stoch_data_list):
    x = np.arange(seg[0],seg[0]+d.shape[0])
    x = x[tp+1:tp+nT-seg[0]+1]
    axx.plot(x, d[tp+1:tp+nT-seg[0]+1,v-1], lw=0.2,
            color="darkgray", alpha=0.3, zorder=-1)
for i, d in enumerate(data_list):
    if d is None:
        continue
    x = np.arange(seg[0],seg[0]+d.shape[0])
    x = x[tp+1:tp+nT-seg[0]+1]
    if i >= 1:
        tp, nT = (728,30)
        axx.plot(x, d[tp+1:tp+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
    else:
        axx.plot(x, d[tp+1:tp+nT-seg[0]+1,v-1], label=labels[i], lw=lw,
                ls=ls_list[i], color="C{}".format(c_list[i]), zorder=10-i)
    axx.axhline(100, color="black", ls="dashed", lw=0.5)
    axx.set_yticks([])
    axx.set_ylim([75,125])
    axx.set_xlim([x[0],x[-1]])
    axx.set_xticks([x[0], x[14], x[-1]])
    axx.set_xticklabels([x[0], x[14], x[-1]], rotation=45, fontsize=6)
    axx.set_yticks([105])
    axx.set_yticklabels(["100"], rotation=90, fontsize=6)
    axx.tick_params(axis="y", length=0, pad=0.3)
    axx.tick_params(axis="x", pad=1)

ax.set_xlim([0, 1439])
ax.axhline(100, color="black", ls="dashed", lw=0.5)
ax.legend(ncol=4, fontsize=7, loc="upper right", bbox_to_anchor=(1.02, 1.13))
ax.set_xlabel("Time (minute)")
ax.set_ylabel("Water level (%)")

#%%

err = 0.46
fig, ax = plt.subplots(figsize=(6,6))
rv = chi2(df=dof_a, scale=s)
x = np.linspace(chi2.ppf(0.000001, df=dof_a, scale=s),
                chi2.ppf(0.999999, df=dof_a, scale=s), 100000)
y = rv.pdf(x)
y[y>2] = np.nan
ax.plot(x, y, 'k-', lw=1)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='$\chi^2_{dof=n-n_c}$')
x_ = x[x<=1.8]
ax.fill_between(x_, rv.pdf(x_), color="grey", alpha=0.5)
rv = chi2(df=dof_a, loc=err, scale=s)
x = np.linspace(chi2.ppf(0.000001, df=dof_a, loc=err, scale=s),
                chi2.ppf(0.999999, df=dof_a, loc=err, scale=s), 100000)
y = rv.pdf(x)
y[y>2] = np.nan
ax.plot(x, y, color='royalblue', lw=2, label='$\chi^2_{dof=n-n_c}$+ shift')
x_ = x[x<=1.65]
ax.fill_between(x_, rv.pdf(x_), color="royalblue", alpha=0.5)
ax.axvline(1.65, color='orange', lw=2, label="Operator $\epsilon$")
ax.axhline(0, color='k', ls="--", lw=0.5)
ax.set_ylim([-0.1, 2])
ax.set_xlim([0, 2.9])

ax.arrow(x=x[np.where(y==max(y))[0][0]]-err, y=max(y)+0.03, dx=err, dy=0,
         length_includes_head=True, head_width=0.03, zorder=10)
ax.arrow(x=x[np.where(y==max(y))[0][0]], y=max(y)+0.03, dx=-err, dy=0,
         length_includes_head=True, head_width=0.03, zorder=10)
ax.annotate("$Err_t$", xy=(x[np.where(y==max(y))[0][0]]-err/2, max(y)+0.07), xytext=(-9, 0),
          textcoords='offset points', fontsize=14)
ax.legend(ncol=3, fontsize=11, loc="upper right", bbox_to_anchor=(1.01, 1.1))
ax.set_ylabel("PDF of $\chi^2_{dof}$ distribution", fontsize=14)
ax.set_xlabel("$Err_t=z^T_t\cdot\mathscr{G}^{-1}\cdot z_t$", fontsize=14)
ax.set_xticks([])
