import os
import numpy as np
import pickle
from scipy.sparse import diags, issparse
import gurobipy as gp
from gurobipy import GRB
from lqg_controller import LQG, LQE, SS

def collect_res(pkl_dict, f=None, n=None, strategies=None, show=True):
    for k, v in pkl_dict.items():
        gaps = [round(vv["gap"],4) if vv["status"] != 3 else vv["gap"] for vv in v]
        string = "{}: {} \t Gap: {}%".format(k, [vv["status"] for vv in v], gaps)
        if show:
            print(string, "\n")
        if f is not None:
            f.write(string)
            f.write("\n")
        if n is not None and strategies is not None:
            if None in gaps:
                n.append("{}: {} \t Gap: {}%".format(k, [vv["status"] for vv in v], gaps))
                strategies.append(v[0]["name"])
    if show:
        print("\n")
    if f is not None:
        f.write("\n")
    



class GPModel(object):
    def __init__(self, runoffs, lqg, lqe, ss, Ref, uc, name="attacker", demo=False):
        # Ymax has to be assigned here!!!!!
        self.Ymax = [240.0, 60.0, 180.0, 40.0, 40.0, 91.44, 120.0, 180.0, 110.0]     # [cm]
        if demo:
            self.Ymax = [110,110,110]
        self.uc = list(uc)
        self.Uc = diags(self.uc)
        self.M0 = np.zeros(runoffs.shape)

        self.runoffs = runoffs
        self.lqg = lqg
        self.lqe = lqe
        self.ss = ss
        self.Ref = Ref
        # if V is None:
        #     self.V = self.M0
        # else:
        #     self.V = V
        self.name = name
        self.m = None

        self.nS = lqg.C.shape[0]
        self.nN = lqg.C.shape[1]
        self.Fh = None
        self.wd = os.getcwd()

        # Record
        self.records = {}
        records = self.records
        records["Y"] = np.empty(runoffs.T.shape) * np.nan
        records["U"] = np.empty(runoffs.T.shape) * np.nan
        records["Fh"] = np.empty(runoffs.T.shape) * np.nan
        records["Err"] = np.empty(runoffs.shape[1]) * np.nan
        records["Ya"] = np.empty(runoffs.T.shape) * np.nan

    def set_gp(self, iniT, nT, uYa=0):

        self.iniT = iniT
        self.nT = nT

        lqg = self.lqg
        lqe = self.lqe
        ss = self.ss
        runoffs = self.runoffs
        uc = self.uc
        Ref = self.Ref
        # V = self.V

        ##### Inputs
        nS = self.nS
        nN = self.nN
        A = lqg.A; Bu = lqg.Bu; Bw = lqg.Bw; C = lqg.C
        L = lqg.L; Kk = lqg.Kk; Kr = lqg.Kr; Kw = lqg.Kw
        W = runoffs[:, iniT:(iniT+nT)]
        # if V is None:
        #     V = self.M0[:, iniT:(iniT+nT)]
        R = Ref[:, iniT:(iniT+nT)]
        Ymax = self.Ymax
        Uc = self.Uc

        g = lqe.g                                   # lqe
        X0 = ss.X[iniT].flatten().round(5)          # ss
        Xhat0 = lqe.X_hat[iniT].flatten().round(5)  # lqe

        ##### Variables
        m = gp.Model(self.name, env=gp.Env())

        ##### Bounds (default bounds 0 and infinity are used in gp)
        inf = float('inf')
        lbUaw = np.array([[-Ymax[i] for i in range(nS)]]*nT).T
        lbUuc = np.array([[-uc[i] * Ymax[i]**0.5 for i in range(nS)]]*nT).T
        lbU = np.array([[max(-Ymax[i], -uc[i] * Ymax[i]**0.5) for i in range(nS)]]*nT).T
        ubYsqrt = np.array([[Ymax[i]**0.5 for i in range(nS)]]*nT).T
        ubYact = np.array([Ymax]*nT).T

        # nT+1: to acquire the consequence of the Ya_t
        X = m.addMVar((nN,nT+1), vtype="C", name="X", lb=-inf, ub=inf)
        Xhat = m.addMVar((nN,nT), vtype="C", name="Xhat", lb=-inf, ub=inf)
        Y = m.addMVar((nS,nT), vtype="C", name="Y", lb=-inf, ub=inf)             # CX + Ya
        Yhat = m.addMVar((nS,nT), vtype="C", name="Yhat", lb=-inf, ub=inf)
        Ysim = m.addMVar((nS,nT+1), vtype="C", name="Ysim", lb=-inf, ub=inf)       # CX
        Yact1 = m.addMVar((nS,nT+1), vtype="C", name="Yact1", lb=0, ub=inf)
        Yact = m.addMVar((nS,nT), vtype="C", name="Yact", lb=0, ub=ubYact)
        Ysqrt = m.addMVar((nS,nT), vtype="C", name="Ysqrt", lb=0, ub=ubYsqrt)
        Uini = m.addMVar((nS,nT), vtype="C", name="Uini", lb=-inf, ub=inf)  # have to have b
        Ub = m.addMVar((nS,nT), vtype="C", name="Ub", lb=-inf, ub=0)        # have to have b
        U = m.addMVar((nS,nT), vtype="C", name="U", lb=lbU, ub=0)           # have to have b
        Umax_aw = m.addMVar((nS,nT), vtype="C", name="Umax_aw", lb=lbUaw)   # have to have lb
        Umax_uc = m.addMVar((nS,nT), vtype="C", name="Umax_uc", lb=lbUuc)
        Z = m.addMVar((nS,nT), vtype="C", name="Z", lb=-inf, ub=inf)
        Err = m.addMVar(nT, vtype="C", name="Err", lb=0, ub=inf)

        if isinstance(uYa, list):
            self.YaS = [np.nan if y == 0 else 1 for y in uYa]
            uYa = np.array([[uYa[i] for i in range(nS)]]*nT).T
            print("Set Ya with bounds {} for {} ponds.".format(uYa, nS))
        else:
            print("Set Ya with bound [-{},{}].".format(uYa, uYa))

        Ya = m.addMVar((nS,nT), vtype="C", name="Ya", lb=-uYa, ub=uYa)
        #Ya.VarHintVal = np.array([[0]*nS]*nT).T

        self.X = X; self.Xhat = Xhat
        self.Y = Y; self.Yhat = Yhat; self.Yact = Yact; self.Yact1 = Yact1
        self.U = U; self.Uini = Uini; self.Ub = Ub
        self.Umax_aw = Umax_aw; self.Umax_uc = Umax_uc
        self.Z = Z; self.Err = Err; self.Ya = Ya

        ##### Initial values
        m.addConstr(X[:, 0] == X0, name="cX0")
        m.addConstr(Xhat[:, 0] == Xhat0, name="cXhat0")
        print("Add initial values constraints.")
        m.update()

        ##### Stormwater system
        m.addConstrs((X[:, t] == A @ X[:, t-1] + Bu @ U[:, t-1] + Bw @ W[:, t-1] for t in range(1, nT+1)),
                     name="cX")
        m.addConstrs((Y[:, t] == C @ X[:, t] + Ya[:, t] for t in range(0, nT)),
                     name="cY")     #  + V[:, t]
        m.addConstrs((Ysim[:, t] == C @ X[:, t] for t in range(0, nT+1)),
                     name="cYsim")
        m.addConstrs((Yact1[s,t] == gp.max_(Ysim[s,t], 0) for s in range(nS) for t in range(0, nT+1)),
                     name="cYact1") # Yact1 = max(Ysim, 0)
        m.addConstrs((Yact[s,t] == gp.min_(Yact1[s,t], Ymax[s]) for s in range(nS) for t in range(0, nT)),
                     name="cYact")  # Yact = min(Yact1, Ymax)
        print("Add stormwater system constraints.")

        ##### LQG controller
        m.addConstrs((Xhat[:, t] == A @ Xhat[:, t-1] + Bu @ U[:, t-1] + Bw @ W[:, t-1] + L @ Z[:, t-1] for t in range(1, nT)),
                     name="cXhat")  # LQE
        m.addConstrs((Yhat[:, t] == C @ Xhat[:, t] for t in range(0, nT)),
                     name="cYhat")  # LQE
        m.addConstrs((Z[:, t] == Y[:, t] - Yhat[:, t] for t in range(0, nT)),
                     name="cZ")     # Z = Y - Ysim; Note both Y are not bounded.
        m.addConstrs((Uini[:, t] == - Kk @ Xhat[:, t] + Kr @ R[:, t] + Kw @ W[:, t] for t in range(0, nT)),
                     name="cUini")  # LQR; Can add Wn if needed.
        Pow = [] # Ysqrt = Yact^0.5
        for t in range(nT):
            for s in range(nS):
                Pow.append(m.addGenConstrPow(Yact[s,t], Ysqrt[s,t], 0.5, "uc{}-{}".format(s, t),
                                             "FuncPieces=-1 FuncPieceError=0.0001 FuncPieceRatio=0.5"))
        m.addConstrs((Umax_aw[:, t] == - Yact[:, t] for t in range(0, nT)),
                     name="cUaw")   # Available water.
        m.addConstrs((Umax_uc[:, t] == - Uc @ Ysqrt[:, t] for t in range(0, nT)),
                     name="cUuc")   # Max uncontrolled outflow.
        m.addConstrs((Ub[s,t] == gp.min_(Uini[s,t], 0) for s in range(nS) for t in range(0, nT)),
                     name="cUb")    # Ub = min(Uini, 0)
        m.addConstrs((U[s,t] == gp.max_(Ub[s,t], Umax_aw[s,t], Umax_uc[s,t]) for s in range(nS) for t in range(0, nT)),
                     name="cU")     # U = max(Ub, Umax_aw, Umax_uc)
        print("Add LQG controller constraints.")

        ##### Bad data detector.
        m.addConstrs((Err[t] == sum(Z[s, t] * Z[s, t] * g[s] for s in range(nS)) for t in range(0, nT)),
                     name="cErr")
        print("Add bad data detector constraints.")

        m.update()

        if self.m is None:
            self.m = m
            print("Set gp model (self.m).")
        else:
            self.m = m
            print("Reset gp model (self.m).")

    def addObj_maxY(self, targetS):
        m = self.m
        nT = self.nT
        Yact1 = self.Yact1
        s = int(targetS[1:])-1
        m.setObjective(sum(Yact1[s,t] for t in range(0, nT+1)), GRB.MAXIMIZE)
        m.update()
        print("Add objective to maximize node {}.".format(targetS))

    def addObj_maxInflow(self, targetS):
        m = self.m
        iniT = self.iniT
        nT = self.nT
        W = self.runoffs[:, iniT:(iniT+nT)]
        Ain = self.lqg.Ain_dict[targetS].todense()
        sh = Ain.shape
        s = int(targetS[1:])-1
        tp = np.argmax(W[s,:])
        Xt = self.X[:, tp]
        m.setObjective(sum(Ain[m,n] * Xt[n] for m in range(sh[0]) for n in range(sh[1])), GRB.MAXIMIZE)
        m.update()
        print("Add objective to maximize inflow at node {}.".format(targetS))

    def addC_Err(self, eps):
        self.eps = eps
        m = self.m
        nT = self.nT
        Err = self.Err
        m.addConstrs((Err[t] <= eps for t in range(0, nT)), name="cEPS")
        m.update()
        print("Add cEPS constraints with epsillon={}".format(eps))

    def addC_noflood(self, S_nodes=[], tol=1e-5):
        m = self.m
        nT = self.nT
        Ymax = self.Ymax
        Yact1 = self.Yact1
        for sn in S_nodes:
            si = int(sn[1:])-1
            m.addConstrs((Yact1[si, t] <= Ymax[si]+tol for t in range(0, nT)),
                         name="cNF_{}".format(sn))
            print(sn, "  ", Ymax[si])
        m.update()
        print("Add no flood constraints for {}".format(S_nodes))

    def load_start_points(self, optimal_m, skip=[]):
        def check(x): # Ture if pass
            return(any([v in x for v in skip]) == False)
        m = self.m
        m.NumStart = 1
        m.update()
        # iterate over all MIP starts
        for i in range(m.NumStart):
            print("Add start point set {}.".format(i))
            m.params.StartNumber = i  # set StartNumber
            # now set MIP start values using the Start attribute, e.g.:
            for v in optimal_m.getVars():
                var_name = v.varName
                if check(var_name):
                    try:
                        var = m.getVarByName(var_name)
                        var.Start = v.x
                        # m.update()
                    except:
                        print("Var {} is not exist in new model.".format(var_name))
                else:
                    print("Skip {}.".format(var_name))
        m.update()
        print("Load start points.")

    def cal_Fh(self, tol=1e-5):
        Fh = np.zeros((self.nS, self.nT+1))
        fh = self.Yact1.x - np.array(self.Ymax).reshape(-1,1)
        Fh[np.where(fh>0+tol)] = 1
        self.Fh = Fh
        return Fh

    def add_start_Ya(self):
        Ya = self.Ya; nS = self.nS; nT = self.nT
        Ya.Start = np.array([[0]*nS]*nT).T
        self.m.update()

    def update_gp(self, m):
        m.update()
        self.m = m

    def optimize_gp(self, wd, DualReductions=1, MIPFocus=1, TimeLimit=float('inf'), FeasibilityTol=1e-5, log=True):
        #StartNodeLimit=2**20, PreSparsify=-1, NoRelHeurTime=0, ImproveStartTime=float('inf'),
        StartNodeLimit=2**20
        name = self.name
        m = self.m
        self.wd = wd
        self.DualReductions = DualReductions
        self.MIPFocus = MIPFocus
        self.TimeLimit = TimeLimit
        if log:
            LogFile = os.path.join(wd, name+".log")
        else:
            LogFile = ""
        self.LogFile = LogFile

        m.setParam('NonConvex', 2)  # Needed for Q contraints.
        m.setParam('Presolve', 2)   # off (0), conservative (1), or aggressive (2).
        m.setParam('DualReductions', DualReductions) # =0 if status is 5.
        m.setParam('MIPFocus', MIPFocus)  # 0: solver default; 1: feasible solutions; 2: optimality; 3: bound
        m.setParam('TimeLimit', TimeLimit)
        #m.setParam("StartNodeLimit", StartNodeLimit)
        m.setParam('FeasibilityTol', FeasibilityTol)
        #m.setParam('PreSparsify', PreSparsify)
        #m.setParam('NoRelHeurTime', NoRelHeurTime)
        #m.setParam('ImproveStartTime', ImproveStartTime)
        m.setParam('LogFile', LogFile)
        m.write(os.path.join(wd, '{}.lp'.format(name)))
        m.optimize()

        status = m.Status
        self.status = status
        self.runtime = m.Runtime
        self.bestObj = None
        self.bestBound = None
        self.gap = None

        def save():
            self.bestObj = m.ObjVal
            self.bestBound = m.ObjBound
            self.gap = m.MIPgap
            print('The optimal objective is %g' % m.ObjVal)
            iniT = self.iniT; nT = self.nT
            m.write(os.path.join(wd, '{}_{}-{}.sol'.format(name, iniT, nT)))
            # Record
            records = self.records
            records["Y"][iniT:(iniT+nT+1), :] = self.Yact1.x.T
            records["U"][iniT:(iniT+nT), :] = self.U.x.T
            records["Fh"][iniT:(iniT+nT+1), :] = self.cal_Fh().T
            records["Err"][iniT:(iniT+nT)] = self.Err.x
            records["Ya"][iniT:(iniT+nT), :] = self.Ya.x.T
            for i, v in enumerate(self.YaS):
                records["Ya"][:, i] = records["Ya"][:, i] * v

        if status == GRB.OPTIMAL:
            save()
        elif status == GRB.TIME_LIMIT:
            print('Optimization was stopped with status 9 (reach time limit).')
            try:
                save()
            except:
                self.note = "No feasible solution is found given time limit."
        elif status == GRB.INFEASIBLE:
            print('Infeasible')
        elif status == GRB.INF_OR_UNBD:
            print("Infeasible or unbounded. Please resolve with DualReductions = 0.")
        else:
            print('Optimization was stopped with status %d' % status)
        self.m = m

    def get_pickable_dict(self):
        """Auto remove unpickable items"""
        pkdict = {}
        d = self.__dict__
        for k, v in d.items():
            if isinstance(v, (int, list, str, float, dict, np.ndarray,
                              np.generic, type(None), LQG, LQE, SS)):
                pkdict[k] = v
            else:
                try:
                    if issparse(v):
                        pkdict[k] = v
                    continue
                except:
                    pass

                try:
                    sol = v.x
                    if isinstance(sol, (int, list, str, float, np.ndarray, type(None))):
                        pkdict[k] = sol
                except:
                    pkdict[k] = "unpickable"
        return pkdict
    def pickle(self, filename=None):
        if filename is None:
            filename = os.path.join(self.wd, self.name)
        if ".pickle" not in filename:
            filename += ".pickle"
        pkdict = self.get_pickable_dict()
        with open(filename, "wb") as f:
            pickle.dump(pkdict, f)

    @staticmethod
    def unpickle(filename):
        with open(filename, "rb") as f:
            pkdict = pickle.load(f)
        return pkdict

    def do_IIS_gp(self):
        """Analyzing infeasible model."""
        m = self.m
        # do IIS
        print('The model is infeasible; computing IIS')
        m.computeIIS()
        if m.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('\nThe following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.ConstrName)
        m.write(os.path.join(self.wd, '{}.ilp'.format(self.name)))


    # def addC_floodi(self, S_nodes=[]):
    #     if self.Fh is None:
    #         m = self.m
    #         nT = self.nT
    #         Ymax = self.Ymax
    #         Yact1 = self.Yact1
    #         Fh = m.addMVar((len(S_nodes),nT), vtype="B", name="Fh", lb=0, ub=1)
    #         self.Fh = Fh
    #         for i, sn in enumerate(S_nodes):
    #             s = int(sn[1:])-1
    #             for t in range(nT):
    #                 m.addGenConstrIndicator(Fh[i,t], True, Yact1[s,t] >= Ymax[s],
    #                                         name="Fh[{},{}]".format(s,t))
    #         m.update()
    #         print("Add flooding indicators for {}".format(S_nodes))
    #     else:
    #         print("Error: Flooding indicators already exist.")