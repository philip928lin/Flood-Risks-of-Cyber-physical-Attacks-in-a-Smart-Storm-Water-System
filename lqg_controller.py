from scipy.linalg import solve_discrete_are, eigvals, solve
from scipy.sparse import dok_matrix, csr_matrix, issparse
from scipy.stats import lognorm, norm, truncnorm, chi2
import numpy as np
from numpy.linalg import inv, pinv

def cal_prob_fdi(Err, q_o=0.95, dof_o=9, dof_a=8, s=0.25**2, roun=8):
    thre = chi2.ppf(q=q_o**(1/1440), df=dof_o, scale=s)
    prob = 1
    for err in Err:
        #prob = prob * chi2.cdf(x=thre, df=df, loc=err, scale=s)
        prob = prob * chi2.cdf(x=thre-err, df=dof_a, scale=s)
    print("Prob of successful {} FDIs = {:0.3f} given q_o = {:0.2f} and s = {:0.3f}".format(len(Err), prob, q_o, s))
    return round(prob, roun)

def gen_Wn(W, s, m=1, alpha=None, dist="lognorm", size=1, rngen=None):
    """
    Sythesize forecast info.
    W' = 1/beta * W * 1/e
    Wn = (1/(beta*e) - 1) * W
    Wn = (alpha/e - 1) * W

    Parameters
    ----------
    W : 1darray
        Runoffs (s, t).
    s : float
        Sigma.
    m : float, optional
        Median or Mean. The default is 1.
    alpha : float, optional
        Discount scale. The default is None.
    dist : str, optional
        lognorm or norm. The default is "lognorm".
    size : int, optional
        Size. The default is 1.

    Returns
    -------
    tuple
        Wn_dict, std.

    """
    if dist == "lognorm":
        Wn_ = (alpha / lognorm.rvs(s=s, loc=0, scale=m, size=(size, W.shape[1]), random_state=rngen) - 1)
    elif dist == "normal":
        Wn_ = (alpha / norm.rvs(loc=m, scale=s, size=(size, W.shape[1]), random_state=rngen) - 1)

    Wn_dict = {}
    std_list = []
    for i in range(size):
        Wn = W * Wn_[i,:]
        Wn_dict[i] = Wn
        std_list.append(np.std(Wn, axis=1))
    std = np.mean(std_list, axis=0).round(5)
    print("Gen Wn with {}(m={}, s={}) and alpha={}.\nAvg_Wn_std = {}".format(dist, m, s, alpha, std))
    return Wn_dict, std

def gen_V(W, s, m=0, size=1, dist="normal", rngen=None):
    V_dict = {}
    for i in range(size):
        if dist == "normal":
            V = norm.rvs(loc=m, scale=s, size=W.shape, random_state=rngen)
        elif dist == "truncnorm":
            V = truncnorm.rvs(a=-3*s, b=3*s, loc=0, scale=s, size=W.shape, random_state=rngen)
        V_dict[i] = V
    print("Gen V with {}(m={}, s={}).".format(dist, m, s))
    return V_dict

def gen_noises(W, sv, sw, m=1, alpha=1, dist=("normal","lognorm"), size=1, rngen=None):
    V_dict = gen_V(W, sv, 0, size, dist[0], rngen)
    Wn_dict, std_w = gen_Wn(W, sw, m, alpha, dist[1], size, rngen)
    return V_dict, Wn_dict, std_w

class LQG(object):
    def __init__(self, A, Bu, Bw, C, Cov_V, Cov_W, rho_q=1, rho_r=1, roun=5,
                 Q=None, R=None, Ain_dict=None):
        """
        Solving a steady state lqg controller for time-invariant state-space
        model shown below.

        X(k+1) = A * X(k) + Bu * U(k) + Bw * W(k)
        Y(k+1) = C * X(k) + V(k)

        User might apply Kalman decomposition to seperate controllable and
        uncontrollable parts.

        Parameters
        ----------
        A : 2darray
            State matrix (observable & controllable).
        Bu : 2darray
            Control matrix (controllable).
        Bw : 2darray
            Disturbance matrix (controllable).
        C : 2darray
            Output matric (observable).
        Cov_V : 2darray
            Covariance of sensor errors.
        Cov_W : 2darray
            Covariance of disturbances (For solving Kalman gain, we use
            cov = Bw cov_W Bw.T).
        rho_q : float or 1darray
            Weight of states cost.
        rho_r : float or 1darray
            Weight of control cost.
        Q : 2darray
            State error control weight matrix Ex: C.T Wx C.
        R : 2darray
            Control cost weight matrix Ex: Wu.
        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        self.dense_org = {"A": self.sparse2dense(A),
                          "Bu": self.sparse2dense(Bu),
                          "Bw": self.sparse2dense(Bw),
                          "C": self.sparse2dense(C),
                          "Cov_V": self.sparse2dense(Cov_V),
                          "Cov_W": self.sparse2dense(Cov_W)}

        # To ensure all matrices are dense for other calculation.
        self.A = self.dok_to_csr(A, roun, "A")
        self.Bu = self.dok_to_csr(Bu, roun, "Bu")
        self.Bw = self.dok_to_csr(Bw, roun, "Bw")
        self.C = self.dok_to_csr(C, roun, "C")
        self.Cov_V = self.dok_to_csr(Cov_V, roun, "Cov_V")
        self.Cov_W = self.dok_to_csr(Cov_W, roun, "Cov_W")
        self.roun = roun
        self.rho_q = rho_q
        self.rho_r = rho_r
        self.Ain_dict = {k: self.dok_to_csr(v, roun, "Ain{}".format(k)) for k, v in Ain_dict.items()}

        if Q is not None:
            self.Q = self.dok_to_csr(Q, None, "Q")
            print("rho_q is not applied.")
        else:
            if isinstance(rho_q, list) or isinstance(rho_q, np.ndarray):
                self.Q = self.dok_to_csr((C.T @ np.diag(rho_q) @ C) , None, "Q")
            else:
                self.Q = self.dok_to_csr((C.T @ C) * rho_q, None, "Q")
        self.dense_org["Q"] = self.sparse2dense(self.Q)
        self.Q = self.dok_to_csr(self.Q , roun, "Q")

        if R is not None:
            self.R = self.dok_to_csr(R, None, "R")
            print("rho_r is not applied.")
        else:
            if isinstance(rho_q, list) or isinstance(rho_q, np.ndarray):
                self.R = np.eye(self.Bu.shape[1]) @ np.diag(rho_r)
            else:
                self.R = np.eye(self.Bu.shape[1]) * rho_r
        self.dense_org["R"] = self.sparse2dense(self.R)
        self.R = self.dok_to_csr(self.R, roun, "R")

        try:
            self.feedback_gain_dare()
            print("Done Kk.")
        except Exception as e:
            print(e)
            print("Kk is not done.")

        try:
            self.kalman_gain_dare()
            print("Done L.")
        except Exception as e:
            print(e)
            print("L is not done.")

        try:
            self.get_feedforward_dist_cancellation_Kw()
            print("Done Kw.")
        except Exception as e:
            print(e)
            print("Kw is not done.")

        try:
            self.get_feedforward_ref_tracking_Kr()
            print("Done Kr.")
        except Exception as e:
            print(e)
            print("Kr is not done.")

        # Save dense rounded matrix
        self.dense_roun = {}
        for k, v in self.dense_org.items():
            try:
                self.dense_roun[k] = self.sparse2dense(eval("self."+k))
            except Exception as e:
                pass  # ignore xi_P and xi_X

    def sparse2dense(self, matrix):
        if issparse(matrix):
            return matrix.todense()
        else:
            return matrix.copy()

    def dok_to_csr(self, matrix, roun=None, name=None):
        if matrix is None:
            return None

        if issparse(matrix) is False:
            print("Auto-convert to csr sparse matrix.")
            if roun is None:
                m = matrix
            else:
                m = matrix.round(roun)                  # round matrix
            m = csr_matrix(m)
            # if roun is not None:
            #     m.data = np.round(m.data, roun)   # round matrix
            shape = m.get_shape()
            print("{}: nnz = {}, dim = {}, % = {}.\n{}".format(
                name, m.getnnz(), shape,
                round(m.getnnz()/(shape[0]*shape[1])*100, 0), m.data))
        else:
            m = matrix.tocsr()
            if roun is not None:
                m.data = np.round(m.data, roun)     # round matrix
        return m

    def feedback_gain_dare(self):
        """
        Solve feedback gain (Kk) for steady state time invariant system.

        Riccati difference equation (solve for steady state):
            xi_P = A^T [xi_P - xi_P Bu (Bu^T xi_P Bu + R)^{-1} Bu^T xi_P] A + Q
        Feedback gain:
            Kk = (Bu^T xi_P Bu + R)^(-1) Bu^T xi_P A

        Returns
        -------
        xi_P : 2darray
            Covariance matrix.
        Kk : 2darray
            Feedback gain.

        """
        A = self.dense_org["A"]
        Bu = self.dense_org["Bu"]
        Q = self.dense_org["Q"]
        R = self.dense_org["R"]
        # Feedback gain for steady state time invariant system.
        # Steady state covariance matrix solved by dare.
        xi_P = solve_discrete_are(A, Bu, Q, R)
        Kk = solve(Bu.T @ xi_P @ Bu + R, Bu.T @ xi_P @ A)
        # check stability for discrete sys.
        # eig = eigvals(np.eye(A.shape[0]) - A + Bu @ Kk)
        self.dense_org["Kk"] = Kk
        self.Kk = self.dok_to_csr(Kk, self.roun, "Kk")
        self.dense_org["xi_P"] = xi_P
        return xi_P, Kk

    def kalman_gain_dare(self):
        """
        Solve Kalman gain (L) for steady state time invariant system.

        Riccati difference equation (solve for steady state):
            xi_X = A [xi_X - xi_X C^T (C xi_X C^T + V)^(-1) C xi_X] A^T
                   + Bw cov_W Bw.T
        Kalman gain:
            L = xi_X C^T (C xi_X C^T + V)^(-1)

        Note: For disturbance new cov = Bw cov_W Bw.T
        Returns
        -------
        xi_X : 2darray
            Covariance matrix.
        L : 2darray
            Kalman gain.

        """
        # A has to be observable and controllable.
        A = self.dense_org["A"]
        C = self.dense_org["C"]
        Bw = self.dense_org["Bw"]
        Cov_W = self.dense_org["Cov_W"]
        Cov_V = self.dense_org["Cov_V"]

        # Steady state covariance matrix solved by dare.
        xi_X = solve_discrete_are(A.T, C.T, Bw @ Cov_W @ Bw.T, Cov_V)

        # Note that x' A = b' is equivalent to A' x = b
        # Solve for L = xi_X C' (C xi_X C' + cov_V)^(-1)
        # Kalman gain for steady state time invariant system.
        L = solve( (C.dot(xi_X).dot(C.T) + Cov_V).T, C.dot(xi_X.T) ).T
        # eig =
        self.dense_org["L"] = L
        self.L = self.dok_to_csr(L, self.roun, "L")
        self.dense_org["xi_X"] = xi_X
        return xi_X, L#, eig

    def get_feedforward_dist_cancellation_Kw(self, sys="discrete"):
        """
        Calculate feedforward disturbance cancellation matrix.

        # Control to Y = 0.
        For discrete system,
        Kw = -[C(I-A+BuKk)^(-1)Bu]^(-1) C(I-A+BuKk)^(-1)Bw
        Note: (I-A+BuKk) has to be square and invertable.

        For continuous system,
        Kw = -[C(A-BuKk)^(-1)Bu]^(-1) C(A-BuKk)^(-1)Bw
        Note: (I-A+BuKk) has to be square and invertable.

        Parameters
        ----------
        sys : str, optional
            "discrete" or "continuous" system. The default is "discrete".

        Returns
        -------
        Kw : 2darray
        """
        A = self.dense_org["A"]
        Bu = self.dense_org["Bu"]
        Bw = self.dense_org["Bw"]
        C = self.dense_org["C"]
        try:
            Kk = self.dense_org["Kk"]
        except Exception as e:
            print("Cannot find Kk. So we solve it using feedback_gain_dare")
            self.feedback_gain_dare()
            Kk = self.dense_org["Kk"]

        # Maybe use sparse matrix in the future.
        if sys == "discrete":
            F = inv(np.eye(A.shape[0]) - A + Bu @ Kk)
        elif sys == "continuous":
            F = inv(A - Bu @ Kk)

        Kw = solve(-C @ F @ Bu, C @ F @ Bw)

        self.dense_org["Kw"] = Kw
        self.Kw = self.dok_to_csr(Kw, self.roun, "Kw")
        return Kw

    def get_feedforward_ref_tracking_Kr(self, sys="discrete"):
        """
        Calculate feedforward reference tracking matrix (dc-gain).

        # Control to Y = r.
        For discrete system,
        Kr = [C (I - A + Bu Kk)^(-1) Bu ]^(-1)
        Note: Eqn above has to be square and invertable.

        For continuous system,
        Kr = -[ C (A - Bu Kk)^(-1) Bu]^(-1)
        Note: Eqn above has to be square and invertable.

        Parameters
        ----------
        sys : str, optional
            "discrete" or "continuous" system. The default is "discrete".

        Returns
        -------
        Kr : 2darray
        """
        A = self.dense_org["A"]
        Bu = self.dense_org["Bu"]
        C = self.dense_org["C"]
        try:
            Kk = self.dense_org["Kk"]
        except Exception as e:
            print("Cannot find Kk. So we solve it using feedback_gain_dare")
            self.feedback_gain_dare()
            Kk = self.dense_org["Kk"]

        if sys == "discrete":
            F = inv(np.eye(A.shape[0]) - A + Bu @ Kk)
            # This is wrong
            #else:
            #    F = inv(np.eye(A.shape[0]) - A + Bu @ Kk - Bu @ Kw)
            Kr = inv(C @ F @ Bu)
        elif sys == "continuous":
            F = inv(A - Bu @ Kk)
            Kr = -inv(C @ F @ Bu)
        self.dense_org["Kr"] = Kr
        self.Kr = self.dok_to_csr(Kr, self.roun, "Kr")
        return Kr

class LQE(object):
    def __init__(self, A, Bu, Bw, C, L, X0, c_scale=1, g=None, gw=1, Ain_dict=None):

        # Matrices should be dok sparse matrices
        self.A = A.tocsr()
        self.Bu = Bu.tocsr()
        self.Bw = Bw.tocsr() # We assume a perfect forecast
        self.C = C.tocsr()
        self.L = L
        self.c_scale = c_scale
        self.Ain_dict = Ain_dict

        if g is None:
            self.g = np.array([gw]*self.C.shape[0])
            self.G = csr_matrix(np.eye(self.C.shape[0]) * gw)
        else:
            self.G = g

        # Store values
        self.X_hat = [X0]
        self.Y_hat = [(C @ X0)*c_scale]
        self.U = [np.zeros((Bu.shape[1],1))]
        self.measurement_error = [np.zeros((C.shape[0],1))]
        self.eps = [0]
        if Ain_dict is not None:
            self.Sin_dict = {k: [v @ X0] for k, v in Ain_dict.items() if v is not None}

    def run_step(self, Y, U, W, Err=None):
        A = self.A
        Bu = self.Bu
        Bw = self.Bw
        C = self.C
        L = self.L
        G = self.G
        c_scale = self.c_scale
        Ain_dict = self.Ain_dict

        # Get last estimates
        X_hat = self.X_hat[-1]
        Y_hat = self.Y_hat[-1]

        # LQE
        if Err is None:
            error_no_scale = (Y - Y_hat)
        else:
            error_no_scale = Err
        error = error_no_scale * c_scale
        self.eps.append( (error_no_scale.T @ G @ error_no_scale)[0,0] )
        X_hat = A @ X_hat + Bu @ U + Bw @ W + L @ (error)
        Y_hat = (C @ X_hat)/c_scale

        # Store
        self.X_hat.append(X_hat)
        self.Y_hat.append(Y_hat)
        self.U.append(U)
        self.measurement_error.append(error_no_scale)
        if Ain_dict is not None:
            for k, v in Ain_dict.items():
                if v is not None:
                    self.Sin_dict[k].append(v @ X_hat)

        return X_hat

    @staticmethod
    def get_record(X):
        if isinstance(X, dict):
            return {k: np.array(np.hstack(v)) for k, v in X.items()}
        else:
            return np.array(np.hstack(X))

class SS(object):
    def __init__(self, A, Bu, Bw, C, X0, c_scale=1, Ain_dict=None):

        # Matrices should be dok sparse matrices
        self.A = A.tocsr()
        self.Bu = Bu.tocsr()
        self.Bw = Bw.tocsr() # We assume a perfect forecast
        self.C = C.tocsr()
        self.c_scale = c_scale
        self.Ain_dict = Ain_dict

        # Store values
        self.X = [X0]
        self.Y = [(C @ X0) * c_scale]
        self.Y_act = [(C @ X0) * c_scale]
        self.U = [np.zeros((Bu.shape[1],1))]
        if Ain_dict is not None:
            self.Sin_dict = {k: [v @ X0] for k, v in Ain_dict.items() if v is not None}

    def run_step(self, U, W, V=None):
        A = self.A
        Bu = self.Bu
        Bw = self.Bw
        C = self.C
        c_scale = self.c_scale # scale yh
        # Get last estimates
        X = self.X[-1]
        Ain_dict = self.Ain_dict

        # LQE
        X = A @ X + Bu @ U + Bw @ W
        if V is None:
            Y = (C @ X)/c_scale
        else:
            Y = (C @ X)/c_scale + V
        Y_act = (C @ X)/c_scale

        # Store
        self.X.append(X)
        self.Y.append(Y)
        self.Y_act.append(Y_act)    # Actual water level.
        self.U.append(U)

        if Ain_dict is not None:
            for k, v in Ain_dict.items():
                if v is not None:
                    self.Sin_dict[k].append(v @ X)
        return Y

    @staticmethod
    def get_record(X):
        if isinstance(X, dict):
            return {k: np.array(np.hstack(v)) for k, v in X.items()}
        else:
            return np.array(np.hstack(X))

    def cal_Fh(self, ymax, tol=1e-5):
        Yact = SS.get_record(self.Y_act)[:,:-1]
        Fh = np.zeros(Yact.shape)
        fh = Yact - np.array(ymax).reshape(-1,1)
        Fh[np.where(fh>0+tol)] = 1
        self.Fh = Fh.T
        return Fh.T

    def get_ss_record(self):
        sim_y = SS.get_record(self.Y_act).T[:-1,:]
        sim_o = -SS.get_record(self.U).T[:-1,:]
        return sim_y, sim_o




class SWMM2SSModel(object):

    def __init__(self):
        pass

    def create_ss_model(self, storages, junctions, orifices, conduits,
                        subcatchs, delay_n, links, links_sb,
                        storage_areas, T=None, nodes=None, ratio=False):
        self.storages = storages
        self.junctions = junctions
        self.orifices = orifices
        self.conduits = conduits
        self.subcatchs = subcatchs
        self.delay_n = delay_n
        self.links = links
        self.links_sb = links_sb
        self.storage_areas = storage_areas
        self.T = T
        if nodes is None:
            self.nodes = storages + junctions
        else:
            self.nodes = nodes
        self.node_index = SWMM2SSModel.form_node_index(self.nodes, storages, junctions, links,
                                                       delay_n)

        # Create matrices in a Dictionary Of Keys format
        # (scipy.sparse.dok_matrix)
        node_index = self.node_index
        self.A = SWMM2SSModel.form_A(node_index, storages, junctions, delay_n, links, storage_areas,
                                     ratio)
        self.Ain_dict = SWMM2SSModel.form_Ain(node_index, storages, conduits, delay_n, links,
                                              storage_areas, ratio)
        if ratio:
            areas = None
        else:
            areas = storage_areas
        self.Bu = SWMM2SSModel.form_Bu(node_index, storages, junctions, delay_n, links, orifices,
                                       areas, T)
        self.Bw = SWMM2SSModel.form_Bw(node_index, storages, delay_n, links_sb, subcatchs, areas, T)
        self.C = SWMM2SSModel.form_C(node_index, storages, delay_n)

    @staticmethod
    def form_node_index(nodes, storages, junctions, links, delay_n):
        """
        Form node index dictionary for building A Bu Bw C matrices.

        Parameters
        ----------
        nodes : list
            Ordered node (storages and junctions) list.
        storages : list
        junctions : list
        delay_n : list
            No. delay segments for each junction.
        Returns
        -------
        node_index : dict

        """
        node_index = {}
        acc_i = 0
        for n in nodes:
            if n in storages:
                node_index[n] = (acc_i, acc_i)
                acc_i += 1
            else: # junctions
                dn = delay_n[junctions.index(n)]
                node_index[n] = (acc_i, acc_i+dn-1)
                acc_i += dn
        return node_index

    @staticmethod
    def form_A(node_index, storages, junctions, delay_n, links, storage_areas, ratio=False):
        """
        Form state matrix A.

        Parameters
        ----------
        node_index : dict
            Node index dictionary.
        storages : list
        junctions : list
        delay_n : list
        links : dict
            Dictionary of linkages among nodes.
        storage_areas : dict
            Dictionary of storage_areas given storage node.

        Returns
        -------
        A : scipy.sparse.dok_matrix
            State matrix.

        """
        n_nodes = len(storages) + sum(delay_n)
        size = (n_nodes, n_nodes)

        # Create dictionary-based sparse matrix.
        A = dok_matrix(size)

        # Add integrator for storages
        for s in storages:
            a, a = node_index[s]
            A[a, a] = 1
        # Add delay for junctions
        for j in junctions:
            a, b = node_index[j]
            for i in range(a, b):
                A[i+1,i] = 1
        # Add link (routing)
        for l in links:
            head, tail = links[l]
            if head in junctions and tail in storages:
                a_m, a_m = node_index[tail]
                a_n, b_n = node_index[head]
                dnn = b_n-a_n+1
                if dnn < 1:
                    print("Delay has to be at least 1.")
                for i in range(a_n, b_n+1):
                    if ratio:
                        # e.g., S2 -> J2 (head) -> S1 (tail)
                        print("Use {}'s area for {} inflowing to {}".format("S"+head[1:], head, tail))
                        A[a_m, i] = 1/(dnn)*storage_areas["S"+head[1:]]/storage_areas[tail]
                    else:
                        A[a_m, i] = 1/(dnn)/storage_areas[tail]

        return A

    @staticmethod
    def form_Ain(node_index, storages, conduits, delay_n, links, storage_areas, ratio=False):
        """
        Form state matrix Ain. Follow the conduits order.

        Parameters
        ----------
        node_index : dict
            Node index dictionary.
        storages : list
        conduits : list
        delay_n : list
        links : dict
            Dictionary of linkages among nodes.
        storage_areas : dict
            Dictionary of storage_areas given storage node.

        Returns
        -------
        Ain_dict : scipy.sparse.dok_matrix
            State matrix.

        """
        n_nodes = len(storages) + sum(delay_n)

        Ain_dict = {}
        for s in storages:
            C_list = [c for c in conduits if links[c][1] == s] # conduits flow to s
            if len(C_list) >= 1:
                Ain = dok_matrix((len(C_list), n_nodes))
            else:
                Ain = None
            for i, c in enumerate(C_list):
                head, tail = links[c]
                a_m, a_m = node_index[tail]
                a_n, b_n = node_index[head]
                dnn = b_n-a_n+1
                if dnn < 1:
                    print("Delay has to be at least 1.")
                for j in range(a_n, b_n+1):
                    if ratio:
                        # e.g., S2 -> J2 (head) -> S1 (tail)
                        print("Use {}'s area for {} inflowing to {}".format("S"+head[1:], head, tail))
                        Ain[i, j] = 1/(dnn)*storage_areas["S"+head[1:]]/storage_areas[tail]
                    else:
                        Ain[i, j] = 1/(dnn)/storage_areas[tail]
            Ain_dict[s] = Ain

        return Ain_dict

    @staticmethod
    def form_Bu(node_index, storages, junctions, delay_n, links, orifices,
                storage_areas=None, T=None):
        """
        Form control matrix Bu.

        Parameters
        ----------
        Bu : ndarray
            Empty array.
        node_index : dict
            Node index dictionary.
        storages : list
        junctions : list
        delay_n : list
        links : dict
            Dictionary of linkages among nodes.
        orifices : list
        storage_areas : dict
            Dictionary of storage_areas given storage node.
        T : str
            Time interval [sec]

        Returns
        -------
        Bu : scipy.sparse.dok_matrix
            Control matrix.

        """
        n_nodes = len(storages) + sum(delay_n)
        n_orifice = len(orifices)
        size = (n_nodes, n_orifice)

        if T is None:
            T = 1

        # Create dictionary-based sparse matrix.
        Bu = dok_matrix(size)

        for i, o in enumerate(orifices):
            head, tail = links[o]
            if head in storages:
                a_m, a_m = node_index[head]
                if storage_areas is None:
                    Bu[a_m, i] = T
                else:
                    Bu[a_m, i] = T/storage_areas[head]
            if tail in junctions:
                a_m, b_m = node_index[tail]
                Bu[a_m, i] = -T
        return Bu

    @staticmethod
    def form_Bw(node_index, storages, delay_n, links_sb, subcatchs,
                storage_areas=None, T=None):
        """
        Form disturbance matrix Bw.

        Parameters
        ----------
        Bw : ndarray
            Empty array.
        node_index : dict
            Node index dictionary.
        storages : list
        delay_n : list
        links_sb : dict
            Dictionary of linkages for subcatchment outlets and nodes.
        subcatchs : list
        storage_areas : dict
            Dictionary of storage_areas given storage node.
        T : str
            Time interval [sec]

        Returns
        -------
        Bw : ndarray
            Disturbance matrix.

        """

        n_nodes = len(storages) + sum(delay_n)
        n_subcatch = len(subcatchs)
        size = (n_nodes, n_subcatch)

        if T is None:
            T = 1

        # Create dictionary-based sparse matrix.
        Bw = dok_matrix(size)

        for node in links_sb:
            if node in storages:
                a_m, a_m= node_index[node]
                for sb in links_sb[node]:
                    if storage_areas is None:
                        Bw[a_m, subcatchs.index(sb)] = T
                    else:
                        Bw[a_m, subcatchs.index(sb)] = T/storage_areas[node]
            else: # junctions
                a_m, b_m= node_index[node]
                for sb in links_sb[node]:
                    # Assume the manhole can acommondate the runoff.
                    Bw[a_m, subcatchs.index(sb)] = 1
        return Bw

    @staticmethod
    def form_C(node_index, storages, delay_n):
        """
        Form output matrix C.

        Parameters
        ----------
        C : ndarray
            Empty array.
        node_index : dict
            Node index dictionary.
        storages : list
        delay_n : list

        Returns
        -------
        C : ndarray
            Output matrix.

        """
        n_nodes = len(storages) + sum(delay_n)
        n_storage = len(storages)
        size = (n_storage, n_nodes)

        # Create dictionary-based sparse matrix.
        C = dok_matrix(size)

        for i, s in enumerate(storages):
            a_m, a_m = node_index[s]
            C[i, a_m] = 1
        return C

    def convert_Y0toX0(self, Y0):
        """
        Convert initial measurement of Y0 into internal state vector X0.

        Automatically consider the matrix augmentation for integrator.

        Parameters
        ----------
        Y0 : 1darray
            Initial measurement vector.

        Returns
        -------
        X0 : 2darray

        """
        n_nodes = len(self.storages) + sum(self.delay_n)
        X0 = np.zeros((n_nodes, 1))
        node_index = self.node_index
        for i, s in enumerate(self.storages):
            X0[node_index[s][0], 0] = Y0[i]

        self.X0 = X0
        return X0

    @staticmethod
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


# class LQG(object):
#     def __init__(self, A, Bu, Bw, C, cov_V, cov_W, rho_q=1, rho_r=1, roun=None,
#                  Q=None, R=None):
#         """
#         Solving a steady state lqg controller for time-invariant state-space
#         model shown below.

#         X(k+1) = A * X(k) + Bu * U(k) + Bw * W(k)
#         Y(k+1) = C * X(k) + V(k)

#         User might apply Kalman decomposition to seperate controllable and
#         uncontrollable parts.

#         Parameters
#         ----------
#         A : 2darray
#             State matrix (observable & controllable).
#         Bu : 2darray
#             Control matrix (controllable).
#         Bw : 2darray
#             Disturbance matrix (controllable).
#         C : 2darray
#             Output matric (observable).
#         cov_V : 2darray
#             Covariance of sensor errors.
#         cov_W : 2darray
#             Covariance of disturbances (For solving Kalman gain, we use
#             cov = Bw cov_W Bw.T).
#         rho_q : float or 1darray
#             Weight of states cost.
#         rho_r : float or 1darray
#             Weight of control cost.
#         Q : 2darray
#             State error control weight matrix Ex: C.T Wx C.
#         R : 2darray
#             Control cost weight matrix Ex: Wu.
#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         """
#         def check_sparse2dense(matrix, roun):
#             if issparse(matrix):
#                 m = matrix.todense()
#                 return self.round_arr(m, roun)
#             else:
#                 return self.round_arr(matrix, roun)
#         # To ensure all matrices are dense for other calculation.
#         self.A = check_sparse2dense(A, roun)
#         self.Bu = check_sparse2dense(Bu, roun)
#         self.Bw = check_sparse2dense(Bw, roun)
#         self.C = check_sparse2dense(C, roun)
#         self.cov_V = check_sparse2dense(cov_V, roun)
#         self.cov_W = check_sparse2dense(cov_W, roun)
#         self.roun = roun
#         self.rho_q = rho_q
#         self.rho_r = rho_r

#         if Q is not None:
#             self.Q = check_sparse2dense(Q, roun)
#             print("rho_q is not applied.")
#         else:
#             if isinstance(rho_q, list) or isinstance(rho_q, np.ndarray):
#                 self.Q = check_sparse2dense((C.T @ np.diag(rho_q) @ C) , roun)
#             else:
#                 self.Q = check_sparse2dense((C.T @ C) * rho_q, roun)

#         if R is not None:
#             self.R = check_sparse2dense(R, roun)
#             print("rho_r is not applied.")
#         else:
#             if isinstance(rho_q, list) or isinstance(rho_q, np.ndarray):
#                 self.R = np.eye(self.Bu.shape[1]) @ np.diag(rho_r)
#             else:
#                 self.R = np.eye(self.Bu.shape[1]) * rho_r

#         try:
#             self.feedback_gain_dare()
#             print("Done Kk.")
#         except Exception as e:
#             print(e)
#             print("Kk is not done.")

#         try:
#             self.kalman_gain_dare()
#             print("Done L.")
#         except Exception as e:
#             print(e)
#             print("L is not done.")

#         try:
#             self.get_feedforward_dist_cancellation_Kw()
#             print("Done Kw.")
#         except Exception as e:
#             print(e)
#             print("Kw is not done.")

#         try:
#             self.get_feedforward_ref_tracking_Kr()
#             print("Done Kr.")
#         except Exception as e:
#             print(e)
#             print("Kr is not done.")


#     def round_arr(self, matrix, roun=None):
#         if roun is not None:
#             return matrix.round(roun)
#         else:
#             return matrix

#     def feedback_gain_dare(self):
#         """
#         Solve feedback gain (Kk) for steady state time invariant system.

#         Riccati difference equation (solve for steady state):
#             xi_P = A^T [xi_P - xi_P Bu (Bu^T xi_P Bu + R)^{-1} Bu^T xi_P] A + Q
#         Feedback gain:
#             Kk = (Bu^T xi_P Bu + R)^(-1) Bu^T xi_P A

#         Returns
#         -------
#         xi_P : 2darray
#             Covariance matrix.
#         Kk : 2darray
#             Feedback gain.

#         """
#         A = self.A
#         Bu = self.Bu
#         Q = self.Q
#         R = self.R
#         # Feedback gain for steady state time invariant system.
#         # Steady state covariance matrix solved by dare.
#         xi_P = solve_discrete_are(A, Bu, Q, R)
#         Kk = solve(Bu.T @ xi_P @ Bu + R, Bu.T @ xi_P @ A)
#         # check stability for discrete sys.
#         # eig = eigvals(np.eye(A.shape[0]) - A + Bu @ Kk)
#         self.Kk = self.round_arr(Kk, self.roun)
#         self.xi_P = xi_P
#         return xi_P, Kk

#     def kalman_gain_dare(self):
#         """
#         Solve Kalman gain (L) for steady state time invariant system.

#         Riccati difference equation (solve for steady state):
#             xi_X = A [xi_X - xi_X C^T (C xi_X X^T + V)^(-1) C xi_X] A^T
#                    + Bw cov_W Bw.T
#         Kalman gain:
#             L = xi_X C^T (C xi_X C^T + V)^(-1)

#         Note: For disturbance new cov = Bw cov_W Bw.T
#         Returns
#         -------
#         xi_X : 2darray
#             Covariance matrix.
#         L : 2darray
#             Kalman gain.

#         """
#         # A has to be observable and controllable.
#         A = self.A
#         C = self.C
#         Bw = self.Bw
#         cov_W = self.cov_W
#         cov_V = self.cov_V

#         # Steady state covariance matrix solved by dare.
#         xi_X = solve_discrete_are(A.T, C.T, Bw @ cov_W @ Bw.T, cov_V)

#         # Note that x' A = b' is equivalent to A' x = b
#         # Solve for L = xi_X C' (C xi_X C' + cov_V)^(-1)
#         # Kalman gain for steady state time invariant system.
#         L = solve( (C.dot(xi_X).dot(C.T) + cov_V).T, C.dot(xi_X.T) ).T
#         # eig =
#         self.L = self.round_arr(L, self.roun)
#         self.xi_X = xi_X
#         return xi_X, L#, eig

#     def get_feedforward_dist_cancellation_Kw(self, sys="discrete"):
#         """
#         Calculate feedforward disturbance cancellation matrix.

#         # Control to Y = 0.
#         For discrete system,
#         Kw = -[C(I-A+BuKk)^(-1)Bu]^(-1) C(I-A+BuKk)^(-1)Bw
#         Note: (I-A+BuKk) has to be square and invertable.

#         For continuous system,
#         Kw = -[C(A-BuKk)^(-1)Bu]^(-1) C(A-BuKk)^(-1)Bw
#         Note: (I-A+BuKk) has to be square and invertable.

#         Parameters
#         ----------
#         sys : str, optional
#             "discrete" or "continuous" system. The default is "discrete".

#         Returns
#         -------
#         Kw : 2darray
#         """
#         A = self.A
#         Bu = self.Bu
#         Bw = self.Bw
#         C = self.C
#         try:
#             Kk = self.Kk
#         except Exception as e:
#             print("Cannot find Kk. So we solve it using feedback_gain_dare")
#             self.feedback_gain_dare()
#             Kk = self.Kk

#         # Maybe use sparse matrix in the future.
#         if sys == "discrete":
#             F = inv(np.eye(A.shape[0]) - A + Bu @ Kk)
#         elif sys == "continuous":
#             F = inv(A - Bu @ Kk)

#         Kw = solve(-C @ F @ Bu, C @ F @ Bw)
#         self.Kw = self.round_arr(Kw, self.roun)
#         return Kw

#     def get_feedforward_ref_tracking_Kr(self, sys="discrete"):
#         """
#         Calculate feedforward reference tracking matrix (dc-gain).

#         # Control to Y = r.
#         For discrete system,
#         Kr = [C (I - A + Bu Kk)^(-1) Bu ]^(-1)
#         Note: Eqn above has to be square and invertable.

#         For continuous system,
#         Kr = -[ C (A - Bu Kk)^(-1) Bu]^(-1)
#         Note: Eqn above has to be square and invertable.

#         Parameters
#         ----------
#         sys : str, optional
#             "discrete" or "continuous" system. The default is "discrete".

#         Returns
#         -------
#         Kr : 2darray
#         """
#         A = self.A
#         Bu = self.Bu
#         C = self.C
#         try:
#             Kk = self.Kk
#         except Exception as e:
#             print("Cannot find Kk. So we solve it using feedback_gain_dare")
#             self.feedback_gain_dare()
#             Kk = self.Kk

#         if sys == "discrete":
#             F = inv(np.eye(A.shape[0]) - A + Bu @ Kk)
#             # This is wrong
#             #else:
#             #    F = inv(np.eye(A.shape[0]) - A + Bu @ Kk - Bu @ Kw)
#             Kr = inv(C @ F @ Bu)
#         elif sys == "continuous":
#             F = inv(A - Bu @ Kk)
#             Kr = -inv(C @ F @ Bu)
#         self.Kr = self.round_arr(Kr, self.roun)
#         return Kr

