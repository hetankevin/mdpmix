import confound_mdp
import numpy as np
from tqdm import tqdm

import scipy
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1, BFGS
from scipy.optimize import minimize

#------------------------------------------------------------------------------------
#   Importance sampling
#------------------------------------------------------------------------------------

def IS(dataset, gamma, horizon, pihat, pi_e):
    rets = confound_mdp.calc_returns(dataset, gamma, horizon)
    weighted_data = 0
    for traj,ret in zip(dataset,rets):
        rho = 1
        for x,a,u,xp,r in traj:
            rho *= (pi_e[int(x), int(a)] / pihat[int(x), int(a)])
        weighted_data += ret * rho
    return weighted_data / len(dataset)

def WIS(dataset, gamma, horizon, pihat, pi_e):
    rets = confound_mdp.calc_returns(dataset, gamma, horizon)
    weighted_data = 0
    is_sum = 0
    for traj,ret in zip(dataset,rets):
        rho = 1
        for x,a,u,xp,r in traj:
            rho *= (pi_e[int(x), int(a)] / pihat[int(x), int(a)])
        weighted_data += ret * rho
        is_sum += rho
    return weighted_data / is_sum

#------------------------------------------------------------------------------------
#   FQE and helper functions
#------------------------------------------------------------------------------------

def fitted_q_update(f, pi_e, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    regression_data = np.array([[x, a, r + mdp.gamma * (pi_e[int(xp)] @ f[int(xp),:])] for x,a,u,xp,r in data])
    Tf_hat = np.zeros((nStates, nActions))
    for x,a,y in regression_data:
        Tf_hat[int(x), int(a)] += y
    idx, count = np.unique(data[:,:2], axis=0, return_counts=True)
    for i,[x,a] in enumerate(idx):
        Tf_hat[int(x),int(a)] /= count[i]
    return Tf_hat

def fitted_q_evaluation(pi_e, dataset, horizon, mdp):
    Qhat = np.zeros((mdp.n_states, mdp.n_actions))
    for k in tqdm(range(horizon)):
        newQ = fitted_q_update(Qhat, pi_e, dataset, mdp)
        #trueNewQ = bellman_eval_update(Qhat, np.array([pi_e,pi_e]))
        #print("Squared error: " + str(((newQ - trueNewQ)**2).sum()))
        Qhat = newQ
    return Qhat


# compute empirical frequency of u given a state and an action 
def calc_u_prob(dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    u_hit = np.zeros((nStates, nActions))
    counts = np.zeros((nStates, nActions))

    for traj in dataset:
        for x,a,u,xp,r in traj:
            u_hit[int(x), int(a)] += u
            counts[int(x), int(a)] += 1

    u_prob = u_hit/counts
    #xa_prob = counts / (counts.sum())
    return u_prob

# true freq of u given s and a
def u_cond_x_a(pi, mdp):
    udist = mdp.u_dist
    nStates = mdp.n_states
    nActions = mdp.n_actions
    u_prob = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            pi_axu = pi[1, x, a]
            pi_ax = udist[0] * pi[0, x, a] + udist[1] * pi[1, x, a]
            u_prob[x,a] = (pi_axu * udist[1]) / pi_ax
    return u_prob

### this is the true expected value that FQE is targetting
def bellman_eval_update_biased(f, pi_b, pi, mdp):
    P = mdp.P
    R = mdp.R
    gamma = mdp.gamma
    nStates = mdp.n_states
    nActions = mdp.n_actions
    u_prob = u_cond_x_a(pi_b, mdp)

    Tf = np.zeros(f.shape)
    for s in range(nStates):
        for a in range(nActions):
            f_pi_0 = np.array([pi[0, xp] @ f[xp, :] for xp in range(nStates)])
            f_pi_1 = np.array([pi[1, xp] @ f[xp, :] for xp in range(nStates)])
            f_pi_avg = 0.5 * f_pi_0 + 0.5 * f_pi_1
            u0 = P[0, a, s] @ (R[a, s] + gamma * f_pi_avg)
            u1 = P[1, a, s] @ (R[a, s] + gamma * f_pi_avg)
            p_u = u_prob[s,a]
            Tf[s,a] = (1-p_u) * u0 + p_u * u1
    return Tf

def marginal_policy(pi_b, u_dist):
    return pi_b[0] * u_dist[0] + pi_b[1] * u_dist[1]

def reweighted_q_update(f, pi_b, pi_e, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions

    marg_pi = marginal_policy(pi_b, mdp.u_dist)
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    y_data = np.array([[x, a, r + mdp.gamma * (pi_e[int(xp)] @ f[int(xp),:])] for x,a,u,xp,r in data])
    
    # importance weight style TRUE reweighting term:
    adjustment_weights = np.array([[1, 1, marg_pi[int(x), int(a)] / pi_b[int(u), int(x), int(a)]] for x,a,u,xp,r in data])
    regression_data = adjustment_weights * y_data
    
    Tf_hat = np.zeros((nStates, nActions))
    for x,a,y in regression_data:
        Tf_hat[int(x), int(a)] += y
    idx, count = np.unique(data[:,:2], axis=0, return_counts=True)
    for i,[x,a] in enumerate(idx):
        Tf_hat[int(x),int(a)] /= count[i]
    return Tf_hat

# reweighting when you don't know u:
def bound_reweighted_update(f, pi_e, dataset, weight_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions

    pihat = estimate_pi(dataset, mdp)
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    
    def worst_weight(datapoint):
        x,a,_,xp,r = datapoint
        y = r + mdp.gamma * (pi_e[int(xp)] @ f[int(xp),:])
        if y >= 0:
            inv_weight = np.fmax( 1/weight_bound, pihat[int(x), int(a)])
            return y*inv_weight
        else:
            return y*weight_bound

    regression_data = np.array([[datapoint[0], datapoint[1], worst_weight(datapoint)] for datapoint in data])
    
    Tf_hat = np.zeros((nStates, nActions))
    for x,a,y in regression_data:
        Tf_hat[int(x), int(a)] += y
    idx, count = np.unique(data[:,:2], axis=0, return_counts=True)
    for i,[x,a] in enumerate(idx):
        Tf_hat[int(x),int(a)] /= count[i]
    return Tf_hat

def find_worst_weights(f, pi_e, xa_pihat, xa_data, weight_bound):
    n = len(xa_data)
    # number of variables = 4 + n
    
    lower = [0 for _ in range(4+n)]
    lower[2] = xa_pihat / weight_bound
    lower[3] = xa_pihat / weight_bound
    upper = [1 for _ in range(4+n)]
    upper[2] = np.fmin(xa_pihat * weight_bound,1)
    upper[3] = np.fmin(xa_pihat * weight_bound,1)
    bounds = Bounds(lower, upper)
    lincon = [0 for _ in range(4+n)]
    lincon[0] = 1
    lincon[1] = 1
    linear_constraint1 = LinearConstraint([lincon], [1], [1])
    
    def picons_f(x):
        return x[0]*x[2] + x[1]*x[3] - xa_pihat
    def picons_J(x):
        der = np.zeros(n+4)
        der[0] = x[2]
        der[1] = x[3]
        der[2] = x[0]
        der[3] = x[1]
        return der
    from scipy.sparse import csc_matrix
    def picons_H_sparse(x, v):
        row = np.array([0, 1, 2, 3])
        col = np.array([2, 3, 0, 1])
        data = np.array([1, 1, 1, 1])
        return csc_matrix((data, (row, col)), shape=(n+4, n+4))
    
    nonlinear_constraint = NonlinearConstraint(picons_f, 0.0, 0.0, jac=picons_J, hess=picons_H_sparse)
    
    def freq_f(x):
        return x[4:].sum()/n - x[1] * x[3] / xa_pihat
    def freq_J(x):
        der = np.zeros(n+4)
        der[1] = - x[3] / xa_pihat
        der[3] = - x[1] / xa_pihat
        der[4:] = 1/n
        return der
    def freq_H_sparse(x, v):
        row = np.array([1, 3])
        col = np.array([3, 1])
        data = np.array([-1/xa_pihat,-1/xa_pihat])
        return csc_matrix((data, (row, col)), shape=(n+4, n+4))
    
    nonlinear_constraint2 = NonlinearConstraint(freq_f, 0.0, 0.0, jac=freq_J, hess=freq_H_sparse)
    
    def value(x):
        pi_u0 = x[2]
        pi_u1 = x[3]
        uval = np.array(x[4:])
        weight0 = (xa_pihat / pi_u0)
        weight1 = (xa_pihat / pi_u1)
        weights = (1 - uval) * weight0 + uval * weight1
        return (weights @ xa_data)/n
    
    def value_jac(x):
        dat = np.array(xa_data)
        der = np.zeros(n+4)
        uval = x[4:]
        w2 = (1-uval) * xa_pihat / (x[2]*x[2])
        der[2] -= w2 @ dat
        w3 = uval * xa_pihat / (x[3]*x[3])
        der[3] -= w3 @ dat
        der[4:] = (xa_pihat / x[3] - xa_pihat / x[2]) * dat
        return der
    
    def value_hessp(x, p):
        dat = np.array(xa_data)
        hp = np.zeros(n+4)
        # each entry is dot product of p with row of hessian (second derivatives of single first deriv)
        uval = x[4:]
        hp2w = (2 * (1-uval) * xa_pihat) / (x[2]*x[2]*x[2]) 
        hp[2] += p[2] * (hp2w @ dat)
        hp[2] += p[4:] @ ((dat * xa_pihat) / (x[2]*x[2]))
        
        hp2w = (2 * uval * xa_pihat) / (x[3]*x[3]*x[3]) 
        hp[3] += p[3] * (hp2w @ dat)
        hp[3] -= p[4:] @ ((dat * xa_pihat) / (x[3]*x[3]))
        
        hp[4:] = p[3] * (-xa_pihat*dat)  / (x[3]*x[3]) + p[2] * (xa_pihat*dat)  / (x[2]*x[2])
        return hp
    
    x0 = np.zeros(n+4)
    x0[0] = 0.4
    x0[1] = 0.6
    x0[3] = 0.5
    x0[2] = (xa_pihat - x0[1]*x0[3]) / x0[0]
    x0[4:] = x0[1] * x0[3] / xa_pihat
    res = minimize(value, x0, method='trust-constr', jac=value_jac, hessp=value_hessp,
               constraints=[linear_constraint1, nonlinear_constraint, nonlinear_constraint2],
               options={'verbose': 0}, bounds=bounds)

    return res

def sorted_reweight_update(f, pi_e, dataset, weight_bound, mdp, verbose=False):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    pihat = estimate_pi(dataset, mdp)
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    regression_data = np.array([[x, a, r + mdp.gamma * (pi_e[int(xp)] @ f[int(xp),:])] for x,a,u,xp,r in data])
    
    datalists = {}
    for x,a,y in regression_data:
        if not (int(x),int(a)) in datalists:
            datalists[(int(x),int(a))] = []
        datalists[(int(x),int(a))].append(y)
        
    Tf_hat = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            if verbose:
                print((x,a))
            # find worst case weights
            res = find_worst_weights(f, pi_e, pihat[a,x], datalists[(x,a)], weight_bound)
            Tf_hat[x,a] = res.fun
            #print(res.fun)
            #print(res.x)
    return Tf_hat

#------------------------------------------------------------------------------------
#   Model-based and helper functions
#------------------------------------------------------------------------------------

def estimate_P(dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    
    Phat = np.zeros((nActions, nStates, nStates))
    counts = np.zeros((nActions, nStates))
    for x,a,u,xp,r in data:
        Phat[int(a), int(x), int(xp)] += 1
        counts[int(a), int(x)] += 1
    for a in range(nActions):
        for x in range(nStates):
            if counts[a,x] > 0:
                Phat[a,x] /= counts[a,x]
    return Phat

def estimate_R(dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    
    Rhat = np.zeros((nActions, nStates, nStates))
    counts = np.zeros((nActions, nStates, nStates))
    for x,a,u,xp,r in data:
        Rhat[int(a), int(x), int(xp)] += r
        counts[int(a), int(x), int(xp)] += 1
    for a in range(nActions):
        for x in range(nStates):
            for xp in range(nStates):
                if counts[a,x,xp] > 0:
                    Rhat[a,x, xp] /= counts[a,x,xp]
    return Rhat

def estimate_pi(dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions   
    data = dataset.reshape((dataset.shape[0]*dataset.shape[1],5))
    
    pihat = np.zeros((nActions, nStates))
    counts = np.zeros(nStates)
    for x,a,u,xp,r in data:
        pihat[int(a), int(x)] += 1
        counts[int(x)] += 1
    for x in range(nStates):
        if counts[x] > 0:
            pihat[:,x] /= counts[x]
    return pihat.T


def worst_case_ax_norm(action, state, dataset, pi_e, f, x0, P_bound, cond_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions  
    Phat = estimate_P(dataset, mdp)
    Rhat = estimate_R(dataset, mdp)
    eps = 1/np.sqrt(dataset.shape[0])
    
    bounds = Bounds([0, 0, 0, 0, 0, 0, cond_bound, cond_bound], [1, 1, 1, 1, 1, 1, 1-cond_bound, 1-cond_bound])

    linear_constraint1 = LinearConstraint([[1, 1, 1, 0, 0, 0, 0, 0]], [1], [1])
    linear_constraint2 = LinearConstraint([[0, 0, 0, 1, 1, 1, 0, 0]], [1], [1])
    linear_constraint3 = LinearConstraint([[0, 0, 0, 0, 0, 0, 1, 1]], [1], [1])
    
    def cons_f(x):
        return (x[0]*x[6] + x[3]*x[7] - Phat[action, state, 0])**2 + (x[1]*x[6] + x[4]*x[7] - Phat[action, state, 1])**2 + (x[2]*x[6] + x[5]*x[7] - Phat[action, state, 2])**2
    def cons_J(x):
        f1 = x[0]*x[6] + x[3]*x[7] - Phat[action, state, 0]
        f2 = x[1]*x[6] + x[4]*x[7] - Phat[action, state, 1]
        f3 = x[2]*x[6] + x[5]*x[7] - Phat[action, state, 2]
        der = np.array([
            2*x[6]*f1,
            2*x[6]*f2,
            2*x[6]*f3,
            2*x[7]*f1,
            2*x[7]*f2,
            2*x[7]*f3,
            2*x[0]*f1 + 2*x[1]*f2 + 2*x[2]*f3,
            2*x[3]*f1 + 2*x[4]*f2 + 2*x[5]*f3
        ])
        return der
    nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, eps**2, jac=cons_J, hess=SR1())
    
    def confound_f(x):
        return (x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2
    def confound_J(x):
        der = np.array([
            2*(x[0]-x[3]),
            2*(x[1]-x[4]),
            2*(x[2]-x[5]),
            -2*(x[0]-x[3]),
            -2*(x[1]-x[4]),
            -2*(x[2]-x[5]),
            0.0,
            0.0
        ])
        return der

    nonlinear_constraint2 = NonlinearConstraint(confound_f, 0.0, P_bound**2, jac=confound_J, hess=SR1())
    
    def update_cost(x):
        P_ax0 = x[0:nStates]
        P_ax1 = x[nStates:nStates+nStates]

        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        return mdp.u_dist[0] * (y @ P_ax0) + mdp.u_dist[1] * (y @ P_ax1)
    
    def update_jac(x):
        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        der = np.zeros(8)
        der[:nStates] = mdp.u_dist[0] * y
        der[nStates:nStates+nStates] = mdp.u_dist[1] * y
        return der
    
    res = minimize(update_cost, x0, method='trust-constr', jac=update_jac, hess=SR1(),
               constraints=[linear_constraint1, linear_constraint2, linear_constraint3, nonlinear_constraint, nonlinear_constraint2],
               options={'verbose': 0}, bounds=bounds)
    
    return res

def worst_case_update(f, pi_e, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions  
    P_bound = 0.848528137423857
    cond_bound = 0.25
    Qworst = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            worst = np.inf
            for _ in range(3):
                p = np.random.uniform(size=3)
                x0 = np.array([p[0], 1-p[0], 0, p[1], 1-p[1], 0, p[2], 1-p[2]])
                fun = worst_case_ax_norm(a, x, dataset, pi_e, f, P_bound, cond_bound, x0, mdp).fun
                if fun < worst:
                    worst = fun
            Qworst[x, a] = worst
    return Qworst

def worst_case_ax_policy(action, state, dataset, pi_e, f, P_bound, pi_bound, x0, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions  
    Phat = estimate_P(dataset, mdp)
    Rhat = estimate_R(dataset, mdp)
    pihat = estimate_pi(dataset, mdp)
    eps = 1/np.sqrt(dataset.shape[0])
    #eps = 0.0

    # variable: 10-dimensional
    #    x[0]: Phat[0|x, a, u=0] 
    #    x[1]: Phat[1|x, a, u=0]
    #    x[2]: Phat[2|x, a, u=0]
    #    x[3]: Phat[0|x, a, u=1] 
    #    x[4]: Phat[1|x, a, u=1]
    #    x[5]: Phat[2|x, a, u=1]
    #    x[6]: pi[a | x, u=0]
    #    x[7]: pi[a | x, u=1]
    #    x[8]: p[u=0]
    #    x[9]: p[u=1]
    
    #, Phat[:|x, a, 1] , pi(a | x, :) , p(:)
    
    bounds = Bounds([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # note we can require with a constraint that p(u|x,a) is in (0.25, 0.75) as before 

    linear_constraint1 = LinearConstraint([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], [1], [1])
    linear_constraint2 = LinearConstraint([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]], [1], [1])
    #linear_constraint3 = LinearConstraint([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]], [1], [1])
    linear_constraint4 = LinearConstraint([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], [1], [1])
    
    def cons_f(x):
        y0 = x[8] * x[6] / pihat[action, state]
        y1 = x[9] * x[7] / pihat[action, state]
        return (x[0]*y0 + x[3]*y1 - Phat[action, state, 0])**2 + (x[1]*y0 + x[4]*y1 - Phat[action, state, 1])**2 + (x[2]*y0 + x[5]*y1 - Phat[action, state, 2])**2
    def cons_J(x):
        pih = pihat[action,state]
        y0 = x[8] * x[6] / pih
        y1 = x[9] * x[7] / pih
        f1 = x[0]*y0 + x[3]*y1 - Phat[action, state, 0]
        f2 = x[1]*y0 + x[4]*y1 - Phat[action, state, 1]
        f3 = x[2]*y0 + x[5]*y1 - Phat[action, state, 2]
        der = np.array([
            2*y0*f1,
            2*y0*f2,
            2*y0*f3,
            2*y1*f1,
            2*y1*f2,
            2*y1*f3,
            2*f1*(x[0]*x[8]/pih) + 2*f2*(x[1]*x[8]/pih) + 2*f3*(x[2]*x[8]/pih),
            2*f1*(x[3]*x[9]/pih) + 2*f2*(x[4]*x[9]/pih) + 2*f3*(x[5]*x[9]/pih),
            2*f1*(x[0]*x[6]/pih) + 2*f2*(x[1]*x[6]/pih) + 2*f3*(x[2]*x[6]/pih),
            2*f1*(x[3]*x[7]/pih) + 2*f2*(x[4]*x[7]/pih) + 2*f3*(x[5]*x[7]/pih)
        ])
        return der
    nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, eps**2, jac=cons_J, hess=SR1())
    
    def picons_f(x):
        return x[6]*x[8] + x[7]*x[9] - pihat[action, state]
    def picons_J(x):
        der = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            x[8],
            x[9],
            x[6],
            x[7]
        ])
        return der
    nonlinear_constraint3 = NonlinearConstraint(picons_f, 0.0, 0.0, jac=picons_J, hess=SR1())
    
    def confound_f(x):
        return (x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2
    def confound_J(x):
        der = np.array([
            2*(x[0]-x[3]),
            2*(x[1]-x[4]),
            2*(x[2]-x[5]),
            -2*(x[0]-x[3]),
            -2*(x[1]-x[4]),
            -2*(x[2]-x[5]),
            0.0,
            0.0,
            0.0,
            0.0
        ])
        return der

    nonlinear_constraint2 = NonlinearConstraint(confound_f, 0.0, P_bound**2, jac=confound_J, hess=SR1())
    
    def pi_confound_f(x):
        return x[6]/x[7]
    def pi_confound_J(x):
        der = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1/x[7],
            -x[6]/(x[7]*x[7]),
            0.0,
            0.0
        ])
        return der
    nonlinear_constraint4 = NonlinearConstraint(pi_confound_f, 1/pi_bound, pi_bound, jac=pi_confound_J, hess=SR1())
    
    def update_cost(x):
        P_ax0 = x[0:nStates]
        P_ax1 = x[nStates:nStates+nStates]

        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        return x[8] * (y @ P_ax0) + x[9] * (y @ P_ax1)
    
    def update_jac(x):
        P_ax0 = x[0:nStates]
        P_ax1 = x[nStates:nStates+nStates]
        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        der = np.zeros(10)
        der[:nStates] = x[8] * y
        der[nStates:nStates+nStates] = x[9] * y
        der[8] = y @ P_ax0
        der[9] = y @ P_ax1
        return der
    
    res = minimize(update_cost, x0, method='trust-constr', jac=update_jac, hess=SR1(),
               constraints=[linear_constraint1, linear_constraint2, linear_constraint4, 
                            nonlinear_constraint, nonlinear_constraint2, nonlinear_constraint3, nonlinear_constraint4],
               options={'verbose': 0}, bounds=bounds)
    
    return res

def worst_case_update_policy(f, pi_e, P_bound, pi_bound, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions  
    #P_bound = 0.848528137423857
    #pi_bound = 0.25
    Qworst = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            worst = np.inf
            for _ in range(3):
                #p = np.random.uniform(size=4)
                #x0 = np.array([p[0], 1-p[0], 0, p[1], 1-p[1], 0, p[2], 1-p[2], p[3], 1-p[3]])
                x0 = np.random.uniform(size=10)
                fun = worst_case_ax_policy(a, x, dataset, pi_e, f, P_bound, pi_bound, x0, mdp).fun
                if fun < worst:
                    worst = fun
            Qworst[x, a] = worst
    return Qworst


def fixed_u_dist_sa(action, state, dataset, pi_e, f, pihat, Phat, P_bound, pi_bound, u_dist, x0, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions  
    Rhat = mdp.R
    eps = 1/np.sqrt(dataset.shape[0])

    pih = pihat[state, action]
  
    # variable dimensions: 2*nStates + 2
    #    x[:nStates]: Phat[:|x, a, u=0] 
    #    x[nStates:nStates*2]: Phat[:|x, a, u=1] 
    #    x[-2]: pi[a | x, u=0]
    #    x[-1]: pi[a | x, u=1]
    nDim = nStates*2 + 2
    
    lowerbounds = [0 for _ in range(nDim)]
    upperbounds = [1 for _ in range(nDim)]
    
    for i in range(nStates):     
        prePupper = P_bound/Phat[action, state, i] + (1-P_bound)
        prePlower = 1/(P_bound*Phat[action, state, i]) + (1 - 1/P_bound)
        lowerbounds[i] = 1/prePupper
        lowerbounds[i+nStates] = 1/prePupper
        upperbounds[i] = min(1/prePlower, 1)
        upperbounds[i+nStates] = min(1/prePlower, 1)

    prepiupper = pi_bound/pih + (1-pi_bound)
    prepilower = 1/(pi_bound*pih) + (1 - 1/pi_bound)
    lowerbounds[-1] = 1/prepiupper
    lowerbounds[-2] = 1/prepiupper
    upperbounds[-1] = min(1/prepilower, 1)
    upperbounds[-2] = min(1/prepilower, 1)
    
    bounds = Bounds(lowerbounds, upperbounds)

    cons1 = np.array([0 for _ in range(nDim)])
    cons1[:nStates] = 1
    linear_constraint1 = LinearConstraint([cons1], [1], [1])
    
    cons2 = np.array([0 for _ in range(nDim)])
    cons2[nStates:2*nStates] = 1
    linear_constraint2 = LinearConstraint([cons2], [1], [1])
    
    # cons3 = np.array([0 for _ in range(nDim)])
    # cons3[-2:] = 1
    # linear_constraint3 = LinearConstraint([cons3], [1], [1])
    
    cons4 = [0 for _ in range(nDim)]
    cons4[-2] = u_dist[0] 
    cons4[-1] = u_dist[1] 
    linear_constraint4 = LinearConstraint([cons4], [pih], [pih])
        
    def cons_f(x):
        y0 = u_dist[0] * x[-2] / pih
        y1 = u_dist[1] * x[-1] / pih
        res = 0
        for i in range(nStates):
            res += (x[i]*y0 + x[i+nStates]*y1 - Phat[action, state, i])**2
        return res
    def cons_J(x):
        
        y0 = u_dist[0] * x[-2] / pih
        y1 = u_dist[1] * x[-1] / pih
        der = np.zeros((nDim))
        
        for i in range(nStates):
            fi = x[i]*y0 + x[i+nStates]*y1 - Phat[action, state, i]
            der[i] = 2*y0*fi
            der[i+nStates] = 2*y1*fi
            der[-2] += 2*fi*(x[i]*u_dist[0]/pih)
            der[-1] += 2*fi*(x[i+nStates]*u_dist[1]/pih)
        
        return der
    nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, eps**2, jac=cons_J, hess=SR1())
    
    def update_cost(x):
        P_ax0 = x[0:nStates]
        P_ax1 = x[nStates:nStates+nStates]
        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        return u_dist[0] * (y @ P_ax0) + u_dist[1] * (y @ P_ax1)
    
    def update_jac(x):
        P_ax0 = x[0:nStates]
        P_ax1 = x[nStates:nStates+nStates]
        y = np.array([Rhat[action,state,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
        der = np.zeros(nDim)
        der[:nStates] = u_dist[0] * y
        der[nStates:nStates+nStates] = u_dist[1] * y
        return der
    
    def update_hess(x):
        return np.zeros((nDim,nDim))
    
    res = minimize(update_cost, x0, method='trust-constr', jac=update_jac, hess=update_hess,
               constraints=[linear_constraint1, linear_constraint2, linear_constraint3, linear_constraint4,
                            nonlinear_constraint], 
               options={'verbose': 0}, bounds=bounds)
    
    return res

def fixed_u_dist(f, pi_e, pihat, Phat, P_bound, pi_bound, u_dist, dataset, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    nDim = nStates*2 + 2
    Qworst = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            worst = np.inf
            for _ in range(3):
                x0 = np.random.uniform(size=nDim)
                res = fixed_u_dist_sa(a, x, dataset, pi_e, f, pihat, Phat, P_bound, pi_bound, u_dist, x0, mdp)
                if res.fun < worst and res.constr_violation < 0.1:
                    worst = res.fun
            Qworst[x, a] = worst
    return Qworst

#account for sampling variance:
def fitted_q_update_reparam_sampling(f, pi_e, pihat, Phat, Gamma, data, mdp):
    nStates = Phat.shape[1]
    nActions = Phat.shape[0]
    gamma = mdp.gamma

    eps = 1 / np.sqrt(len(data))
    regression_data = np.array([[x, a, xp, r + gamma * (pi_e[int(xp)] @ f[int(xp),:])] for x,a,u,xp,r in data])

    next_state_sums = np.zeros((nStates, nActions, nStates))
    sa_count = np.zeros((nStates, nActions))
    for x,a,xp,y in regression_data:
        next_state_sums[(int(x),int(a),int(xp))] += y
        sa_count[int(x), int(a)] += 1
        
    Tf_hat = np.zeros((nStates, nActions))
    for s in range(nStates):
        for a in range(nActions):
            #print((s,a))
            #print(next_state_sums[s,a,:]/sa_count[s,a])
            sub_data = next_state_sums[s,a,:]
            if sa_count[s,a] > 0:
                sub_data /= sa_count[s,a]
                Tf_hat[s,a] = fitted_q_update_reparam_sampling_sa(pihat[s,a], Phat[a, s, :], Gamma, eps, sub_data)
            else:
                Tf_hat[s,a] = 0
    return Tf_hat

def fitted_q_update_reparam_sampling_sa(pi_sa, Phat_sa, Gamma, eps, data):
    
    # optimization algorithm:
    #   minimize sum of data points combined with weights
    #      sum over i : y[i] * g[sp[i] , a, s]
    #      equivalently:  (sum over i such that sp of y) * g[sp, a, s] * pi[a|s]
    
    # opt variable = w[nStates] = g(sp,a,s)*pi(a|s)
    
    upper = Gamma + (1-Gamma)*pi_sa
    lower = 1/Gamma + (1 - 1/Gamma)*pi_sa
    bounds = [(lower, upper)]
    
    A_ub = [Phat_sa, -Phat_sa]
    b_ub = [1 + eps, - 1 + eps]
    
    res = scipy.optimize.linprog(data, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    if not res.success:
        print("LP FAILED!")
        print(data)    
        print(res.x)
        print(res.fun)
        print("")

    return res.fun

import gurobipy as gp
from gurobipy import GRB

def fixed_u_gp_sa(s, a, y, u_param, Phat, pihat, P_bound, pi_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    nU = mdp.n_confound

    m = gp.Model("bilinear")
    m.setParam('OutputFlag', 0) 
    
    u_dist = np.array([1-u_param, u_param])
    
    prepiupper = pi_bound/pihat[s,a] + (1-pi_bound)
    prepilower = 1/(pi_bound*pihat[s,a]) + (1 - 1/pi_bound)
    
    P = m.addMVar((nU,nStates), lb=0.0, ub=1.0) 
    pi = m.addMVar(nU, lb=1/prepiupper, ub=min(1/prepilower, 1))
    
    for i in range(nStates):
        if Phat[a,s,i] == 0:
            m.addConstr(P[:,i] >= 0)
            m.addConstr(P[:,i] <= 0)
        else:
            prePupper = P_bound/Phat[a, s, i] + (1-P_bound)
            prePlower = 1/(P_bound*Phat[a, s, i]) + (1 - 1/P_bound)
            m.addConstr(P[:,i] >= 1/prePupper)
            m.addConstr(P[:,i] <= min(1/prePlower, 1))

    for u in range(nU):
        m.addConstr(P[u, :] @ np.ones(nStates) == 1)

    for i in range(nStates):
        m.addConstr((pi @ np.diag(u_dist)) @ P[:,i] == pihat[s,a] * Phat[a,s,i])

    m.addConstr(pi @ u_dist == pihat[s,a])

    m.setObjective(sum(P[u,:] @ y * u_dist[u] for u in range(nU)), GRB.MINIMIZE)

    m.params.NonConvex = 2

    m.update()
    m.optimize()
    res_x = np.array([m.getVars()[i].x for i in range(nStates*nU+nU)])
    return m.objVal, res_x

def fixed_u_gp(f, pi_e, u_param, Phat, pihat, P_bound, pi_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    Qworst = np.zeros((nStates, nActions))
    for x in range(nStates):
        for a in range(nActions):
            y = np.array([mdp.R[a,x,xp] + mdp.gamma * (pi_e[xp] @ f[xp, :]) for xp in range(nStates)])
            worst, vec = fixed_u_gp_sa(x, a, y, u_param, Phat, pihat, P_bound, pi_bound, mdp)
            Qworst[x, a] = worst
    return Qworst

def fixed_u_gp_s_rect_s(s, y, pi_e, u_param, Phat, pihat, P_bound, pi_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    nU = mdp.n_confound

    m = gp.Model("bilinear")
    m.setParam('OutputFlag', 0) 
    
    
    u_dist = np.array([1-u_param, u_param])
    
    P = m.addMVar((nU,nStates, nActions), lb=0.0, ub=1.0) 
    pi = m.addMVar((nU,nActions), lb=0.0, ub=1.0)
    
    for a in range(nActions):
        prepiupper = pi_bound/pihat[s,a] + (1-pi_bound)
        prepilower = 1/(pi_bound*pihat[s,a]) + (1 - 1/pi_bound)
        m.addConstr(pi[:,a] >= 1/prepiupper)
        m.addConstr(pi[:,a] <= 1/prepilower)
    
    for a in range(nActions):
        for i in range(nStates):
            if Phat[a,s,i] == 0:
                m.addConstr(P[:,i,a] >= 0)
                m.addConstr(P[:,i,a] <= 0)
            else:
                prePupper = P_bound/Phat[a, s, i] + (1-P_bound)
                prePlower = 1/(P_bound*Phat[a, s, i]) + (1 - 1/P_bound)
                m.addConstr(P[:,i,a] >= 1/prePupper)
                m.addConstr(P[:,i,a] <= min(1/prePlower, 1))

    for a in range(nActions):
        for u in range(nU):
            m.addConstr(P[u, :, a] @ np.ones(nStates) == 1)

    for a in range(nActions):
        for i in range(nStates):
            m.addConstr((pi[:,a] @ np.diag(u_dist)) @ P[:,i, a] == pihat[s,a] * Phat[a,s,i])

    for a in range(nActions):
        m.addConstr(pi[:,a] @ u_dist == pihat[s,a])
        
    for u in range(nU):
        m.addConstr(pi[u,:] @ np.ones(nActions) == 1)

    m.setObjective(sum(sum(P[u,:, a] @ y * u_dist[u] for u in range(nU))*pi_e[s,a] for a in range(nActions)), GRB.MINIMIZE)

    #m.params.OptimalityTol = 1e-9 # for testing numerical stability
    m.params.NonConvex = 2
    m.params.PoolSearchMode = 1

    m.update()
    m.optimize()
    res_x = np.array([m.getVars()[i].x for i in range((nStates*nU+nU)*nActions)])
    return m.objVal, res_x

def fixed_u_gp_s_rect(f, pi_e, u_param, Phat, pihat, P_bound, pi_bound, mdp):
    nStates = mdp.n_states
    nActions = mdp.n_actions
    
    R_pi = np.zeros((nStates, nStates))
    for s in range(nStates):
        for sp in range(nStates):
            R_pi[s,sp] = pi_e[s] @ mdp.R[:,s,sp] 
            
    Vworst = np.zeros(nStates)
    for x in range(nStates):
        y = np.array([R_pi[x,xp] + mdp.gamma * f[xp] for xp in range(nStates)])
        worst, vec = fixed_u_gp_s_rect_s(x, y, pi_e, u_param, Phat, pihat, P_bound, pi_bound, mdp)
        Vworst[x] = worst
    return Vworst