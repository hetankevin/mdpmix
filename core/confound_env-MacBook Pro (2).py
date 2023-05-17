import numpy as np

#------------------------------------------------------------------------
# Helper functions to make confounded versions
#------------------------------------------------------------------------

def R_confound(tx, R, confound_weight=0.1):
    nActions = tx.shape[0]
    nStates = tx.shape[1]

    # e.g. take all transitions with R > 0 and upweight, take all transitions with R < 0 and downweight
    P = np.zeros((2, nActions, nStates, nStates))

    pos_r = R > 0
    neg_r = R <= 0

    tx_copy0 = tx.copy()
    valid = (tx_copy0 < (1-confound_weight)) & (tx_copy0 > confound_weight)

    tx_copy0[pos_r & valid] += confound_weight
    tx_copy0[neg_r & valid] -= confound_weight

    tx_copy1 = tx.copy()
    valid = (tx_copy1 < (1-confound_weight)) & (tx_copy1 > confound_weight)

    tx_copy1[pos_r & valid] -= confound_weight
    tx_copy1[neg_r & valid] += confound_weight

    for x in range(nStates):
        for a in range(nActions):
            tx_copy0[a,x] /= tx_copy0[a,x].sum()
            tx_copy1[a,x] /= tx_copy1[a,x].sum()

    P[0] = tx_copy0
    P[1] = tx_copy1
    
    return P

def rand_pi_val(tx, R, x_dist, horizon):
    nActions = tx.shape[0]
    nStates = tx.shape[1]
    rand_pi = np.zeros((nStates, nActions))
    for i in range(nStates):
        rand_pi[i,:] = [1/nActions for a in range(nActions)]
    gamma = 0.98
    
    def bellman_eval_update(f, pi):
        Tf = np.zeros(f.shape)
        for s in range(nStates):
            for a in range(nActions):
                f_pi = np.array([pi[xp] @ f[xp, :] for xp in range(nStates)])
                Tf[s,a] = tx[a, s] @ (R[a, s] + gamma * f_pi) 
        return Tf
    
    def bellman_eval(pi, horizon):
        Q = np.zeros((nStates, nActions))
        for k in range(horizon):
            Q = bellman_eval_update(Q, pi)
        return Q
    
    def get_value(Q, pi):
        V = np.array([Q[x] @ pi[x] for x in range(nStates)])
        avgV = V @ x_dist
        return V, avgV
    
    Q = bellman_eval(rand_pi, horizon)
    V, _ = get_value(Q, rand_pi)
    return V

def confound_V(tx, x_dist, V, confound_weight=0.1):
    nActions = tx.shape[0]
    nStates = tx.shape[1]
    
    P = np.zeros((2, nActions, nStates, nStates))
    for i in range(nStates):
        # which V is higher
        va = np.array([tx[a, i] @ V for a in range(nActions)])

        minaction = va.argmin()
        maxaction = va.argmax()

        # find the highest value action transition
        # in u = 0 shift all actions towards the highest action probs
        # find the lowest value action transition
        # in u = 1 shift all actions towards the lowest value
        for a in range(nActions):
            P[0, a, i, :] = (1-confound_weight)*tx[a, i] + confound_weight*tx[maxaction, i]
            P[1, a, i, :] = (1-confound_weight)*tx[a, i] + confound_weight*tx[minaction, i]
    return P

def confound_pi_R(pi, tx, R, confound_weight=0.1):
    nStates = tx.shape[1]
    nActions = tx.shape[0]
    pi_u = np.zeros((2, nStates, nActions))
         
    # take all transitions with R > 0 and upweight, take all transitions with R < 0 and downweight
    for s in range(nStates):
        for a in range(nActions):
            avgR = R[a, s, :] @ tx[a, s, :]
            if avgR >= 0:
                pi_u[0, s, a] = min(pi[s,a] + confound_weight, 1.0)
                pi_u[1, s, a] = max(pi[s,a] - confound_weight, 0.0)
            else:
                pi_u[0, s, a] = max(pi[s,a] - confound_weight, 0.0)
                pi_u[1, s, a] = min(pi[s,a] + confound_weight, 1.0)
        pi_u[0, s, :] += 0.05
        pi_u[1, s, :] += 0.05
        pi_u[0, s, :] /= pi_u[0, s, :].sum()
        pi_u[1, s, :] /= pi_u[1, s, :].sum()

    return pi_u

def confound_pi_V(pi, tx, V, confound_weight):
    nStates = tx.shape[1]
    nActions = tx.shape[0]
    pi_u = np.zeros((2, nStates, nActions))
    for i in range(nStates):
        va = np.array([tx[a, i] @ V for a in range(nActions)])
        maxaction = va.argmax() 
    
        for a in range(nActions):
            if a == maxaction:
                pi_u[0, i, a] = pi[i, a] + confound_weight
                pi_u[1, i, a] = pi[i, a] - confound_weight
            else:
                pi_u[0, i, a] = pi[i, a] - confound_weight
                pi_u[1, i, a] = pi[i ,a] + confound_weight
        
            if pi_u[0, i, a] < 0:
                pi_u[0, i, a] = 0
            if pi_u[1, i, a] < 0:
                pi_u[1, i, a] = 0
        
        # need to guarantee overlap:
        pi_u[0,i,:] += 0.05
        pi_u[1,i,:] += 0.05
        pi_u[0, i, :] /= pi_u[0, i, :].sum()
        pi_u[1, i, :] /= pi_u[1, i, :].sum()
    return pi_u

#------------------------------------------------------------------------
# Experimental Environment from Yangyi
#------------------------------------------------------------------------

def toyH2(pi_param, P_param, r_param):
    #R a, s, s'
    R = np.zeros((2,2,2))
    R[0] = np.array([[r_param, 1-r_param], [r_param, 1-r_param]])
    R[1] = np.array([[1-r_param, r_param], [1-r_param, r_param]])
    
    P = np.zeros((2, 2, 2, 2)) # u, a, s, s'
    P[0,0] = np.array([[P_param, 1-P_param], [P_param, 1-P_param]])
    P[0,1] = np.array([[1-P_param, P_param], [1-P_param, P_param]])
    P[1,0] = np.array([[P_param, 1-P_param], [P_param, 1-P_param]])
    P[1,1] = np.array([[1-P_param, P_param], [1-P_param, P_param]])
    
    pi = np.zeros((2, 2, 2)) #u, s, a
    pi[0] = np.array([[pi_param, 1-pi_param], [pi_param, 1-pi_param]])
    pi[1] = np.array([[1-pi_param, pi_param], [1-pi_param, pi_param]])
    
    x_dist = np.array([P_param, 1-P_param])
    u_dist = np.array([0.5, 0.5])
    gamma = 0.98
    
    return pi, P, R, x_dist, u_dist, gamma
    

#------------------------------------------------------------------------
# Toy from 227B Final Project
#------------------------------------------------------------------------

def toy227(pi_param, P_param):
    pi, P = toy227_pi_and_P(pi_param, P_param)
    R = toy227_R()
    x_dist = np.array([0.5, 0.5, 0.0])
    u_dist = np.array([0.5, 0.5])
    gamma = 0.98
    return pi, P, R, x_dist, u_dist, gamma

def toy227_pi_and_P(pi_u_param, P_u_param):
    nStates = 3
    nActions = 2
    pi_b_u0 = np.array([
        [1 - pi_u_param, pi_u_param],
        [1 - pi_u_param, pi_u_param],
        [1 - pi_u_param, pi_u_param]
    ])
    pi_b_u1 = np.array([
        [pi_u_param, 1 - pi_u_param],
        [pi_u_param, 1 - pi_u_param],
        [pi_u_param, 1 - pi_u_param]
    ])
    pi = np.zeros((2, nStates, nActions))
    pi[0] = pi_b_u0
    pi[1] = pi_b_u1

    center = 0.45
    P_u0_a0 = np.array([
        [center + P_u_param, 0.1, center - P_u_param],
        [0.1, center + P_u_param, center - P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u0_a1 = np.array([
        [center + P_u_param, 0.1, center - P_u_param],
        [0.1, center + P_u_param, center - P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u1_a0 = np.array([
        [center - P_u_param, 0.1, center + P_u_param],
        [0.1, center - P_u_param, center + P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u1_a1 = np.array([
        [center - P_u_param, 0.1, center + P_u_param],
        [0.1, center - P_u_param, center + P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P = np.zeros((2, nActions, nStates, nStates))
    P[0, 0] = P_u0_a0
    P[1, 0] = P_u1_a0
    P[0, 1] = P_u0_a1
    P[1, 1] = P_u1_a1
    
    return pi, P

def toy227_skew_pi_and_P(pi_u_param, P_u_param):
    nStates = 3
    nActions = 2
    pi_b_u0 = np.array([
        [1 - pi_u_param, pi_u_param],
        [0.8 - pi_u_param, 0.2 + pi_u_param],
        [1 - pi_u_param, pi_u_param]
    ])
    pi_b_u1 = np.array([
        [pi_u_param, 1 - pi_u_param],
        [0.2 + pi_u_param, 0.8 - pi_u_param],
        [pi_u_param, 1 - pi_u_param]
    ])
    pi = np.zeros((2, nStates, nActions))
    pi[0] = pi_b_u0
    pi[1] = pi_b_u1

    center0 = 0.45
    center1 = 0.40
    P_u0_a0 = np.array([
        [center0 + P_u_param, 0.1, center0 - P_u_param],
        [0.1, center0 + P_u_param, center0 - P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u0_a1 = np.array([
        [center1 + P_u_param, 0.2, center1 - P_u_param],
        [0.2, center1 + P_u_param, center1 - P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u1_a0 = np.array([
        [center0 - P_u_param, 0.1, center0 + P_u_param],
        [0.1, center0 - P_u_param, center0 + P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P_u1_a1 = np.array([
        [center1 - P_u_param, 0.2, center1 + P_u_param],
        [0.2, center1 - P_u_param, center1 + P_u_param],
        [0.0, 0.0, 1.0]
    ])

    P = np.zeros((2, nActions, nStates, nStates))
    P[0, 0] = P_u0_a0
    P[1, 0] = P_u1_a0
    P[0, 1] = P_u0_a1
    P[1, 1] = P_u1_a1
    
    return pi, P

def toy227_R():
    nActions = 2
    nStates = 3

    R0 = np.array([
        [0, -1, 1],
        [0, -1, 1],
        [0, 0 , 0]
    ])
    R1 = np.array([
        [0, -1, 1],
        [0, -1, 1],
        [0, 0 , 0]
    ])

    R = np.zeros((nActions, nStates, nStates))
    R[0] = R0
    R[1] = R1
    return R

#------------------------------------------------------------------------
# ope-tools graph 
#------------------------------------------------------------------------

def graph_opetools(horizon=4, slip=0.25, confound_weight=0.1):
    tx,R,x_dist = orig_graph_ope_tools(horizon, slip)
    P = R_confound(tx, R, confound_weight)
    u_dist = np.array([0.5, 0.5])
    gamma = 0.99

    nStates = tx.shape[1]
    nActions = tx.shape[0]

    pi = np.zeros((nStates, nActions))
    for s in range(nStates):
        pi[s] = [0.6, 0.4]
    pi_u = confound_pi_R(pi, tx, R, 0.3)
    return pi_u, P, R, x_dist, u_dist, gamma

def orig_graph_ope_tools(horizon=4, slip=0.25):
    nStates = 2*horizon
    nActions = 2
    tx = np.zeros((nActions, nStates, nStates))
    R = np.zeros((nActions, nStates, nStates))
    
    # starting state = 0
    x_dist = np.zeros(nStates)
    x_dist[0] = 1.0
    
    # if not sparse reward:
    #    reward = 1 into odd states
    #    reward = -1 into even states
    R[:,:,::2] = -1 
    R[:,:,1::2] = 1 
    R[:,-1,-1] = 0

    # absorbing state:
    tx[:, -1, -1] = 1

    #  if state is 2 * self.max_length - 3, reward = 1 and done
    tx[:, -2, -1] = 1
    R[:, -2, -1] = 1
    #  if state is 2 * self.max_length - 2, reward = -1 and done
    tx[:, -3, -1] = 1
    R[:, -3, -1] = -1

    #  if state is 0 then:
    #         action 0: state + 1 w prob 1-slippage else state + 2
    #         action 1: state + 2 w prob 1-slippage else state + 1
    tx[0, 0, 1] = 1 - slip
    tx[0, 0, 2] = slip
    tx[1, 0, 2] = 1 - slip
    tx[1, 0, 1] = slip

    #  other odd state:
    #       action 0: state +2 w/ prob 1-slippage else state + 3
    #       action 1: state +3 w/ prob 1-slippage else state + 2
    for i in range(1,nStates-3,2):
        tx[0, i, i+2] = 1 - slip
        tx[0, i, i+3] = slip
        tx[1, i, i+3] = 1 - slip
        tx[1, i, i+2] = slip

    #  other even state:
    #       action 0: state +1 w/ prob 1-slippage else state +2
    #       action 1: state +2 w/ prob 1-slippage else state +1
    for i in range(2,nStates-3,2):
        tx[0, i, i+1] = 1 - slip
        tx[0, i, i+2] = slip
        tx[1, i, i+2] = 1 - slip
        tx[1, i, i+1] = slip
        
    return tx, R, x_dist

#------------------------------------------------------------------------
# ope-tools toy mc 
#------------------------------------------------------------------------

def toymc_opetools(n_left=10, n_right=10, horizon=100, slip=0.25, confound_weight=0.1):
    tx,R,x_dist = orig_toy_mc_ope_tools(n_left=10, n_right=10, horizon = horizon)

    V = rand_pi_val(tx, R, x_dist, 100)
    P = confound_V(tx, x_dist, V, confound_weight=confound_weight)
 
    u_dist = np.array([0.5, 0.5])
    gamma = 0.99

    nStates = tx.shape[1]
    nActions = tx.shape[0]

    pi = np.zeros((nStates, nActions))
    for s in range(nStates):
        pi[s] = [0.6, 0.4] 
    pi_u = confound_pi_V(pi, tx, V, 0.3)
    return pi_u, P, R, x_dist, u_dist, gamma

def orig_toy_mc_ope_tools(n_left=10, n_right=10, horizon = 100):
    nStates = n_left + n_right + 2
    nActions = 2
    
    tx = np.zeros((nActions, nStates, nStates))

    # if action == 0
    #  if state not -n_left
    #      then state -= 1 (i.e. move left if possible)
    for i in range(1,nStates-1):
        tx[0, i, i-1] = 1.0
    tx[0, 0, 0] = 1.0
    # if action == 1
    #      then state += 1 (i.e. move right)
    for i in range(0,nStates-1):
        tx[1, i, i+1] = 1
    # state = n_right + 1 is absorbing
    tx[:, -1, -1] = 1.0
    
    R = np.zeros((nActions, nStates, nStates))

    # if state = n_right + 1 then reward 0
    # else reward -1
    R[:, :, :] = -1
    R[:, -1, -1] = 0
    
    # starting state = center
    x_dist = np.zeros(nStates)
    x_dist[n_left] = 1.0
    
    return tx, R, x_dist

#------------------------------------------------------------------------
# ope-tools gridworld
#------------------------------------------------------------------------

def gridworld_opetools(horizon = 100, slip = 0.05, confound_weight=0.1, infinite=False, small=True, soft=False):
    if infinite:
        tx,R,x_dist = infty_gridworld_ope_tools(horizon = horizon, slip = slip, small=small)
    else:
        tx,R,x_dist = orig_gridworld_ope_tools(horizon = horizon, slip = slip, small=small, soft=soft)

    V = rand_pi_val(tx, R, x_dist, 100)
    P = confound_V(tx, x_dist, V, confound_weight=confound_weight)
 
    u_dist = np.array([0.5, 0.5])
    gamma = 0.99

    nStates = tx.shape[1]
    nActions = tx.shape[0]

    pi = np.zeros((nStates, nActions))
    for s in range(nStates):
        pi[s] = [0.4, 0.1, 0.1, 0.4]
    pi_u = confound_pi_V(pi, tx, V, 0.2)
    return pi_u, P, R, x_dist, u_dist, gamma

def orig_gridworld_ope_tools(horizon = 100, slip = 0.05, small=True, soft=False):
    h = -0.5
    f = -0.005

    if small:
        grid = np.array(
        [[-0.01, -0.01, -0.01, -0.01],
         [-0.01, -0.01, f    , h    ],
         [-0.01, h    , -0.01, h    ],
         [-0.01, h    , f    , +1   ]])

    else:
        grid = np.array(
            [[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
             [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
             [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1]])
    
    gridlen = grid.shape[0]
    nStates = np.prod(grid.shape)
    nActions = 4
    
    # actions:
    #    0 : state -= 1 unless state % grid.shape[1] == 0
    #    1 : state += 1 unless state % grid.shape[1] == 7
    #    2 : state -= 8 unless state // grid.shape[0] == 0
    #    3 : state += 8 unless state // grid.shape[0] == 7
    
    tx = np.zeros((nActions, nStates, nStates))
    for i in range(nStates-1):
        if i % grid.shape[1] == 0:
            tx[0, i, i] += 1 - 3*slip
            tx[1, i, i] += slip
            tx[2, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[0, i, i-1] += 1 - 3*slip
            tx[1, i, i-1] += slip
            tx[2, i, i-1] += slip
            tx[3, i, i-1] += slip
        if i % grid.shape[1] == gridlen-1:
            tx[1, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[2, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[1, i, i+1] += 1 - 3*slip
            tx[0, i, i+1] += slip
            tx[2, i, i+1] += slip
            tx[3, i, i+1] += slip
        if i // grid.shape[0] == 0:
            tx[2, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[1, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[2, i, i-gridlen] += 1 - 3*slip
            tx[0, i, i-gridlen] += slip
            tx[1, i, i-gridlen] += slip
            tx[3, i, i-gridlen] += slip
        if i // grid.shape[0] == gridlen-1:
            tx[3, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[1, i, i] += slip
            tx[2, i, i] += slip
        else:
            tx[3, i, i+gridlen] += 1 - 3*slip
            tx[0, i, i+gridlen] += slip
            tx[1, i, i+gridlen] += slip
            tx[2, i, i+gridlen] += slip
            
    # absorbing state
    tx[:, -1, -1] = 1.0 

    R = np.zeros((nActions, nStates, nStates))
    for i in range(nStates):
        R[:, i, :] = grid.flatten()
    R[:, -1, -1] = 0

    if not soft:
        x_dist = np.zeros(nStates)
        init_pos = [0, 1, 2, 3, 4, 8, 12]
        x_dist[init_pos] = 1/len(init_pos)
    else:
        x_dist = np.ones(nStates)/nStates
    
    return tx, R, x_dist

def infty_gridworld_ope_tools(horizon = 100, slip = 0.05, small=True):
    h = -0.5
    f = -0.005

    if small:
        grid = np.array(
            [[-0.01, -0.01, -0.01, -0.01],
             [-0.01, -0.01, f    , h    ],
             [-0.01, h    , -0.01, h    ],
             [-0.01, h    , f    , +1   ]])

    else:
        grid = np.array(
            [[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
             [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
             [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1]])
    
    gridlen = grid.shape[0]
    
    nStates = np.prod(grid.shape)
    nActions = 4
    
    # actions:
    #    0 : state -= 1 unless state % grid.shape[1] == 0
    #    1 : state += 1 unless state % grid.shape[1] == 7
    #    2 : state -= 8 unless state // grid.shape[0] == 0
    #    2 : state += 8 unless state // grid.shape[0] == 7
    
    tx = np.zeros((nActions, nStates, nStates))
    for i in range(nStates):
        if i % grid.shape[1] == 0:
            tx[0, i, i] += 1 - 3*slip
            tx[1, i, i] += slip
            tx[2, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[0, i, i-1] += 1 - 3*slip
            tx[1, i, i-1] += slip
            tx[2, i, i-1] += slip
            tx[3, i, i-1] += slip
        if i % grid.shape[1] == gridlen-1:
            tx[1, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[2, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[1, i, i+1] += 1 - 3*slip
            tx[0, i, i+1] += slip
            tx[2, i, i+1] += slip
            tx[3, i, i+1] += slip
        if i // grid.shape[0] == 0:
            tx[2, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[1, i, i] += slip
            tx[3, i, i] += slip
        else:
            tx[2, i, i-gridlen] += 1 - 3*slip
            tx[0, i, i-gridlen] += slip
            tx[1, i, i-gridlen] += slip
            tx[3, i, i-gridlen] += slip
        if i // grid.shape[0] == gridlen-1:
            tx[3, i, i] += 1 - 3*slip
            tx[0, i, i] += slip
            tx[1, i, i] += slip
            tx[2, i, i] += slip
        else:
            tx[3, i, i+gridlen] += 1 - 3*slip
            tx[0, i, i+gridlen] += slip
            tx[1, i, i+gridlen] += slip
            tx[2, i, i+gridlen] += slip
            
    # absorbing state
    #tx[:, -1, -1] = 1.0 

    R = np.zeros((nActions, nStates, nStates))
    for i in range(nStates):
        R[:, i, :] = grid.flatten()
    R[:, -1, -1] = 0

    x_dist = np.zeros(nStates)
    init_pos = [0, 1, 2, 3, 4, 8, 12]
    x_dist[init_pos] = 1/len(init_pos)
    
    return tx, R, x_dist

#------------------------------------------------------------------------
# angela zhou toy
#------------------------------------------------------------------------

def randwalk(pu1 = 1/6, pu2 = 1/2.5, pis1u1 = 1.0/4, pis2u1 = 1.0/6):

    P, pi_u = confounded_random_walk(pu1 = pu1, pu2 = pu2, pis1u1 = pis1u1, pis2u1 = pis2u1)
    x_dist = np.array([0.6,0.4])
    u_dist = np.array([0.5, 0.5])
    gamma = 0.99

    R = np.zeros((2, 2, 2))
    for s in range(2):
        for a in range(2):
            R[a,s,:] = [1,2]

    return pi_u, P, R, x_dist, u_dist, gamma

def confounded_random_walk(pu1 = 1/6, pu2 = 1/2.5, pis1u1 = 1.0/4, pis2u1 = 1.0/6):
    nStates = 2
    nActions = 2
    nU = 2
    
    reshape_byxrow = lambda a,nU: a.reshape(-1,nU,a.shape[-1]).sum(1)
    
    Pi = np.array(# s=a,u1; s=a,u2; s=b,u1; s=b,u2
        [[pis1u1, 1-pis1u1, pis2u1, 1-pis2u1],
            [1-pis1u1, pis1u1, 1-pis2u1, pis2u1]]) 
    P = np.zeros([4,2, 4])
    P[:,0,:] = np.asarray([np.asarray([pu1, 
         pu1, (1/2 - pu1), (1/2 - pu1)]), np.asarray([(1/2 - pu2), (1/2 - pu2), pu2, pu2]), 
    np.asarray([pu1, 
         pu1, (1/2 - pu1), (1/2 - pu1)]), np.asarray([(1/2 - pu2), (1/2 - pu2), pu2, pu2])]);
    P[:,1,:] = np.asarray([np.asarray([(1/2 - pu1), (1/2 - pu1), pu1, pu1]), 
    np.asarray([pu2, 
         pu2, (1/2 - pu2), (1/2 - pu2)]), np.asarray([(1/2 - pu1), (1/2 - pu1), pu1, pu1]), 
    np.asarray([pu2, pu2, (1/2 - pu2), (1/2 - pu2)])]);
    
    P_u = np.zeros((nU, nActions, nStates, nStates))
    for a in range(nActions):
        Pagg = reshape_byxrow(P[:,a,:].T , nU).T
        for i in range(nU):
            P_u[i, a] = Pagg[i::nU, :]
    
    pi_u = np.zeros((nU, nStates, nActions))
    for u in range(nU):
        for a in range(nActions):
            pi_u[u, :, a] = Pi[a, u::nU]
    
    return P_u,pi_u