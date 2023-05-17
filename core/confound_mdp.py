import numpy as np

class ConfoundMDP(object):
    def __init__(self, P, R, x_dist, u_dist, gamma):
        self.P = P  # U by A by S by S
        self.R = R

        self.n_actions = P.shape[1]
        self.n_states = P.shape[2]
        self.n_confound = P.shape[0]
        self.gamma = gamma

        self.x_dist = x_dist
        self.u_dist = u_dist

        self.reset()

    def reset(self):
	    self.state = np.random.choice(self.n_states, p=self.x_dist)
	    self.done = False
	    return self.state

    def step(self, a):
        x = self.state
        xp = np.random.choice(self.n_states, p=self.P[self.u,a,x])
        r = self.R[a, x, xp]
        self.state = xp
        return self.state, r

    def generate_trajectory(self, pi_b, horizon, iid=True):
        self.reset()
        traj = []
        if not iid:
            u = np.random.choice(self.n_confound, p=self.u_dist)
            self.u = u
        for t in range(horizon):
            x = self.state
            if iid:
                u = np.random.choice(self.n_confound, p=self.u_dist)
                self.u = u
            a = np.random.choice(self.n_actions, p=pi_b[u, x])
            xp, r = self.step(a)
            traj.append([x,a,u,xp,r])
        return np.array(traj)

    def bellman_eval_update(self, f, pi):
        R = self.R
        P = self.P
        nStates = self.n_states
        nActions = self.n_actions
        nU = self.n_confound
        gamma = self.gamma
        Tf = np.zeros(f.shape)
        for s in range(nStates):
            for a in range(nActions):
                f_pi_avg = 0
                for u in range(nU):
                    f_pi_u = np.array([pi[u, xp] @ f[xp, :] for xp in range(nStates)])
                    f_pi_avg += f_pi_u * self.u_dist[u]
                for u in range(nU):
                    Tf[s,a] += P[u, a, s] @ (R[a, s] + gamma * f_pi_avg) * self.u_dist[u]
        return Tf

    def bellman_eval(self, pi, horizon):
        Q = np.zeros((self.n_states, self.n_actions))
        for k in range(horizon):
            Q = self.bellman_eval_update(Q, pi)
        return Q

    def get_value(self, Q, pi):
        V = np.array([Q[x] @ pi[x] for x in range(self.n_states)])
        avgV = V @ self.x_dist
        return V, avgV

    def generate_persist_trajectory(self, pi_b, horizon):
        self.reset()
        u = np.random.choice(self.n_confound, p=self.u_dist)
        self.u = u
        traj = []
        for t in range(horizon):
            x = self.state
            a = np.random.choice(self.n_actions, p=pi_b[u, x])
            xp, r = self.step(a)
            traj.append([x,a,u,xp,r])
        return np.array(traj)

"""     def bellman_eval_update_u(self, f, pi):
        R = self.R
        P = self.P
        nStates = self.n_states
        nActions = self.n_actions
        gamma = self.gamma
        Tf = np.zeros(f.shape)
        for s in range(nStates):
            for a in range(nActions):
                for u in range(self.n_confound):
                    f_pi = np.array([pi[u, xp] @ f[xp, :, u] for xp in range(nStates)])
                    Tf[s,a, u] = P[u, a, s] @ (R[a, s] + gamma * f_pi)
        return Tf

    def bellman_eval_u(self, pi, horizon):
        Q = np.zeros((self.n_states, self.n_actions, self.n_confound))
        for k in range(horizon):
            Q = self.bellman_eval_update_u(Q, pi)
        return Q[:,:,0] * self.u_dist[0] + Q[:,:,1] * self.u_dist[1] """


def collect_sample(nsamples, mdp, pi_b, horizon, iid=True):
    dataset = []
    for _ in range(nsamples):
        traj = mdp.generate_trajectory(pi_b, horizon, iid)
        dataset.append(traj)
    dataset = np.array(dataset)
    # x, a, u, x', r
    return dataset

def collect_persist_sample(nsamples, mdp, pi_b, horizon):
    dataset = []
    for _ in range(nsamples):
        traj = mdp.generate_persist_trajectory(pi_b, horizon)
        dataset.append(traj)
    dataset = np.array(dataset)
    # x, a, u, x', r
    return dataset

def calc_returns(data, gamma, horizon):
    rewards = data[:,:,-1]
    g = np.array([gamma**t for t in range(horizon)])
    return (rewards * g).sum(axis=1)