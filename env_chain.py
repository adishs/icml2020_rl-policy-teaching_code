import numpy as np
import copy
from scipy import sparse
import MDPSolver
import teacher
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, M):

        self.n_states = M[0]
        self.n_actions = M[1]
        self.reward = M[2]
        self.T = M[3]
        self.actions = {"left": 0, "right": 1}
        self.T_sparse_list = self.get_transition_sparse_list()
        self.InitD = self.get_init_D()
        self.terminal_state = None
    #enddef

    def get_init_D(self):
        InitD = np.zeros(self.n_states)
        InitD[0] = 1
        return InitD
    #enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))  # T_0
        return T_sparse_list
    # endef

    def get_M_0(self):
        M_0 = (self.n_states, self.n_actions, self.reward, self.T)
        return M_0
    #enddef
    
    def draw_policy(self, pi):
        string="["
        for s in range(self.n_states):
            if pi[s] == 0:
                string += "<-,"
            if pi[s] == 1:
                string += "->,"
        string += ']'
        print(string)
    #enddef

def get_transition_matrix_line(n_states, success_prob):
    unif_prob = (1 - success_prob) / (n_states - 1)
    P = np.zeros((n_states, n_states, 2))
    for s in range(n_states):
        if s > 0 and s < n_states-1:
            # left action
            P[s, :, 0] = unif_prob
            P[s, s-1, 0] = success_prob
            # right action
            P[s, :, 1] = unif_prob
            P[s, s+1, 1] = success_prob
        if s ==0:
            # left action
            P[s, :, 0] = unif_prob
            P[s, s, 0] = success_prob
            # right action
            P[s, :, 1] = unif_prob
            P[s, s+1, 1] = success_prob
        if s == n_states-1:
            # left action
            P[s, :, 0] = unif_prob
            P[s, s-1, 0] = success_prob
            # right action
            P[s, :, 1] = unif_prob
            P[s, s, 1] = success_prob
    return P
#enddef


########################################
if __name__ == "__main__":
    pass
