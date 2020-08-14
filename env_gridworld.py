import numpy as np
import copy
from scipy import sparse
import MDPSolver
import teacher
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, M=None):
        if M is not None:
            self.n_states = M[0]
            self.n_actions = M[1]
            self.reward = M[2]
            self.T = M[3]
            self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
            self.T_sparse_list = self.get_transition_sparse_list()
            self.InitD = self.get_init_D()
        else:
            self.n_states = 9
            self.n_actions = 2
            self.reward = self.get_reward()
            self.T = self.get_transition_matrix()
            self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
            self.T_sparse_list = self.get_transition_sparse_list()
            self.InitD = self.get_init_D()
        self.terminal_state = None
    #enddef

    def get_M_0(self):
        M_0 = (self.n_states, self.n_actions, self.reward, self.T)
        return M_0
    #enddef

    def draw_policy(self, pi):
        string="["
        for s in range(self.n_states):
            if pi[s] == 0:
                string += "^,"
            if pi[s] == 1:
                string += "<-,"
            if pi[s] == 2:
                string += "↓,"
            if pi[s] == 3:
                string += "->"
        string += ']'
        print(string)
    #enddef

    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))  # T_0
        return T_sparse_list
    # endef

    def get_init_D(self):
        InitD = np.ones(self.n_states)
        return InitD/sum(InitD)
    #enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef


    def get_transition_matrix(self, n_states=9, n_actions=2, success_prob=0.9):

        unif_prob = (1 - success_prob) / (n_states - 1)
        P = np.zeros((n_states, n_states, n_actions))


        P[:,:,0] = np.array(
            [
            # UP
            #0->1
            [unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 1->2
            [unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 2->3
            [unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 3->4
            [unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob],

            # Left
            #4-->7
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob],
            #5-->4
            [unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 6-->5
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob],
            # 7-->8
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob],
            # 8-->8
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob],

        ])


        P[:,:,1] = np.array(
            [
            # Down
            #0->0
            [success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 1->0
            [success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 2->1
            [unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 3->2
            [unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob],

            # Right
            #4-->5
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob],
            #5-->6
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob],
            # 6-->6
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob],
            # 7-->4
            [unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob, unif_prob, unif_prob, unif_prob],
            # 8-->7
            [unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, unif_prob, success_prob, unif_prob],

        ])

        return P
    #enddef

    def get_reward(self, n_states=9, n_actions=2):
        reward = np.zeros((n_states, n_actions))
        reward[0, :] = -2.5
        reward[1, :] = -2.5
        reward[2, :] = -2.5
        reward[3, :] = -2.5
        reward[4, :] = 1
        reward[5, :] = 1
        return reward
    # enddef

    # endclass

def get_target_pi():
    env = Environment()
    pi_d = np.zeros(env.n_states, dtype=int)
    return pi_d
# enddef



if __name__ == "__main__":

    env = Environment()

    V, pi_d, _ = MDPSolver.averaged_valueIteration(env, env.reward)

    pi_dagger = env.get_target_pi()




    print("==== Orig pi ====")
    for s in range(len(pi_d)):
        print(s, "=", pi_d[s], end="\t")
    print()

    M_0 = (env.n_states, env.n_actions, env.reward, env.T)

    target_pi = get_target_pi()

    print("==== Target pi ====")
    for s in range(len(target_pi)):
        print(s, "=", target_pi[s], end="\t")
    print()

    exit(0)


