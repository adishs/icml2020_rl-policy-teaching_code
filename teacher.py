import numpy as np
import MDPSolver
import learner
import sys
sys.path.append('code-attacker/')
from reward_attack import *
from dynamic_attack import *
from utils import *
import copy

class teacher:
    def __init__(self, env, target_pi, epsilon, p, epsilon_p, teacher_type, pool=None):
        self.env = env
        self.target_pi = target_pi
        self.epsilon = epsilon
        self.p = p
        self.epsilon_p = epsilon_p
        self.teacher_type = teacher_type
        self.pool= pool
        # self.V_orig, self.pi_orig_d, self.pi_orig_s = self.get_pi_star_for_original_env(env)
        self.pi_T = self.change_policy_to_pi_T()
        self.M_0 = (env.n_states, env.n_actions, env.reward, env.T)
    #enddef

    def get_target_M(self,  M_0):
        if self.teacher_type == "general_attack_on_reward":
            return self.general_attack_on_reward( M_0, self.target_pi, self.epsilon, self.p)
        elif self.teacher_type == "non_target_attack_on_reward":
            return self.non_target_attack_on_reward(M_0, self.target_pi, self.epsilon, self.p)
        elif self.teacher_type == "non_target_attack_on_dynamics":
            return self.non_target_attack_on_dynamics(M_0, self.target_pi, self.epsilon, self.epsilon_p)
        elif self.teacher_type == "general_attack_on_dynamics":
            return self.general_attack_on_dynamics(M_0, self.target_pi, self.epsilon, self.epsilon_p)
        else:
            print("Wrong teacher type!!---", self.teacher_type)
            print("Please choose one of the following:")
            print("{}\n{}".format("general_attack_on_reward", "non_target_attack_on_reward"
                                  "general_attack_on_dynamics", "non_target_attack_on_dynamics"))
            exit(0)
        #enddef

    def get_pi_star_for_original_env(self, env):
        V, expert_policy_deterministic, expert_policy_stochastic = \
            MDPSolver.averaged_valueIteration(env, env.reward)
        return V, expert_policy_deterministic, expert_policy_stochastic
    #enddef

    def change_policy_to_pi_T(self):
        pi_T = np.zeros(self.env.n_states, dtype="int")
        pi_T[:] = 1
        return pi_T
    #enddef

    def general_attack_on_reward(self, M_0, pi_t, epsilon, p):
        n_states, n_action, R_T, P = reward_attack_general(M_0, pi_t, epsilon, p)
        return (n_states, n_action, R_T, P), True
    #enddef

    def non_target_attack_on_reward(self, M_0, pi_t, epsilon, p):
        n_states, n_action, R_T, P = reward_attack_nontargetonly(M_0, pi_t, epsilon, p)
        return (n_states, n_action, R_T, P), True
    #enddef

    def non_target_attack_on_dynamics(self, M_0, pi_t, epsilon, epsilon_p):
        M, feasible = dynamic_attack_nontargetonly( M_0, pi_t, epsilon, epsilon_p)
        n_states, n_action, R, P_T = M[0], M[1], M[2], M[3]
        return (n_states, n_action, R, P_T), feasible
    #enddef

    def general_attack_on_dynamics(self, M_0, pi_t, epsilon, epsilon_p):
        p = self.p
        feasible = False
        num_states, num_actions, R, P_in = M_0[0], M_0[1], M_0[2], M_0[3]

        pool = self.pool
        pool_of_solutions, _, _ = self.solve_pool(num_states, num_actions,
                                                R, epsilon, epsilon_p, pool)
        if len(pool_of_solutions) > 0:
            feasible = True
        closest_P = self.get_P_with_smallest_norm(pool_of_solutions, P_in, p)

        n_states, n_action, R, P_T = num_states, num_actions, R, closest_P
        return (n_states, n_action, R, P_T), feasible
    #enddef

    def non_target_attack_on_dynamics_upperbound(self,M_0, pi_t, epsilon, epsilon_p):
        M, feasible = dynamic_attack_nontargetonly_upperbound(M_0, pi_t, epsilon, epsilon_p)
        n_states, n_action, R, P_T = M[0], M[1], M[2], M[3]
        return n_states, n_action, R, P_T, feasible
    # enddef

    def solve_pool(self, num_states, num_actions, R, epsilon, epsilon_p, pool):
        pool_of_solved_P = []
        pool_of_infeasible_P = []
        pool_of_solutions = []
        target_pi = self.target_pi
        for P in pool:
            M = (num_states, num_actions, R, P)
            M_t, feasible = self.non_target_attack_on_dynamics(M, target_pi, epsilon, epsilon_p)
            if feasible:
                pool_of_solved_P.append(P)
                pool_of_solutions.append(M_t[3])
            else:
                pool_of_infeasible_P.append(P)
        return pool_of_solutions, pool_of_solved_P, pool_of_infeasible_P
    #enddef

    def get_P_with_smallest_norm(self, pool_of_solutions_P, P_0, p):
        minimum = np.inf
        P_closest = None
        for P in pool_of_solutions_P:
            value = self.norm_p(P, P_0, p)
            if value < minimum:
                minimum = copy.deepcopy(value)
                P_closest = copy.deepcopy(P)
        return P_closest
    #enddef

    def norm_p(self, P, P_0, p):
        P_s_a = np.zeros((P.shape[0], P.shape[2]))
        for s in range(P.shape[0]):
            for a in range(P.shape[2]):
                P_s_a[s, a] = np.sum(np.abs(P[s, :, a]-P_0[s, :, a]))
                # P_s_a[s, a] = np.max(np.abs(P[s, :, a] - P_0[s, :, a]))

        return np.linalg.norm(P_s_a.flatten(), ord=p)
    #enddef

    def cost(self, M_0, M_t, p):
        return np.linalg.norm((M_0[2]-M_t[2]).flatten(), ord=p) + self.norm_p(M_0[3], M_t[3], p=p)
    #enddef
#enddef

def normalize(vector):
    return vector/sum(vector)
#enddef

def create_perturb_P_for_target(num_states, num_actions, R, P_in, pi, alpha, beta, N):
    P_out = copy.deepcopy(P_in)
    for i in range(N):
        s = np.random.choice(np.arange(0, num_states, dtype="int"), size=1)[0]
        s_prime = np.random.choice(np.arange(0, num_states, dtype="int"), size=1)[0]

        P_tmp = copy.deepcopy(P_out)
        P_tmp[s, s_prime, pi[s]] = P_tmp[s, s_prime, pi[s]] + alpha
        P_tmp[s, :, pi[s]] = normalize(P_tmp[s, :, pi[s]])

        M_tmp = (num_states, num_actions, R, P_tmp)
        rho_tmp = calc_rho(M_tmp, pi)
        M_out = (num_states, num_actions, R, P_out)
        rho_out = calc_rho(M_out, pi)

        if (rho_tmp - rho_out) > beta:
            P_out = copy.deepcopy(P_tmp)
    return P_out
#enddef

def generate_pool(num_states, num_actions, R, P_in, pi, alpha=0.1, beta=0.0001, n_copies_of_N=5):
    N_array = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]
    pool = []
    for n_copy in range(n_copies_of_N):
        for N in N_array:
            P_out = create_perturb_P_for_target(num_states, num_actions, R, P_in, pi, alpha, beta, N)
            pool.append(P_out)
    return pool
#enddef


if __name__ == "__main__":
    pass