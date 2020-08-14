import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import os
import MDPSolver

class learner:
    def __init__(self, env, type="offline"):
        self.env = env
        self.type = type
        self.accumulator_dict = {}
        if type == "offline":
            _, self.pi_d, pi_s = MDPSolver.averaged_valueIteration(env, env.reward)
        elif type == "online":
            pass
        else:
            print("unknown learner type: ", type)
            exit(0)
    #enddef

    def get_conf_r_t(self, N_t, alpha, t):
        env = copy.deepcopy(self.env)
        conf_r_t = np.zeros((env.n_states, env.n_actions))

        for s in range(env.n_states):
            for a in range(env.n_actions):
                if N_t[s, a] == 0 or t == 0:
                    conf_r_t[s, a] = 1
                else:
                    conf_r_t[s, a]= min(1, np.sqrt((np.log2(4*(t**alpha)*(env.n_states**2)*env.n_actions))
                                      /(2*N_t[s, a])))
        return conf_r_t
    #enddef

    def get_conf_p_t(self, N_t, alpha, t):
        env = copy.deepcopy(self.env)
        conf_p_t = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in range(env.n_actions):
                if N_t[s, a] == 0 or t == 0:
                    conf_p_t[s, a] = 1
                else:
                    conf_p_t[s, a] = min(1, np.sqrt((np.log2(2 * (t ** alpha) * env.n_states * env.n_actions))
                                                  / (2 * N_t[s, a])))
        return conf_p_t
    #enddef

    def get_r_hat(self, N_t, R_t):
        r_hat = np.zeros((self.env.n_states, self.env.n_actions))

        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                r_hat[s, a] = R_t[s, a] / N_t[s, a] if N_t[s, a] > 0 else 1
        return r_hat
    #enddef
    
    def get_p_hat(self, N_t, P_t):
        p_hat = np.zeros((self.env.n_states, self.env.n_states, self.env.n_actions))
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                for s_n in range(self.env.n_states):
                    p_hat[s, s_n, a] = P_t[s, s_n, a] / N_t[s, a] if N_t[s, a] > 0 else 1 / self.env.n_states
        return p_hat
    #enddef

    def compute_expected_action_diff_array(self, pi_star, s_t, a_t):
        accumulator = 0
        if pi_star[s_t] != a_t:
            accumulator += 1
        return accumulator
    #enddef

    def compute_accumulated_cost(self, s_t, a_t, M_0, M_t, cost_p):

        TV_r = np.abs((M_0[2][s_t, a_t] - M_t[2][s_t, a_t]))
        TV_P = np.sum(np.abs(M_0[3][s_t, :, a_t] - M_t[3][s_t, :, a_t]))
        # TV_P = np.max(np.abs(M_0[3][s_t, :, a_t] - M_t[3][s_t, :, a_t]))
        if cost_p == 1:
            accumulator = TV_r + TV_P
        elif cost_p == 2:
            accumulator = np.power(TV_r, cost_p) + np.power(TV_P, cost_p)
        else:
            print("cost_p should be 1 or 2")
            exit(0)

        return accumulator
    #enddef

    def compute_accumulated_cost_no_attack(self, s_t, a_t, M_0, M_t, cost_p):

        TV_r = np.abs((M_0[2][s_t, a_t] - M_0[2][s_t, a_t]))
        TV_P = np.sum(np.abs(M_0[3][s_t, :, a_t] - M_0[3][s_t, :, a_t]))
        # TV_P = np.max(np.abs(M_0[3][s_t, :, a_t] - M_0[3][s_t, :, a_t]))
        if cost_p == 1:
            accumulator = TV_r + TV_P
        elif cost_p == 2:
            accumulator = np.power(TV_r, cost_p) + np.power(TV_P, cost_p)
        else:
            print("cost_p should be 1 or 2")
            exit(0)

        return accumulator
    #enddef

    def UCRL(self, alpha, n_rounds, pi_no_attack, M_0, M_t, cost_p):
        regret_array = []
        regret_learner_array = []
        expected_action_diff_array = []
        expected_action_diff_array_no_attack = []
        accumulated_cost_array = []
        accumulated_cost_array_no_attack = []
        regret_r_accumulator_learner = 0
        regret_r_accumulator_learner_no_attack = 0
        diff_action_count = 0
        accumulated_cost_count = 0
        accumulated_cost_no_attack_count = 0
        #init
        #1)
        env = copy.deepcopy(self.env)
        n_states = env.n_states
        t = 0
        N_t = np.zeros((env.n_states, env.n_actions))
        R_t = np.zeros((env.n_states, env.n_actions))
        P_t = np.zeros((env.n_states, env.n_states, env.n_actions))
        s_t = np.random.choice(np.arange(env.n_states, dtype="int"), size=1, p=env.InitD)[0]
        # confidence
        conf_r = self.get_conf_r_t(N_t, alpha, t)
        conf_p = self.get_conf_p_t(N_t, alpha, t)

        r_hat = np.zeros((env.n_states, env.n_actions))
        p_hat = np.zeros((env.n_states, env.n_states, env.n_actions))
        #2)
        pi = MDPSolver.extended_averaged_value_iteration(env, r_hat, conf_r, p_hat, conf_p)
        _, pi_star, _ = MDPSolver.averaged_valueIteration(env, env.reward)
        regret_r_accumulator_opt = MDPSolver.compute_averaged_reward_given_policy(env, env.reward, pi_star)
        mu_opt = MDPSolver.get_stationary_dist(env, pi_star)
        print("t={}".format(t))
        for t in range(1,int(n_rounds)):
            a_t = pi[s_t]
            r_t = env.reward[s_t, a_t]
            s_t_plus_1 = env.get_next_state(s_t, a_t)
            regret_r_accumulator_learner += r_t

            regret_learner_array.append(regret_r_accumulator_learner/t)
            regret_array.append(t * regret_r_accumulator_opt - regret_r_accumulator_learner)
            diff_action_count += self.compute_expected_action_diff_array(pi_star, s_t, a_t)
            expected_action_diff_array.append((diff_action_count) / t)
            regret_r_accumulator_learner_no_attack += self.compute_expected_action_diff_array(pi_no_attack, s_t, a_t)
            expected_action_diff_array_no_attack.append(regret_r_accumulator_learner_no_attack/t)
            accumulated_cost_count += self.compute_accumulated_cost(s_t, a_t, M_t, M_0, cost_p)
            accumulated_cost_array.append(np.power(accumulated_cost_count, 1/cost_p)/t)
            accumulated_cost_no_attack_count += self.compute_accumulated_cost_no_attack(s_t, a_t, M_t, M_0, cost_p)
            accumulated_cost_array_no_attack.append(np.power(accumulated_cost_no_attack_count, 1/cost_p)/t)

            #Update
            N_t[s_t, a_t] = N_t[s_t, a_t] + 1
            R_t[s_t, a_t] = R_t[s_t, a_t] + r_t
            P_t[s_t, s_t_plus_1, a_t] = P_t[s_t, s_t_plus_1, a_t] + 1
            s_t = copy.deepcopy(s_t_plus_1)


            if ((self.get_conf_r_t(N_t, alpha, t) - conf_r / 2) < 0).any() and \
                    ((self.get_conf_r_t(N_t, alpha, t) - conf_p / 2) < 0).any():

                conf_r = self.get_conf_r_t(N_t, alpha, t)
                conf_p = self.get_conf_p_t(N_t, alpha, t)
                r_hat = self.get_r_hat(N_t, R_t)
                p_hat = self.get_p_hat(N_t, P_t)
                pi = MDPSolver.extended_averaged_value_iteration(env, r_hat, conf_r, p_hat, conf_p)
                print("t={}".format(t))
                print("policy=", end="")
                env.draw_policy(pi)
                # print("expected reward = {}".format(MDPSolver.compute_averaged_reward_given_policy(env, env.reward, pi)))
        print("learner's policy = ", end="")
        env.draw_policy(pi)
        # print("learner's averaged reward =", MDPSolver.compute_averaged_reward_given_policy(env, env.reward, pi))
        # plt.plot(regret_array)
        self.accumulator_dict["regret_array"] = regret_array
        self.accumulator_dict["expected_action_diff_array"] = expected_action_diff_array
        self.accumulator_dict["regret_learner_array"] = regret_learner_array
        self.accumulator_dict["expected_action_diff_array_no_attack"] = expected_action_diff_array_no_attack
        self.accumulator_dict["accumulated_cost_array"] = accumulated_cost_array
        self.accumulator_dict["accumulated_cost_array_no_attack"] = accumulated_cost_array_no_attack
    #enddef
#enddef


if __name__ == "__main__":
    pass
