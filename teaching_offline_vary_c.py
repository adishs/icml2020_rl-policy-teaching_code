import numpy as np
import copy
import sys
import os
import plot_chain
import MDPSolver
import env_chain as environment
import learner
import teacher
import matplotlib.pyplot as plt

class teaaching:
    def __init__(self, M_0, settings_to_run, teachers_to_run, target_pi):
        self.M_0 = M_0
        self.settings_to_run = settings_to_run
        self.teachers_to_run = teachers_to_run
        self.target_pi = target_pi
        self.accumulator = {}
    #enddef

    def offline_attack(self):
        for setting in self.settings_to_run:
            R_c, success_prob = setting[0], setting[1]
            M_in = get_M(self.M_0[0], self.M_0[1], R_c, success_prob)
            env_in = environment.Environment(M_in)
            pool = teacher.generate_pool(M_in[0], M_in[1], M_in[2], M_in[3], self.target_pi)
            for tchr in self.teachers_to_run:
                target_pi = tchr[2]["target_pi"]
                p = tchr[1]
                epsilon = tchr[2]["epsilon"]
                epsilon_p = tchr[2]["epsilon_p"]
                teacher_type = tchr[0]
                cost_p = tchr[2]["cost_p"]
                teacher_obj = teacher.teacher(env=env_in, target_pi=target_pi, p=p, epsilon=epsilon, epsilon_p=epsilon_p, teacher_type=teacher_type, pool=pool) #Pool here

                print("==================================================")
                try:
                    M_out, feasible = teacher_obj.get_target_M(M_in)
                except Exception as e:

                    print("--teacher_type={}--R_c={}--P_success={}".format(teacher_type, R_c, success_prob))

                if not feasible:
                    print("--teacher_type={}--R_c={}--P_success={}".format(teacher_type, R_c, success_prob))
                    cost = self.max_cost_value_if_non_feasible(cost_p)
                    self.append_cost_to_accumulator(cost, teacher_type, p, cost_p, success_prob, R_c)
                    continue
                else:
                    print("--teacher_type={}--R_c={}--P_success={}".format(teacher_type, R_c, success_prob))

                env_out = environment.Environment(M_out)
                _, pi_T, _ = MDPSolver.averaged_valueIteration(env_out, env_out.reward)
                cost = teacher_obj.cost(M_in, M_out, cost_p)
                self.append_cost_to_accumulator(cost, teacher_type, p, cost_p, success_prob, R_c)
        return self.accumulator
    #enddef

    def max_cost_value_if_non_feasible(self, cost_p):
        if cost_p == np.inf:
            return 2.5
        elif cost_p == 1:
            return 2 * self.M_0[1] * self.M_0[2]
        else:
            print("cost_p should be eithet 0 or 1")
            exit(0)
    #enddef

    def append_cost_to_accumulator(self, cost, teacher_type, p, cost_p, success_prob, R_c):
        key = "{}_p={}_cost_p={}_success_prob={}".format(teacher_type, p, cost_p, success_prob)
        key_2 = "{}_p={}_cost_p={}_R_c={}".format(teacher_type, p, cost_p, R_c)
        if key in self.accumulator:
            self.accumulator[key].append(cost)
        else:
            self.accumulator[key] = [cost]
        if key_2 in self.accumulator:
            self.accumulator[key_2].append(cost)
        else:
            self.accumulator[key_2] = [cost]
    #enddef

def write_into_file(accumulator, exp_iter, teacher_type="offline_teaching"):
    directory = 'results/{}'.format(teacher_type)
    filename = "convergence" + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef

def get_M(n_states, n_actions, R_c=-2.5, success_prob=0.9):
    unif_prob = (1 - success_prob) / (n_states - 1)
    R = np.array([[-2.5, -2.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [-0.5, -0.5]])
    R[0,:] = R_c
    P_0 = np.zeros((n_states, n_states, n_actions))
    P_0[:, :, 0] = np.array(
        [[success_prob, unif_prob, unif_prob, unif_prob],
         [success_prob, unif_prob, unif_prob, unif_prob],
         [unif_prob, success_prob, unif_prob, unif_prob],
         [unif_prob, unif_prob, success_prob, unif_prob]]
    )
    P_0[:, :, 1] = np.array(
        [[unif_prob, success_prob, unif_prob, unif_prob],
         [unif_prob, unif_prob, success_prob, unif_prob],
         [unif_prob, unif_prob, unif_prob, success_prob],
         [unif_prob, unif_prob, unif_prob, success_prob]]
    )

    M_0 = (n_states, n_actions, R, P_0)

    return M_0
#enddef

def accumulator_function(tmp_dict, dict_accumulator):
    for key in tmp_dict:
        if key in dict_accumulator:
            dict_accumulator[key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef



if __name__ == "__main__":

    dict_accumulator = {}
    number_of_iterations = 10

    M_0 = get_M(n_states=4, n_actions=2, R_c=-2.5, success_prob=0.9)
    target_pi = np.ones(M_0[0], dtype="int")

    params = {
        "n_states":4,
        "n_action":2,
        "target_pi": target_pi,
        "epsilon": 0.1,
        "epsilon_p": 0.0001,
        "cost_p": np.inf
    }
    teachers_to_run = [("non_target_attack_on_reward", 1, params), ("non_target_attack_on_reward", 2, params),
                       ("non_target_attack_on_reward", np.inf, params), ("general_attack_on_reward", 1, params),
                       ("general_attack_on_reward", 2, params), ("general_attack_on_reward", np.inf, params),
                       ("non_target_attack_on_dynamics", 1, params), ("non_target_attack_on_dynamics", 2, params),
                       ("non_target_attack_on_dynamics", np.inf, params), ("general_attack_on_dynamics", 1, params),
                       ("general_attack_on_dynamics", 2, params), ("general_attack_on_dynamics", np.inf, params)]


    settings_to_run_init_1 = []
    p = 0.9
    for c in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        settings_to_run_init_1.append((c, p))

    settings_to_run_init = settings_to_run_init_1

    for iter_num in range(0, number_of_iterations):
        teaaching_obj = teaaching(M_0, settings_to_run_init, teachers_to_run, target_pi)
        acc_dict = teaaching_obj.offline_attack()
        dict_accumulator = accumulator_function(acc_dict, dict_accumulator)

    dict_accumulator = calculate_average(dict_accumulator, number_of_iterations)
    plot_chain.plot_offline_teaching_vary_c(dict_to_plot=dict_accumulator, plot_every_after_n_points=1, show_plots=False)