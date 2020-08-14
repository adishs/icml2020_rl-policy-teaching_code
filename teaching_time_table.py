import numpy as np
import copy
import sys
import time
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
        dict = {}
        for setting in self.settings_to_run:
            for tchr in self.teachers_to_run:
                R_c, success_prob = setting[0], setting[1]
                M_in = get_M(self.M_0[0], self.M_0[1], R_c, success_prob)
                env_in = environment.Environment(M_in)
                target_pi = tchr[2]["target_pi"]
                p = tchr[1]
                epsilon = tchr[2]["epsilon"]
                epsilon_p = tchr[2]["epsilon_p"]
                teacher_type = tchr[0]
                teacher_obj = teacher.teacher(env=env_in, target_pi=target_pi, p=p, epsilon=epsilon, epsilon_p=epsilon_p, teacher_type=teacher_type, pool=None) #Pool here

                # print("==================================================")
                try:
                    ##########
                    time_start = time.time()
                    ##########
                    # print(time_start)
                    if "general_attack_on_dynamics" == tchr[0]:
                        pool = teacher.generate_pool(M_in[0], M_in[1], M_in[2], M_in[3], self.target_pi)
                    else:
                        pool = None
                    teacher_obj.pool = pool
                    # print("time after pool=", time.time())
                    M_out, feasible = teacher_obj.get_target_M(M_in)
                    #######
                    end_time = time.time()-time_start
                    #######
                    if feasible:
                        dict["time_R_c={}_Teacher_type={}_P={}".format(R_c, teacher_type, p)] = end_time
                    else:
                        dict["time_R_c={}_Teacher_type={}_P={}".format(R_c, teacher_type, p)] = "NF"
                    # print(end_time)
                    # print("R_c={}, Teacher_type={} --- Runtime = {}".format(R_c, teacher_type, end_time))
                except Exception as e:
                    print(e)
                    print("time_R_c={}_Teacher_type={}_P={}".format(R_c, teacher_type, p))
                    pass
        return dict
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
        key = "time_{}_p={}_cost_p={}_success_prob={}".format(teacher_type, p, cost_p, success_prob)
        key_2 = "time_{}_p={}_cost_p={}_R_c={}".format(teacher_type, p, cost_p, R_c)
        if key in self.accumulator:
            self.accumulator[key].append(cost)
        else:
            self.accumulator[key] = [cost]
        if key_2 in self.accumulator:
            self.accumulator[key_2].append(cost)
        else:
            self.accumulator[key_2] = [cost]
    #enddef

def write_into_file(accumulator, exp_iter, teacher_type="offline_teaching_vary_c"):
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

def get_M_old(n_states, n_actions, R_c=-2.5, success_prob=0.9):
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
def get_M(n_states, n_actions, R_c=-2.5, success_prob=0.9):
    unif_prob = (1 - success_prob) / (n_states - 1)
    R = np.ones((n_states, n_actions)) * 0.5
    R[0,:] = R_c
    R[-1,:] = -0.5
    P_0 = environment.get_transition_matrix_line(n_states, success_prob)

    M_0 = (n_states, n_actions, R, P_0)

    return M_0
#enddef

def accumulator_function(tmp_dict, dict_accumulator, n_states):
    for key in tmp_dict:
        accumulator_key = (key+"_{}_nstates".format(n_states))
        if accumulator_key in dict_accumulator:
            dict_accumulator[accumulator_key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[accumulator_key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef



if __name__ == "__main__":

    dict_accumulator = {}
    number_of_iterations = 1
    c_value = -2.5

    # iter = 0

    for iter in range(1, number_of_iterations+1):

        n_states_array = [4, 10, 50, 100]
        for n_states in n_states_array:
            M_0 = get_M(n_states=n_states, n_actions=2, R_c=-2.5, success_prob=0.9)

            target_pi = np.ones(M_0[0], dtype="int")

            params = {
                "target_pi": target_pi,
                "epsilon": 0.001,
                "epsilon_p": 0.0001,
                "cost_p": np.inf
            }
            teachers_to_run = [("non_target_attack_on_reward", 1, params), ("non_target_attack_on_reward", 2, params),
                               ("non_target_attack_on_reward", np.inf, params), ("general_attack_on_reward", 1, params),
                               ("general_attack_on_reward", 2, params), ("general_attack_on_reward", np.inf, params),
                               ("non_target_attack_on_dynamics", 1, params), ("non_target_attack_on_dynamics", 2, params),
                               ("non_target_attack_on_dynamics", np.inf, params), ("general_attack_on_dynamics", 1, params),
                               ("general_attack_on_dynamics", 2, params), ("general_attack_on_dynamics", np.inf, params)]

            # teachers_to_run = [("non_target_attack_on_reward", 1, params), ("non_target_attack_on_reward", 2, params),
            #                    ("non_target_attack_on_reward", np.inf, params), ("general_attack_on_reward", 1, params),
            #                    ("general_attack_on_reward", 2, params), ("general_attack_on_reward", np.inf, params),
            #                    ("non_target_attack_on_dynamics", 1, params), ("non_target_attack_on_dynamics", 2, params),
            #                    ("non_target_attack_on_dynamics", np.inf, params)]

            settings_to_run_init_1 = []
            p = 0.9
            for c in [c_value]:
                settings_to_run_init_1.append((c, p))

            settings_to_run_init = settings_to_run_init_1

            for iter_num in range(0, number_of_iterations):
                teaaching_obj = teaaching(M_0, settings_to_run_init, teachers_to_run, target_pi)
                acc_dict = teaaching_obj.offline_attack()

            for key in acc_dict:
                acc_dict[key] = [acc_dict[key]]

            dict_accumulator = accumulator_function(acc_dict, dict_accumulator, n_states)

    dict_accumulator = calculate_average(dict_accumulator, number_of_iterations)
    # write_into_file(dict_accumulator, iter)
    plot_chain.plot_table(dict_accumulator)