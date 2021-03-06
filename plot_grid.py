import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
import os

################ plot settings #################################
mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 30})
mpl.rc('legend', **{'fontsize': 22})
mpl.rc('text', usetex=True)
# fig_size = (5.5 / 2.54, 4 / 2.54)
fig_size = [6.5, 4.8]

file_path_out = 'plots/env_grid/'



def plot_online_teaching(dict_to_plot, plot_every_after_n_points=5000, show_plots=False):
    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    # figure 1

    plt.figure(1, figsize=fig_size)

    plt.figure(1, figsize=fig_size)

    plt.plot(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['accumulated_cost_array_non_target_attack_on_reward_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"NT-RAttack", color='g', marker="o", ls="--")

    plt.plot(np.arange(len(dict_to_plot["accumulated_cost_array_no_attack_general_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['accumulated_cost_array_no_attack_general_attack_on_reward_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"None", color='#00ffff', marker="s")

    plt.plot(
        np.arange(len(dict_to_plot["accumulated_cost_array_general_attack_on_reward_p=inf"][plot_every_after_n_points:]))[::plot_every_after_n_points],
        dict_to_plot['accumulated_cost_array_general_attack_on_reward_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
        label=r"RAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(
        np.arange(len(dict_to_plot["accumulated_cost_array_general_attack_on_reward_p=1"][plot_every_after_n_points:]))[::plot_every_after_n_points],
        dict_to_plot['accumulated_cost_array_general_attack_on_reward_p=1'][plot_every_after_n_points:][::plot_every_after_n_points],
        label=r"RAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Average attack cost")
    plt.xlabel(r'Time t (x$10^4$)')
    plt.yticks([0, 0.5, 1.0])
    plt.ylim(ymax=1.01)
    plt.xticks(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
               ::plot_every_after_n_points],
               ["2", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "50"])
    plt.savefig(file_path_out + "online_reward_attack-cost" + '.pdf', bbox_inches='tight')

    plt.figure(2, figsize=fig_size)
    small_number = 1

    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_non_target_attack_on_reward_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"NT-RAttack", color='g', marker="o", ls='--')
    plt.plot(np.arange(
        len(dict_to_plot["expected_action_diff_array_no_attack_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_no_attack_non_target_attack_on_reward_p=inf'][plot_every_after_n_points:][
             ::plot_every_after_n_points],
             label=r"None", color='#00ffff', marker="s")

    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_general_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_general_attack_on_reward_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"RAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_general_attack_on_reward_p=1"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_general_attack_on_reward_p=1'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"RAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r'Average mismatch')
    plt.xlabel(r'Time t (x$10^4$)')
    plt.yticks([0, 0.5, 1])
    plt.ylim(ymax=1.01)
    plt.xticks(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
               ::plot_every_after_n_points],
               ["2", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "50"])
    plt.savefig(file_path_out + "online_reward_mismatch" + '.pdf', bbox_inches='tight')

    plt.figure(3, figsize=fig_size)

    plt.plot(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_dynamics_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['accumulated_cost_array_non_target_attack_on_dynamics_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"NT-DAttack", color='g', marker="o", ls="--")

    plt.plot(np.arange(len(dict_to_plot["accumulated_cost_array_no_attack_general_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['accumulated_cost_array_no_attack_general_attack_on_dynamics_p=inf'][plot_every_after_n_points:][
             ::plot_every_after_n_points],
             label=r"None", color='#00ffff', marker="s")

    plt.plot(np.arange(len(dict_to_plot["accumulated_cost_array_general_attack_on_dynamics_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['accumulated_cost_array_general_attack_on_dynamics_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"DAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(
        np.arange(len(dict_to_plot["accumulated_cost_array_general_attack_on_dynamics_p=1"][plot_every_after_n_points:]))[::plot_every_after_n_points],
        dict_to_plot['accumulated_cost_array_general_attack_on_dynamics_p=1'][plot_every_after_n_points:][::plot_every_after_n_points],
        label=r"DAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Average attack cost")
    plt.xlabel(r'Time t (x$10^4$)')
    plt.yticks([0, 0.1, 0.2])
    plt.ylim(ymax=0.201)
    plt.xticks(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
               ::plot_every_after_n_points],
               ["2", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "50"])
    plt.savefig(file_path_out + "online_dynamics_attack-cost" + '.pdf', bbox_inches='tight')

    plt.figure(4, figsize=fig_size)
    small_number = 1
    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_non_target_attack_on_dynamics_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_non_target_attack_on_dynamics_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"NT-DAttack", color='g', marker="o", ls="--")

    plt.plot(np.arange(
        len(dict_to_plot["expected_action_diff_array_no_attack_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_no_attack_non_target_attack_on_reward_p=inf'][plot_every_after_n_points:][
             ::plot_every_after_n_points],
             label=r"None", color='#00ffff', marker="s")

    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_general_attack_on_dynamics_p=inf"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_general_attack_on_dynamics_p=inf'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"DAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")

    plt.plot(np.arange(len(dict_to_plot["expected_action_diff_array_general_attack_on_dynamics_p=1"][plot_every_after_n_points:]))[
             ::plot_every_after_n_points],
             dict_to_plot['expected_action_diff_array_general_attack_on_dynamics_p=1'][plot_every_after_n_points:][::plot_every_after_n_points],
             label=r"DAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r'Average mismatch')
    plt.xlabel(r'Time t (x$10^4$)')
    plt.yticks([0, 0.5, 1])
    plt.ylim(ymax=1.01)
    plt.xticks(np.arange(len(dict_to_plot["accumulated_cost_array_non_target_attack_on_reward_p=inf"][plot_every_after_n_points:]))[
               ::plot_every_after_n_points],
               ["2", "", "", "", "10", "", "", "", "", "20", "", "", "", "", "30",
                "", "", "", "", "40", "", "", "", "50"])
    plt.savefig(file_path_out + "online_dynamics_mismatch" + '.pdf', bbox_inches='tight')
    if show_plots:
        plt.show()
#enddef

def plot_offline_teaching_vary_c(dict_to_plot, plot_every_after_n_points=1, show_plots=False):

    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    ### Figure 1
    plt.figure(1, figsize=fig_size)

    plt.plot(np.arange(len(dict_to_plot["general_attack_on_reward_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_reward_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"RAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_reward_p=2_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_reward_p=2_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"RAttack $(\ell_2)$", color='r', marker="s", ls=":")
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_reward_p=inf_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_reward_p=inf_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"RAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_reward_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['non_target_attack_on_reward_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"NT-RAttack", color='g', ls="--", marker="o")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Attack cost $(\ell_\infty)$")
    plt.xlabel(r'Reward for $s_0$ state ')
    plt.xticks(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
               np.arange(-5, 5.01, 1, dtype="int")[::plot_every_after_n_points])
    plt.ylim(ymax=10.1)
    plt.yticks(np.arange(0, 10.5, 2), ["0.0", "2.0", "4.0", "6.0", "8.0", "10"])
    print(plt.rcParams.get('figure.figsize'))
    # exit(0)
    plt.savefig(file_path_out + "offline_reward_vary-c" + '.pdf', bbox_inches='tight')

    ### Figure 2

    infeasible_scalar = 2.11
    plt.figure(2, figsize=fig_size)
    small_number = 1

    plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"DAttack $(\ell_1)$",
             color='#F08080', marker="*", ls=":")
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=2_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_dynamics_p=2_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"DAttack $(\ell_2)$",
             color='r', marker="s", ls=":")
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=inf_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             dict_to_plot['general_attack_on_dynamics_p=inf_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
             label=r"DAttack $(\ell_\infty)$",
             color='#8B0000', marker="^", ls=":")

    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             ([infeasible_scalar - 0.025] * len(
                 dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             color='gray', lw=3)

    plt.annotate('Infeasible', xy=(4.1, 2.1), xycoords='data',
                 xytext=(0.328, 0.87), textcoords='axes fraction',
                 arrowprops=dict(facecolor='gray', shrink=0.0001),
                 horizontalalignment='left', verticalalignment='top', size="20"
                 )

    array_of_dynamics = dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points][
             :feasible_upper_index], np.minimum(
        dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points][
        :feasible_upper_index], infeasible_scalar), label=r"NT-DAttack",
             color='g', ls="--", marker="o")
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
             np.minimum(dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9'][::plot_every_after_n_points],
                        infeasible_scalar),
             color='g', ls="--")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r'Attack cost $(\ell_\infty)$')
    plt.xlabel(r'Reward for $s_0$ state')
    plt.xticks(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
               np.arange(-5, 5.01, 1, dtype="int")[::plot_every_after_n_points])
    plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.ylim(ymax=2.2)
    plt.xticks(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_success_prob=0.9"]))[::plot_every_after_n_points],
               np.arange(-5, 5.01, 1, dtype="int")[::plot_every_after_n_points])

    plt.savefig(file_path_out + "offline_dynamics_vary-c" + '.pdf', bbox_inches='tight')

    if show_plots:
        plt.show()

#enddef

def plot_offline_teaching_vary_eps( dict_to_plot, plot_every_after_n_points=1, show_plots=False):

    if not os.path.isdir(file_path_out):
        os.makedirs(file_path_out)

    ### Figure 1
    plt.figure(1, figsize=fig_size)

    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_reward_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        dict_to_plot['general_attack_on_reward_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
        label=r"RAttack $(\ell_1)$", color='#F08080', marker="*", ls=":")
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_reward_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        dict_to_plot['general_attack_on_reward_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
        label=r"RAttack $(\ell_2)$", color='r', marker="s", ls=":")
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_reward_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        dict_to_plot['general_attack_on_reward_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
        label=r"RAttack $(\ell_\infty)$", color='#8B0000', marker="^", ls=":")
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_reward_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points],
             dict_to_plot['non_target_attack_on_reward_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
             label=r"NT-RAttack", color='g', ls="--", marker="o")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel("Attack cost $(\ell_\infty)$")
    plt.xlabel(r'$\epsilon$ margin')
    plt.yticks(np.arange(0, 10.5, 2), ["0.0", "2.0", "4.0", "6.0", "8.0", "10"])
    plt.xticks(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        ["0", "", "", "", "0.2", "", "", "", "0.4", "", "", "", "0.6", "", "", "", "0.8", "", "", "", "1.0"])
    plt.ylim(ymax=10.1)
    # exit(0)
    plt.savefig(file_path_out + "offline_reward_vary-eps" + '.pdf', bbox_inches='tight')

    ### Figure 2
    infeasible_scalar = 2.11
    plt.figure(2, figsize=fig_size)
    array_of_dynamics = dict_to_plot['general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points][
        :feasible_upper_index], np.minimum(
            dict_to_plot['general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points][
            :feasible_upper_index], infeasible_scalar),
        label=r"DAttack $(\ell_1)$",
        color='#F08080', marker="*", ls=":")
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        np.minimum(dict_to_plot['general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
                   infeasible_scalar),
        color='#F08080', ls=":")

    # plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],np.minimum(dict_to_plot['general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points], infeasible_scalar),
    #          label=r"DAttack $(\ell_1)$",
    #                          color='#F08080', marker="*", ls=":")

    array_of_dynamics = dict_to_plot['general_attack_on_dynamics_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points][
        :feasible_upper_index], np.minimum(
            dict_to_plot['general_attack_on_dynamics_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points][
            :feasible_upper_index], infeasible_scalar),
        label=r"DAttack $(\ell_2)$",
        color='r', marker="s", ls=":")
    plt.plot(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        np.minimum(dict_to_plot['general_attack_on_dynamics_p=2_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
                   infeasible_scalar),
        color='r', ls=":")

    array_of_dynamics = dict_to_plot['general_attack_on_dynamics_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9'][
                        ::plot_every_after_n_points]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points][:feasible_upper_index], np.minimum(
        dict_to_plot['general_attack_on_dynamics_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points][
        :feasible_upper_index], infeasible_scalar),
             label=r"DAttack $(\ell_\infty)$",
             color='#8B0000', marker="^", ls=":")
    plt.plot(np.arange(len(dict_to_plot["general_attack_on_dynamics_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points], np.minimum(
        dict_to_plot['general_attack_on_dynamics_p=inf_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
        infeasible_scalar),
             color='#8B0000', ls=":")

    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points], ([infeasible_scalar - 0.025] * len(
        dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
             color='gray', lw=3)

    plt.annotate('Infeasible', xy=(4.1, 2.1), xycoords='data',
                 xytext=(0.186, 0.87), textcoords='axes fraction',
                 arrowprops=dict(facecolor='gray', shrink=0.0001),
                 horizontalalignment='left', verticalalignment='top', size="20"
                 )
    x_coord = feasible_upper_index
    x_coord_frac = x_coord / len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"][
             ::plot_every_after_n_points])

    plt.annotate('Infeasible', xy=(x_coord, 2.1), xycoords='data',
                 xytext=(x_coord_frac, 0.87), textcoords='axes fraction',
                 arrowprops=dict(facecolor='gray', shrink=0.0001),
                 horizontalalignment='left', verticalalignment='top', size="20"
                 )

    array_of_dynamics = dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][
                        ::plot_every_after_n_points]
    feasible_upper_index = np.argwhere(array_of_dynamics > infeasible_scalar)[0][0]
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points][:feasible_upper_index], np.minimum(
        dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points][
        :feasible_upper_index], infeasible_scalar), label=r"NT-DAttack",
             color='g', ls="--", marker="o")
    plt.plot(np.arange(len(dict_to_plot["non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[
             ::plot_every_after_n_points], np.minimum(
        dict_to_plot['non_target_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9'][::plot_every_after_n_points],
        infeasible_scalar),
             color='g', ls="--")

    plt.plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r'Attack cost $(\ell_\infty)$')
    plt.xlabel(r'$\epsilon$ margin')
    plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.ylim(ymax=2.2)
    plt.xticks(
        np.arange(len(dict_to_plot["general_attack_on_dynamics_p=1_cost_p=inf_R_c=-2.5_success_prob=0.9"]))[::plot_every_after_n_points],
        ["0", "", "", "", "0.2", "", "", "", "0.4", "", "", "", "0.6", "", "", "", "0.8", "", "", "", "1.0"])

    plt.savefig(file_path_out + "offline_dynamics_vary-eps" + '.pdf', bbox_inches='tight')

    if show_plots:
        plt.show()
#enddef
