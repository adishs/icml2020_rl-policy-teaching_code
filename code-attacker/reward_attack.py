import numpy as np
import cvxpy as cp
from utils import *

def reward_attack_general(M_0, pi_t, epsilon, p):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    A_pi_t_mu = calc_A_mu(M_0, pi_t, calc_sd(M_0, pi_t))
    R = cp.Variable(states_count * actions_count)
    constraints = []
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                pi = neighbor(pi_t, s, a)
                mu = calc_sd(M_0, pi)
                A_mu = calc_A_mu(M_0, pi, mu)
                constraints.append(R @ A_mu <= R @ A_pi_t_mu - epsilon)

    R_0_vector = np.asarray(R_0).reshape(-1)
    obj = cp.Minimize(cp.norm(R - R_0_vector, p))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.ECOS, max_iters=5000)
    # prob.solve()
    R = R.value.reshape((M_0[0], M_0[1]))
    return (M_0[0], M_0[1], R, M_0[3])
#enddef

def reward_attack_nontargetonly(M_0, pi_t, epsilon, p):
    R_0 = M_0[2]
    R = R_0 - calc_chi(M_0, pi_t, epsilon)
    return (M_0[0], M_0[1], R, M_0[3])
#enddef


if __name__ == "__main__":
    pass



