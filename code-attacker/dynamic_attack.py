import math
from utils import *
import cvxpy as cp

def get_cost_func(M_0, pi_t, epsilon, epsilon_p, p):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    P_0 = M_0[3]

    def cost_func(target_trans):
        P_0_prime = np.copy(P_0)
        for s1 in range(states_count):
            for s2 in range(states_count):
                P_0_prime[s1, s2, pi_t[s1]] = target_trans[s1, s2]
        M_0_prime = (M_0[0], M_0[1], M_0[2], P_0_prime)
        M, feasibilty = dynamic_attack_nontargetonly(M_0_prime, pi_t, epsilon, epsilon_p)
        if feasibilty:
            return calc_diff(M[3], P_0, p)
        else:
            return math.inf

    return cost_func
#enddef

def find_candid_dist(d_0, sgood, alpha):
    states_count = d_0.shape[0]
    d = cp.Variable(states_count)
    constraints = []
    constraints.append(d >= 0)
    constraints.append(d @ np.ones(states_count) == 1)
    constraints.append(cp.norm(d - d_0, 1) <= alpha)

    obj = cp.Minimize(-d[sgood])

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return d.value
#enddef

def dynamic_attack_nontargetonly(M_0, pi_t, epsilon, epsilon_p):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    P_0 = M_0[3]

    V = calc_V_values(M_0, pi_t)  # V^\{pi_t}_0
    rho = calc_rho(M_0, pi_t)  # \rho^{\pi_t}_0
    T = calc_reachtimes(M_0, pi_t)  # T^{pi_t}(s, s')
    B = np.array([V[s] - R_0[s, pi_t[s]] + rho for s in range(states_count)])  # B^{\pi_t}_0
    U = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            U[s1, s2] = V[s2] + (epsilon * T[s2, s1] if s2 != s1 else 0)

    feasible = True
    P = np.copy(P_0)
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                # P(s, a, .)
                d = cp.Variable(states_count)
                constraints = []
                constraints.append(d >= epsilon_p * P_0[s, :, a])
                constraints.append(R_0[s, pi_t[s]] + B[s] - R_0[s, a] - epsilon >= d @ U[s, :])
                # being distribution
                constraints.append(d >= 0)
                constraints.append(d @ np.ones(states_count) == 1)

                obj = cp.Minimize(cp.norm(d - P_0[s, :, a], 1))
                prob = cp.Problem(obj, constraints)
                prob.solve(solver=cp.ECOS, max_iters=5000)

                if d.value is None:
                    feasible = False

                P[s, :, a] = d.value

    return (M_0[0], M_0[1], M_0[2], P), feasible
#enddef

if __name__ == "__main__":
    pass
