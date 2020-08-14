import itertools
import numpy as np

def calc_sd(M, pi):
    P = M[3]
    states_count = M[0]
    T = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            T[s1, s2] = P[s1, s2, pi[s1]]
    A = np.zeros((states_count, states_count))
    A[0:states_count - 1, :] = np.transpose(T - np.identity(states_count))[1:, :]
    A[-1, :] = np.ones(states_count)

    b = np.zeros(states_count)
    b[-1] = 1

    return np.linalg.solve(A, b)
#enddef

def neighbor(pi, s, a):
    n = np.copy(pi)
    n[s] = a
    return n
#enddef

def calc_A_mu(M, pi, mu):
    states_count = M[0]
    actions_count = M[1]
    A_mu = np.zeros(states_count * actions_count)
    for s in range(states_count):
        for a in range(actions_count):
            if a == pi[s]:
                A_mu[s * actions_count + a] = mu[s]
    return A_mu
#enddef

def calc_rho(M, pi):
    R_0 = M[2]
    R_0_vector = np.asarray(R_0).reshape(-1)
    return calc_A_mu(M, pi, calc_sd(M, pi)) @ R_0_vector
#enddef

def calc_reachtimes(M, pi):
    P = M[3]
    states_count = M[0]
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    reach_times = np.zeros((states_count, states_count))
    for s2 in range(states_count):
        A = np.delete(T, s2, 1)
        A = np.delete(A, s2, 0)
        h = np.linalg.inv(np.identity(states_count - 1) - A) @ np.ones(states_count - 1)
        h = np.insert(h, s2, 0)
        reach_times[:, s2] = h

    return reach_times
#enddef

def calc_diameter(M, pi):
    return np.max(calc_reachtimes(M, pi))
#enddef

def span(v):
    return np.max(v) - np.min(v)
#enddef

def calc_chi(M_0, pi_t, epsilon):
    states_count = M_0[0]
    actions_count = M_0[1]
    R_0 = M_0[2]
    R_0_vector = np.asarray(R_0).reshape(-1)
    rho_t = calc_A_mu(M_0, pi_t, calc_sd(M_0, pi_t)) @ R_0_vector
    chi = np.zeros((states_count, actions_count))
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                pi = neighbor(pi_t, s, a)
                mu = calc_sd(M_0, pi)
                rho = calc_A_mu(M_0, pi, mu) @ R_0_vector
                chi[s, a] = max((rho - rho_t + epsilon) / mu[s], 0)
    return chi
#enddef

def calc_V_values(M, pi):
    states_count = M[0]
    actions_count = M[1]
    R = M[2]
    P = M[3]
    T = np.zeros((states_count, states_count))
    for x in range(states_count):
        for y in range(states_count):
            T[x, y] = P[x, y, pi[x]]

    R_pi = np.zeros(states_count)
    for x in range(states_count):
        R_pi[x] = R[x, [pi[x]]]

    rho = calc_rho(M, pi)

    A = np.zeros((states_count, states_count))
    A[0:states_count - 1, :] = (np.identity(states_count) - T)[1:, :]
    A[states_count - 1, 0] = 1

    b = np.zeros(states_count)
    b[0:states_count - 1] = (R_pi - rho * np.ones(states_count))[1:]
    return np.linalg.solve(A, b)
#enddef

def calc_P_s(M_0, pi_t, epsilon_p):
    states_count = M_0[0]
    actions_count = M_0[1]
    P_0 = M_0[3]
    P_s = np.copy(P_0)
    V = calc_V_values(M_0, pi_t)
    s_min = np.argmin(V)
    for s in range(states_count):
        for a in range(actions_count):
            if a != pi_t[s]:
                P_s[s, :, a] = epsilon_p * P_0[s, :, a]
                P_s[s, s_min, a] += 1 - epsilon_p
    return P_s
#enddef

def check_results(M, pi_t, epsilon, print_rhos=False):
    states_count = M[0]
    actions_count = M[1]
    rho_t = calc_rho(M, pi_t)
    all_policies = [np.array(x) for x in itertools.product(range(actions_count), repeat=states_count)]
    result = True
    for pi in all_policies:
        rho = calc_rho(M, pi)
        if print_rhos:
            if np.array_equal(pi, pi_t):
                print(pi, rho, "---> target")
            else:
                print(pi, rho)

        if rho > rho_t - epsilon + 0.001 and not np.array_equal(pi, pi_t):
            result = False
    return result
#enddef

def calc_diff(P1, P2, p):
    states_count = P1.shape[0]
    actions_count = P1.shape[2]
    l1diffs = []
    for s in range(states_count):
        for a in range(actions_count):
            l1diffs.append(np.linalg.norm(P1[s, :, a] - P2[s, :, a], ord=1))
    l1diffs = np.array(l1diffs)
    return np.linalg.norm(l1diffs, ord=p)
#enddef