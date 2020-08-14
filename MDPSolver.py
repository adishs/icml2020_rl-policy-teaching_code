import numpy as np
import copy
import numpy.matlib
import random
import time
from scipy import sparse
import matplotlib.pyplot as plt
import time
np.set_printoptions(suppress=True);
np.set_printoptions(precision=12);
# np.set_printoptions(threshold=np.nan);
np.set_printoptions(linewidth=500)

def get_stationary_dist(env, policy):
    start = time.time()
    n_states = env.n_states
    P_pi = get_P_pi(env, policy)
    A = np.zeros((n_states, n_states))
    A[0: n_states - 1, :] = np.transpose(P_pi - np.identity(n_states))[1:, :]
    A[-1, :] = np.ones(n_states)
    b = np.zeros(n_states)
    b[-1] = 1
    mu = np.linalg.solve(A, b)
    return mu
#enddef

def get_P_pi_sparse_transpose(env, policy):
    if policy.ndim == 1:
        policy = convert_det_to_stochastic_policy(env, policy)
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    P_pi = np.transpose(P_pi)
    P_pi_sparse = sparse.csr_matrix(P_pi)
    return P_pi_sparse, P_pi
# enddef

def get_P_pi_sparse(env, policy):
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    P_pi_sparse = sparse.csr_matrix(P_pi)
    return P_pi_sparse, P_pi
# enddef

def span_norm(v):
    return np.max(v) - np.min(v)
#enddef

def averaged_valueIteration(env, reward, tol=1e-5):
    V = np.zeros((env.n_states))
    Q = np.zeros((env.n_states, env.n_actions))
    iter = 0
    while True:
        iter +=1
        V_old = copy.deepcopy(V)

        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] + env.T_sparse_list[a].dot(V)
        V = np.max(Q, axis=1)
        if span_norm(V - V_old) < tol:
            break
    #endwhile

    # For a deterministic policy
    pi_d = np.argmax(Q, axis=1)
    pi_s = Q - np.max(Q, axis=1)[:, None]
    pi_s[np.where((-1e-12 <= pi_s) & (pi_s <= 1e-12))] = 1
    pi_s[np.where(pi_s <= 0)] = 0
    pi_s = pi_s/pi_s.sum(axis=1)[:, None]
    return V, pi_d, pi_s
#enddef

def extended_averaged_value_iteration(env, r_hat, conf_r, p_hat, conf_p, tol=1e-5):
    n_states = env.n_states
    n_actions = env.n_actions
    start = time.time()
    u = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    r_tilda = r_hat + conf_r

    while True:
        u_old = copy.deepcopy(u)
        for s in range(n_states):
            for a in range(n_actions):
                p_hat_vec = p_hat[s, :, a]
                conf_p_scalar = conf_p[s, a]

                Q[s, a] = r_tilda[s, a] + compute_inner_maximum(n_states, u_old, p_hat_vec, conf_p_scalar).dot(u_old)
        u = np.max(Q, axis=1)
        # print(span_norm(u-u_old))
        if span_norm(u - u_old) < tol:
            break
    pi_d = np.argmax(Q, axis=1)
    end = time.time()
    return pi_d
#enddef

def compute_inner_maximum(n_states, u, p_hat_vector, conf_p_scalar):
    # print("inside compute_inner_maximum")
    S = numpy.argsort(u)[::-1]  # sort descending
    p = np.zeros(n_states)
    for i in range(len(S)):
        #1)
        if i == 0:
            p[S[i]] = min(1.0, p_hat_vector[S[i]] + conf_p_scalar/2)
        else:
            p[S[i]] = p_hat_vector[S[i]]
    #2)
    l = n_states-1
    #3)
    while(np.sum(p) > 1 + 1e-10):
        #(a)
        p[S[l]] = max(0, 1 - (sum(p) - p[S[l]]))
        #(b)
        l = l - 1
    return p
#enddef

def compute_averaged_reward_given_policy(env, reward, policy):
    stationary_dist = get_stationary_dist(env, policy)
    expected_reward = 0
    for s in range(env.n_states):
        expected_reward += stationary_dist[s] * reward[s, policy[s]]
    return expected_reward
#enddef

def convert_det_to_stochastic_policy(env, deterministicPolicy):
    # Given a deterministic Policy, I will return a stochastic policy
    stochasticPolicy = np.zeros((env.n_states, env.n_actions))
    if env.terminal_state == 1:
        n_states = env.n_states-1
    else:
        n_states = env.n_states
    for i in range(n_states):
        stochasticPolicy[i][deterministicPolicy[i]] = 1
    return stochasticPolicy
#enddef

def get_P_pi(env, policy):
    if policy.ndim == 1:
        policy = convert_det_to_stochastic_policy(env, policy)
    P_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            P_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    return P_pi
# enddef

def computeValueFunction_given_policy(env, reward, expected_reward, policy, tol=1e-5):
    # Given a policy (could be either deterministic or stochastic), I return the Value Function
    # Using the Bellman Equations
    # Let's check if this policy is deterministic or stochastic
    if len(policy.shape) == 1:
        changed_policy = convert_det_to_stochastic_policy(env, policy)
    else:
        changed_policy = policy

    P_pi = get_P_pi(env, changed_policy)
    # Converting this T to a sparse matrix
    P_pi_sparse = sparse.csr_matrix(P_pi)
    # Some more initialisations
    V = np.zeros(env.n_states)
    Q = np.zeros((env.n_states, env.n_actions))
    iter = 0
    # Bellman Equation
    while True:
        iter += 1
        V_old = copy.deepcopy(V)
        for a in range(env.n_actions):
            Q[:, a] = reward[:, a] - expected_reward + P_pi_sparse.dot(V)
        V = np.max(Q, axis=1)
        if span_norm(V - V_old) < tol:
            break
    # Converged. let's return the V
    return V
#enddef

def compute_B_pi(env, reward, expected_reward, policy):
    B_pi = np.zeros((env.n_states))

    V_pi = computeValueFunction_given_policy(env, reward, expected_reward, policy)

    P_pi = get_P_pi(env, policy)

    # Converting this T to a sparse matrix
    P_pi_sparse = sparse.csr_matrix(P_pi)
    B_pi[:] = P_pi_sparse.dot(V_pi)
    return B_pi
#enddef




########################################
if __name__ == "__main__":
    pass





