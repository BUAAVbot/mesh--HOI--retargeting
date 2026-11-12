"""Actual-contact minimax allocation; geometry and dynamics are read-only."""
import numpy as np
from scipy.optimize import linprog
from fourth_contact_wrench import contact_columns, wrench_lp as original_wrench_lp


def wrench_lp(columns, normal_rows, gravity_wrench, normal_cap_N=None,
              motor_torque_matrix=None, motor_gravity_bias_Nm=None, motor_cap_Nm=.4):
    out = original_wrench_lp(columns, normal_rows, gravity_wrench, normal_cap_N,
                            motor_torque_matrix, motor_gravity_bias_Nm, motor_cap_Nm)
    if not out.get('feasible'):
        return out
    A = np.asarray(columns, float).T
    N = np.asarray(normal_rows, float).T
    K = A.shape[1]
    rows = [np.r_[row, -1.] for row in N]
    limits = [0.] * len(N)
    if normal_cap_N is not None:
        rows.extend(np.column_stack([N, np.zeros(len(N))]))
        limits.extend(np.broadcast_to(normal_cap_N, (len(N),)))
    # Contacted digits share a small genuine preload; absent digits are not
    # fabricated as supporting contacts.
    for row in N:
        if np.any(row):
            rows.append(np.r_[-row, 0.])
            limits.append(-.05)
    if motor_torque_matrix is not None:
        M = np.asarray(motor_torque_matrix, float)
        bias = np.asarray(motor_gravity_bias_Nm, float)
        caps = np.broadcast_to(motor_cap_Nm, bias.shape)
        rows.extend(np.column_stack([M, np.zeros(len(M))]))
        limits.extend(bias+caps)
        rows.extend(np.column_stack([-M, np.zeros(len(M))]))
        limits.extend(caps-bias)
    scale = np.array([1.,1.,1.,10.,10.,10.])
    fit = linprog(np.r_[np.ones(K)*1e-7, 1.],
                  A_ub=np.asarray(rows), b_ub=np.asarray(limits),
                  A_eq=np.column_stack([scale[:,None]*A, np.zeros(6)]),
                  b_eq=-scale*np.asarray(gravity_wrench), bounds=(0,None), method='highs')
    if fit.success:
        w = fit.x[:-1]
        out['allocation'] = 'Minimax actual-digit normal force; genuine contacted-digit preload >=.05N'
        out['normal_per_digit_N'] = (N@w).tolist()
        out['equilibrium_max_error'] = float(np.max(abs(A@w+gravity_wrench)))
        out['minimum_maximum_digit_normal_N'] = float(np.max(N@w))
        if motor_torque_matrix is not None:
            out['required_motor_torque_Nm'] = (bias-M@w).tolist()
    else:
        out['allocation'] = 'Original feasible LP retained; contacted-digit preload minimax infeasible'
    return out
