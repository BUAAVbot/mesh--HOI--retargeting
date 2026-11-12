"""Same physical LP constraints, with minimax per-digit load allocation."""
import numpy as np
from scipy.optimize import linprog
from fourth_contact_wrench import contact_columns, wrench_lp as original_wrench_lp


def wrench_lp(columns, normal_rows, gravity_wrench, normal_cap_N=None,
              motor_torque_matrix=None, motor_gravity_bias_Nm=None, motor_cap_Nm=.4):
    result = original_wrench_lp(columns, normal_rows, gravity_wrench, normal_cap_N,
                               motor_torque_matrix, motor_gravity_bias_Nm, motor_cap_Nm)
    if not columns or not result['feasible']:
        return result
    A = np.asarray(columns, float).T
    N = np.asarray(normal_rows, float).T
    count = A.shape[1]
    rows = [np.r_[row, -1.] for row in N]
    limits = [0.]*len(rows)
    if normal_cap_N is not None:
        rows.extend(np.column_stack([N, np.zeros(N.shape[0])]))
        limits.extend(np.broadcast_to(normal_cap_N, (N.shape[0],)))
    if motor_torque_matrix is not None:
        M = np.asarray(motor_torque_matrix, float)
        bias = np.asarray(motor_gravity_bias_Nm, float)
        cap = np.broadcast_to(motor_cap_Nm, bias.shape)
        rows.extend(np.column_stack([M, np.zeros(M.shape[0])]))
        limits.extend(bias+cap)
        rows.extend(np.column_stack([-M, np.zeros(M.shape[0])]))
        limits.extend(cap-bias)
    scale = np.array([1., 1., 1., 10., 10., 10.])
    objective = np.r_[.00001*N.sum(0), 1.]
    optimum = linprog(objective, A_ub=np.asarray(rows), b_ub=np.asarray(limits),
                      A_eq=np.column_stack([scale[:, None]*A, np.zeros(6)]),
                      b_eq=-scale*np.asarray(gravity_wrench), bounds=(0, None), method='highs')
    if not optimum.success:
        result['balanced_solver_failed'] = True
        return result
    weights = optimum.x[:count]
    result['normal_per_digit_N'] = (N@weights).tolist()
    result['equilibrium_max_error'] = float(np.max(abs(A@weights+gravity_wrench)))
    result['normal_allocation'] = 'minimax_per_digit_with_all_original_caps'
    result['ray_weights'] = weights.tolist()
    if motor_torque_matrix is not None:
        result['required_motor_torque_Nm'] = (bias-M@weights).tolist()
    return result
