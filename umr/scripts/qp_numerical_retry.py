"""Opt-in numerical retries of the SAME conic QP, with explicit feasibility checks."""
import json
import os
from pathlib import Path
import clarabel
import numpy as np
from scipy import sparse


def cone_violation(vector, cones):
    offset = 0; violation = 0.
    for cone in cones:
        dim = cone.dim
        values = vector[offset:offset+dim]; offset += dim
        if isinstance(cone, clarabel.NonnegativeConeT):
            violation = max(violation, float(-values.min(initial=0.)))
        elif isinstance(cone, clarabel.SecondOrderConeT):
            violation = max(violation, float(np.linalg.norm(values[1:])-values[0]))
        else:
            raise ValueError(f'unsupported retry cone {cone}')
    assert offset == len(vector)
    return violation


def capture_failed_qp(p, q, a, b, cones, lower, upper, nvar, initial_status):
    # Explicit diagnostic mode: capture the first failed QP and stop this
    # isolated replay AFTER the unchanged v1 retries fail. Never promotes data.
    capture = os.environ.get('UMR_QP_CAPTURE_ONLY')
    if capture:
        folder = Path(capture)
        folder.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(folder/'P.npz', p)
        sparse.save_npz(folder/'A.npz', a)
        np.savez(folder/'vectors.npz', q=q, b=b, lower=lower, upper=upper,
                 nvar=np.array(nvar))
        (folder/'metadata.json').write_text(json.dumps(dict(
            initial_status=initial_status,
            cones=[dict(type=type(c).__name__, dim=c.dim) for c in cones])))
        raise RuntimeError('Diagnostic QP capture complete; isolated replay stopped')


def retry_same_qp(p, q, a, b, cones, lower, upper, nvar, initial_status):
    # No rows are removed, no slack is added, and P/q/A/b are unchanged.
    variants = [
        ('equilibrate30', dict(equilibrate_max_iter=30)),
        ('regularization1e-10', dict(equilibrate_max_iter=30, static_regularization_constant=1e-10)),
        ('no_equilibration', dict(equilibrate_enable=False)),
        ('regularization1e-6', dict(equilibrate_max_iter=30, static_regularization_constant=1e-6)),
    ]
    if os.environ.get('UMR_QP_STEP_RETRY') == '1':
        variants.extend((f'step{step}', dict(max_step_fraction=step))
                        for step in (.95, .9, .8, .7, .5))
    symmetric_p = p+p.T-sparse.diags(p.diagonal(), format='csc')
    records = []
    for name, overrides in variants:
        settings = clarabel.DefaultSettings()
        settings.verbose = False; settings.max_threads = 1; settings.max_iter = 500
        for key, value in overrides.items():
            setattr(settings, key, value)
        result = clarabel.DefaultSolver(p, q, a, b, cones, settings).solve()
        record = dict(variant=name, status=str(result.status))
        if str(result.status) == 'Solved':
            x = np.asarray(result.x, dtype=np.float64).copy()
            z = np.asarray(result.z, dtype=np.float64)
            x[:nvar] = np.clip(x[:nvar], lower, upper)
            if np.isfinite(x).all() and np.isfinite(z).all():
                slack = b-a@x
                primal = cone_violation(slack, cones)
                dual = cone_violation(z, cones)
                px = symmetric_p@x; az = a.T@z
                stationarity = float(np.linalg.norm(px+q+az, ord=np.inf)/(1+max(
                    np.linalg.norm(px, ord=np.inf), np.linalg.norm(q, ord=np.inf),
                    np.linalg.norm(az, ord=np.inf))))
                complementarity = float(abs(slack@z)/(1+abs(.5*x@px+q@x)))
                record.update(primal=primal, dual=dual, stationarity=stationarity,
                              complementarity=complementarity)
                if primal <= 1e-7 and dual <= 1e-7 and stationarity <= 1e-7 and complementarity <= 1e-6:
                    print('[QP numerical retry] '+json.dumps(dict(initial_status=initial_status, accepted=record)), flush=True)
                    return x[:nvar]
        records.append(record)
    capture_failed_qp(p, q, a, b, cones, lower, upper, nvar, initial_status)
    raise RuntimeError('Same-QP numerical retries failed: '+json.dumps(dict(initial=initial_status, attempts=records)))
