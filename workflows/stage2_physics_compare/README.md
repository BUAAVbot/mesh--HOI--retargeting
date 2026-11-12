# Vita Stage-2 same-track comparison

This experiment compares official SPIDER, SBTO and PRIME+Crocoddyl algorithmic
cores on exactly the same ③ `gallery08_p01` Vita28 reference, calibrated scene,
PD contract and no-reset physical replay.  PRIME+Aligator is not an official
PRIME path and is reported separately if implemented.

The canonical reference is prepared once by `prepare_reference.py`.  Every
backend must emit a normalized archive containing `qpos`, `qvel`, and `ctrl`.
PRIME must additionally emit `tau`, `contact_lambda`, `dynamics_residual`, and
`friction_residual`.  A backend's internal objective is not an acceptance test.

Results are not called successful until two fresh, single-environment,
no-reset/no-root-write replays pass the frozen Gallery08 gates and a video is
rendered from the physical record.

## Gallery08 failure window, 2026-11-05

`audit_dynamics_feasibility.py` now uses `M*a + bias - passive`, fits contact
forces only once, and uses the native multi-point foot contact manifold with
eight-ray friction cones. Declared support without an actual manifold cannot
provide force. This restricted framewise screen is not a general infeasibility
proof; it excludes hand/seat contacts and impact impulses. Run the five force
regressions with `python test_dynamics_feasibility.py`.

The 0–80 source-frame window (1.6 seconds) was tested with official SBTO and
DSMS cores. `run_gallery08_window.py` freezes the shared model, initial state
and reference and performs two independent continuous replays per result.
`refine_sbto_window.py` strengthens terminal tracking and refines the saved
controls; robot self-contact is included in the contact-aware penetration cost.

The final SBTO candidate passes all 15 frozen gates and the additional 5 mm
self-penetration check: root RMSE 0.05832 m, final root error 0.07661 m,
scene/self penetration 3.418/4.296 mm, vertical foot support 100%, peak effort
54.96% of the actuator cap. The two replay records have identical hashes.
`export_dynamic_window.py` exports the 200 Hz states, motor torques and measured
contact wrenches and checks exact agreement with the accepted replay.

DSMS was tested with reference and physical seeds, 20 ms and 5 ms nodes, and a
0.4 second diagnostic window. It did not converge within 150 iterations. The
5 ms-node trial passed the objective directional-derivative check and started
with a defect of 2.99e-8, but ended at 17.36; it is explicitly rejected. DSMS
uses the upstream solver/tracking objective with a binary-model loader patch.
MuJoCo 3.5.0 was retained for the frozen MJB; upstream currently requests >=3.8,
so this is a Vita adaptation experiment, not an exact environment reproduction.

Evidence: `outputs/gallery08_dynamics_window_20261105/` in the workspace;
server: `/data/yihang.bai/experiments/gallery08_dynamics_window_20261105/`.
The accepted motion is executed through the saved PD targets and frozen gains.
Full 5.76 second motion, perturbation robustness and hardware remain untested.

## N21 latest reference and rebuilt stairs, 2026-11-05

`prepare_n21_dynamics.py` freezes the latest `smooth_targets_strong_r3/alpha_1`
N21 reference (SHA256 `19d71482c52c3717d33bdeb69e95c4568937a0710d8c7cd1f0ac93cd5367719b`)
and replaces the old static collision with the 200-part stair candidate. Robot
kinematics, mass and inertia are checked against the previous model. The full
209-frame, 30 Hz clock is preserved, with 10 physics steps per source interval.
All comparisons use the same initial pose, zero velocity, PD gains, collision
and 18 frozen gates. Candidate stair dimensions remain uncalibrated.
Replay the frozen `compiled_model.mjb`; `scene_model.xml` is an intermediate
motor model before the shared position-PD parameters are applied.

`repair_n21_contacts.py` provides a contact SQP preprocessing baseline; its output
is not a dynamics certificate. `run_n21_method.py` executes a recorded method
command and checks two continuous no-reset replays against the unchanged raw
target. `run_n21_seeded.py` compares projected targets with PD and retries SBTO
and DSMS with recorded seeds. `collect_n21_compare.py` verifies the replay
hashes and input hashes, then builds the numeric report, plot and synchronized
video from physical states. Recorded commands and evidence are stored in
`outputs/n21_dynamics_compare_20261105/` and the same-named server experiment.

The audit, clock, native support-force sign, self-contact, conservative support
pruning and contact-projection checks comprise 13 tests. No robustness or
hardware claim is made from this single-clip adaptation comparison.
