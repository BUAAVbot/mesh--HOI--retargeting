"""Execute PHC's official kinematic-fitting loop on shared Vita52 observations.

Official PHC FK methods and optimization loop are executed from pinned source.
Only dataset/embodiment plumbing is adapted; this is not a PHC policy rollout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/data/KinoGen/dependencies/retarget_compare/phc_fitting_pinned'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--frames', type=int, default=172)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial-joint-noise', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory: ' + str(args.output))
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    import numpy as np
    import torch
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU is unavailable')
    rot_path = args.source / 'phc/phc/utils/rotation_conversions.py'
    fk_path = args.source / 'phc/phc/utils/torch_humanoid_batch.py'
    fit_path = args.source / 'phc/scripts/data_process/fit_smpl_motion.py'
    smooth_path = args.source / 'smplsim/smpl_sim/utils/smoothing_utils.py'
    spec = importlib.util.spec_from_file_location('official_phc_rotation_conversions', rot_path)
    rotation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rotation_module)
    smooth_globals = runpy.run_path(str(smooth_path))
    official_kernel = smooth_globals['gaussian_kernel_1d']
    # Original smoothing code builds a CPU kernel. This device-only adapter
    # retains its exact kernel values and filtering operation on GPU inputs.
    smooth_fn = smooth_globals['gaussian_filter_1d_batch']
    smooth_fn.__globals__['gaussian_kernel_1d'] = lambda size, sigma: official_kernel(size, sigma).to(device)

    # Loading only the official class avoids unrelated mesh/visualization and
    # Isaac training dependencies. The class method bodies are not rewritten.
    fk_ast = ast.parse(fk_path.read_text())
    fk_class = next(node for node in fk_ast.body if isinstance(node, ast.ClassDef) and node.name == 'Humanoid_Batch')
    namespace = {'torch': torch, 'np': np, 'tRot': rotation_module,
                 'EasyDict': SimpleNamespace, 'sRot': Rotation}
    exec(compile(ast.Module(body=[fk_class], type_ignores=[]), str(fk_path), 'exec'), namespace)
    OfficialHumanoid = namespace['Humanoid_Batch']
    fit_ast = ast.parse(fit_path.read_text())
    fit_func = next(node for node in fit_ast.body if isinstance(node, ast.FunctionDef) and node.name == 'process_motion')
    loop_ast = next(node for node in ast.walk(fit_func) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'iteration')
    loop_code = compile(ast.Module(body=[loop_ast], type_ignores=[]), str(fit_path), 'exec')

    model = mj.MjModel.from_xml_path(str(args.model.resolve()))
    if (model.nq, model.nv) != (59, 58):
        raise ValueError('Shared Vita52 required')
    jids = np.array([j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE])
    jids = jids[np.argsort(model.jnt_qposadr[jids])]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in jids]
    if np.max(np.abs(model.jnt_pos[jids])) > 1e-9:
        raise ValueError('This adapter requires hinge pivots at local body origins')
    kept = [b for b in range(1, model.nbody) if model.body_jntnum[b] > 0]
    if len(kept) != 53:
        raise ValueError('Expected a root and52 one-hinge moving bodies')
    for k, b in enumerate(kept[1:]):
        if int(model.body_jntadr[b]) != int(jids[k]):
            raise ValueError('Body-order and joint-order differ')
    index = {b: k for k, b in enumerate(kept)}
    offsets, local_mats, parents = [], [], []
    for b in kept:
        if b == kept[0]:
            offsets.append([0., 0., 0.]); local_mats.append(np.eye(3)); parents.append(-1)
            continue
        pos = model.body_pos[b].copy()
        mat = Rotation.from_quat(model.body_quat[b], scalar_first=True).as_matrix()
        parent = int(model.body_parentid[b])
        while parent not in index:
            if parent == 0:
                raise ValueError('Unexpected second root')
            parent_mat = Rotation.from_quat(model.body_quat[parent], scalar_first=True).as_matrix()
            pos = model.body_pos[parent] + parent_mat @ pos
            mat = parent_mat @ mat
            parent = int(model.body_parentid[parent])
        offsets.append(pos); local_mats.append(mat); parents.append(index[parent])

    fk = OfficialHumanoid.__new__(OfficialHumanoid)
    fk.cfg = SimpleNamespace(extend_config=[])
    fk.num_dof = 52
    fk.num_bodies = 53
    fk.num_bodies_augment = 53
    fk.body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in kept]
    fk.body_names_augment = list(fk.body_names)
    fk._parents = torch.tensor(parents, dtype=torch.int64)
    fk._offsets = torch.tensor(np.asarray(offsets)[None], dtype=torch.float32, device=device)
    fk._local_rotation_mat = torch.tensor(np.asarray(local_mats)[None], dtype=torch.float32, device=device)
    fk.dof_axis = torch.tensor(model.jnt_axis[jids].copy(), dtype=torch.float32, device=device)
    fk.joints_range = torch.tensor(model.jnt_range[jids].copy(), dtype=torch.float32, device=device)

    # Validate the adapted skeleton against independent MuJoCo FK before any fit.
    rng = np.random.default_rng(174114)
    testq = np.tile(model.qpos0, (8, 1))
    testq[:, :3] = rng.normal(size=(8, 3)) * .2 + [0., 0., 1.]
    test_rot = rng.normal(size=(8, 3)) * .4
    testq[:, 3:7] = Rotation.from_rotvec(test_rot).as_quat(scalar_first=True)
    testq[:, 7:] = rng.uniform(model.jnt_range[jids, 0], model.jnt_range[jids, 1], size=(8, 52))
    test_pose = torch.cat([torch.tensor(test_rot[None, :, None], dtype=torch.float32, device=device),
                          fk.dof_axis * torch.tensor(testq[None, :, 7:, None], dtype=torch.float32, device=device)], dim=2)
    with torch.no_grad():
        tested = fk.fk_batch(test_pose, torch.tensor(testq[None, :, :3], dtype=torch.float32, device=device))
        actual = tested.global_translation[0].cpu().numpy()
    md = mj.MjData(model)
    expected = []
    for q in testq:
        md.qpos[:] = q
        mj.mj_forward(model, md)
        expected.append(md.xpos[kept].copy())
    fk_error = float(np.max(np.abs(actual-np.asarray(expected))))
    if fk_error > 3e-6:
        raise ValueError('PHC adapted FK disagrees with MuJoCo: ' + str(fk_error))

    U = np.asarray(json.loads(args.support.read_text())['source_to_sim_U'], float)
    C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    times = np.asarray(np.load(args.data/'timeline.npz', allow_pickle=False)['pts_seconds'][:args.frames], float)
    times -= times[0]
    # Consistent, explicit Vita skeleton correspondences. No phantom hand data.
    matches = [('base_link', 0), ('waist_pitch_link', 9),
               ('left_hip_roll_link', 1), ('right_hip_roll_link', 2),
               ('left_knee_link', 4), ('right_knee_link', 5),
               ('left_ankle_roll_link', 7), ('right_ankle_roll_link', 8),
               ('left_shoulder_yaw_link', 16), ('right_shoulder_yaw_link', 17),
               ('left_elbow_link', 18), ('right_elbow_link', 19),
               ('left_wrist_pitch_link', 20), ('right_wrist_pitch_link', 21)]
    robot_ids = [fk.body_names.index(body) for body, _ in matches]
    human_ids = [joint for _, joint in matches]
    aux_ids = np.array([i for i, name in enumerate(names) if name.startswith(('lh_', 'rh_', 'neck_'))])
    allq, person_reports, observations = [], [], []

    class Progress:
        def __init__(self, person):
            self.person = person; self.rows = []; self.started = time.perf_counter()
        def set_description_str(self, text):
            loss = float(text.split()[-1]) / 1000.
            iteration = len(self.rows)
            self.rows.append({'iteration': iteration, 'loss': loss, 'elapsed_seconds': time.perf_counter()-self.started})
            if iteration % 100 == 0 or iteration == args.iterations-1:
                print(json.dumps({'person': self.person, **self.rows[-1]}), flush=True)

    for person in (1, 2):
        person_begun = time.perf_counter()
        directory = args.data/('human/person_%02d' % person)
        params = np.load(directory/'smplx_params_world.npz', allow_pickle=False)
        source = np.asarray(np.load(directory/'smplx_meshes_world.npz', allow_pickle=False)['joints_world'][:len(times)], float) @ U.T
        joints = torch.tensor(source, dtype=torch.float32, device=device)
        source_rotation = U @ Rotation.from_rotvec(params['global_orient'][:len(times)]).as_matrix() @ C
        heading = np.arctan2(source_rotation[:, 1, 0], source_rotation[:, 0, 0])
        root_rotation = np.c_[np.zeros(len(times)), np.zeros(len(times)), heading]
        dof_pos_new = torch.zeros((1, len(times), 52, 1), dtype=torch.float32, device=device)
        if args.initial_joint_noise:
            dof_pos_new += args.initial_joint_noise * torch.randn_like(dof_pos_new)
            dof_pos_new[:, :, aux_ids] = 0.
        dof_pos_new.requires_grad_(True)
        root_rot_new = torch.tensor(root_rotation, dtype=torch.float32, device=device, requires_grad=True)
        root_pos_offset = torch.zeros((1, 3), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=.02)
        progress = Progress(person)
        env = {'torch': torch, 'cfg': {'fitting_iterations': args.iterations},
               'humanoid_fk': fk, 'num_augment_joint': 0,
               'N': len(times), 'dof_pos_new': dof_pos_new, 'root_rot_new': root_rot_new,
               'root_pos_offset': root_pos_offset, 'root_trans_offset': joints[:, 0],
               'joints': joints, 'robot_joint_pick_idx': robot_ids,
               'smpl_joint_pick_idx': human_ids, 'optimizer': optimizer,
               'pbar': progress, 'data_key': 'person_%02d' % person,
               'gaussian_filter_1d_batch': smooth_fn, 'kernel_size': 5, 'sigma': .75,
               'device': device}
        # Exact official for-iteration AST: mean joint norm + .01*dof^2,
        # Adam(.02), range clamp, replicate-padded Gaussian(5,.75).
        exec(loop_code, env)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        with torch.no_grad():
            dof_pos_new.clamp_(fk.joints_range[:, 0, None], fk.joints_range[:, 1, None])
            dof_pos_new[:, :, aux_ids] = 0.
            root_positions = joints[:, 0] + root_pos_offset
            dofs = dof_pos_new[0, :, :, 0].cpu().numpy()
            roots = root_positions.cpu().numpy()
            root_rotvec = root_rot_new.cpu().numpy()
            q = np.c_[roots, Rotation.from_rotvec(root_rotvec).as_quat(scalar_first=True), dofs]
            pose = torch.cat([root_rot_new[None, :, None], fk.dof_axis*dof_pos_new], dim=2)
            final_fk = fk.fk_batch(pose, root_positions[None])
            residual = final_fk.global_translation[0, :, robot_ids] - joints[:, human_ids]
            joint_error = torch.linalg.vector_norm(residual, dim=-1).cpu().numpy()
        np.savez_compressed(args.output/('person_%02d_native.npz' % person),
                            qpos=q, times=times, dof_names=np.array(names),
                            source_joints_zup=source, joint_fit_error_m=joint_error,
                            optimized_root_constant_offset=root_pos_offset.detach().cpu().numpy(),
                            loss_history=np.array([row['loss'] for row in progress.rows]))
        person_reports.append({'person': person, 'iterations': len(progress.rows),
                               'wall_seconds': time.perf_counter()-person_begun,
                               'mean_joint_fit_error_m': float(joint_error.mean()),
                               'max_joint_fit_error_m': float(joint_error.max()),
                               'optimized_root_constant_offset_m': root_pos_offset.detach().cpu().numpy().tolist(),
                               'loss_history': progress.rows})
        allq.append(q); observations.append(source)
    reference = args.output/'reference.npz'
    np.savez_compressed(reference, qpos=np.asarray(allq), times=times, dof_names=np.array(names),
                        source_joints_zup=np.asarray(observations),
                        matched_robot_bodies=np.array([body for body, _ in matches]),
                        matched_human_indices=np.array(human_ids))
    source_manifest = args.source/'source_manifest.json'
    report = {'schema': 'retarget_compare.phc_fitting.v1',
              'method': 'PHC kinematic fitting, Vita52 adapter',
              'status': 'completed_official_fitting_loop_adapted_input_and_embodiment',
              'physical_simulation_executed': False, 'phc_policy_executed': False,
              'qpos_shape': list(np.asarray(allq).shape), 'frames_per_person': len(times),
              'duration_seconds': float(times[-1]), 'world_scale': 1., 'world_origin_reset': False,
              'seed': args.seed, 'initial_joint_noise_rad': args.initial_joint_noise,
              'device': str(device), 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
              'torch_version': torch.__version__, 'mujoco_version': mj.__version__,
              'official_fk_independent_mujoco_check_max_abs_error_m': fk_error,
              'official_fk_check_poses': 8, 'fixed_bodies_composed': ['lh_hand_base_link', 'rh_hand_base_link'],
              'model': str(args.model), 'model_sha256': sha(args.model),
              'source_data': str(args.data), 'source_manifest': str(source_manifest),
              'source_manifest_sha256': sha(source_manifest) if source_manifest.exists() else None,
              'upstream_files': [{'path': str(path), 'sha256': sha(path)}
                                 for path in (fk_path, fit_path, rot_path, smooth_path)],
              'reference_sha256': sha(reference), 'persons_report': person_reports,
              'optimizer': {'algorithm': 'Adam', 'learning_rate': .02, 'iterations_per_person': args.iterations,
                            'loss': 'mean norm joint position error + .01*mean(dof^2)',
                            'post_step': 'original limit clamp + Gaussian kernel5 sigma.75 with replicate padding'},
              'total_wall_seconds': time.perf_counter()-begun,
              'adaptations': ['Executed original official Humanoid_Batch FK methods and original fitting for-loop AST from saved source, without rewriting their arithmetic.',
                 'Compiled shared MuJoCo skeleton replaces XML/mesh loader; two fixed hand bases are algebraically composed into children; independent MuJoCo FK checked.',
                 'Already decoded shared repaired SMPL-X world joints replace AMASS loading and SMPL shape refit; no additional human scale/shape optimization.',
                 'Vita body joint matches use the same14 semantic landmarks as the GMR adapter.',
                 'Original smoothing CPU-created kernel is moved to the selected device only; kernel arithmetic and filtering are unchanged.',
                 'Original post-fit whole-world mesh-height grounding is omitted to retain the fixed shared scene/world; optimized constant root offsets are recorded.',
                 'Unobserved hands/neck held neutral; no inter-person collision or seat constraints added.'],
              'limitations': ['This evaluates PHC kinematic fitting only, not PHC/PHC+ policy or physics imitation.',
                 'Vita target and decoded SMPL-X input are local adaptations, not official out-of-box benchmark reproduction.',
                 'No physical stability or contact-force claim; common independent evaluation is required.']}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['status', 'qpos_shape', 'total_wall_seconds',
                     'official_fk_independent_mujoco_check_max_abs_error_m']}), flush=True)


if __name__ == '__main__':
    main()
