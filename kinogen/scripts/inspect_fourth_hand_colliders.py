"""Read-only hand collider/FK audit and isolated visual/collision comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--time', type=float, default=3.06)
    a = p.parse_args()
    import mujoco as mj
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    m = (mj.MjModel.from_binary_path(str(a.model.resolve())) if a.model.suffix == '.mjb'
         else mj.MjModel.from_xml_path(str(a.model.resolve())))
    z = np.load(a.trace, allow_pickle=False)
    i = int(np.argmin(abs(z['time']-a.time)))
    d = mj.MjData(m)
    d.qpos[:] = z['qpos'][i]
    mj.mj_forward(m, d)
    bnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or 'world' for b in range(m.nbody)]
    active = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    records = []
    for g in range(m.ngeom):
        body = int(m.geom_bodyid[g])
        if bnames[body].startswith(('rh_', 'lh_')):
            records.append(dict(id=g, name=mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g),
                                body=bnames[body], type=int(m.geom_type[g]),
                                size=m.geom_size[g].tolist(), local_pos=m.geom_pos[g].tolist(),
                                contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
                                collision=bool(active[g])))
    old_pos, old_mat = d.geom_xpos.copy(), d.geom_xmat.copy()
    q0 = d.qpos.copy()
    changed = []
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ''
        if name.startswith('rh_') and name.endswith('_dip'):
            qa = int(m.jnt_qposadr[j])
            d.qpos[qa] = min(float(m.jnt_range[j, 1]), q0[qa]+.8)
            changed.append(dict(joint=name, original=float(q0[qa]), modified=float(d.qpos[qa])))
    mj.mj_forward(m, d)
    q1 = d.qpos.copy()
    ids = [r['id'] for r in records if r['body'].startswith('rh_') and r['collision']]
    delta_pos = float(np.max(abs(d.geom_xpos[ids]-old_pos[ids])))
    delta_rot = float(np.max(abs(d.geom_xmat[ids]-old_mat[ids])))
    right = [r for r in records if r['body'].startswith('rh_')]
    visual = [r['id'] for r in right if not r['collision']]
    visual_delta = float(np.max(abs(d.geom_xmat[visual]-old_mat[visual])))
    report = dict(model=str(a.model), model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  trace=str(a.trace), trace_time_s=float(z['time'][i]), geometric_diagnostic_only=True,
                  changed_joints=changed, right_hand_active_colliders=len(ids),
                  dip_change_collision_position_max_delta_m=delta_pos,
                  dip_change_collision_rotation_max_delta=delta_rot,
                  dip_change_visual_rotation_max_delta=visual_delta, hand_geometries=records)
    # Colours/groups below affect this unsaved renderer model only.
    m.geom_group[:] = 5
    for r in right:
        m.geom_group[r['id']] = 0 if r['collision'] else 1
    hand = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'rh_hand_base_link')
    d.qpos[:] = q0
    mj.mj_forward(m, d)
    rotation = d.xmat[hand].reshape(3, 3)
    centre = d.xpos[hand]+rotation @ np.array([0., 0., .095])
    eye = centre+rotation @ np.array([.25, .20, 0.])
    forward = centre-eye
    forward /= np.linalg.norm(forward)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = centre
    camera.distance = np.linalg.norm(centre-eye)
    camera.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    camera.elevation = np.degrees(np.arcsin(forward[2]))
    m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
    m.vis.headlight.ambient[:] = [.6, .6, .6]
    renderer = mj.Renderer(m, height=480, width=640)
    opt = mj.MjvOption()
    canvas = Image.new('RGB', (1920, 1080), '#141b23')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    try:
        for row, q in enumerate((q0, q1)):
            d.qpos[:] = q
            mj.mj_forward(m, d)
            for col, label in enumerate(('Visual meshes', 'Active collision shapes', 'Overlay: collision cyan')):
                opt.geomgroup[:] = [col != 0, col != 1, 0, 0, 0, 0]
                for r in right:
                    m.geom_rgba[r['id']] = ([.1, .75, 1., .65 if col == 2 else 1.]
                        if r['collision'] else [.7, .72, .74, .2 if col == 2 else 1.])
                renderer.update_scene(d, camera=camera, scene_option=opt)
                for view in renderer.scene.camera:
                    view.up[:] = rotation[:, 2]
                canvas.paste(Image.fromarray(renderer.render()), (640*col, 60+540*row))
                draw.text((640*col+12, 540*row+8), label, font=font, fill='white')
                draw.text((640*col+12, 540*row+33), 'Trace joint pose | geometry audit' if row == 0 else 'Only 4 DIP joints changed (+0.8 rad, limited)', font=font, fill='#ffcb60')
    finally:
        renderer.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'hand_geometries'}), flush=True)


if __name__ == '__main__':
    main()
