"""Read-only mesh/collision-frame audit for Gallery08's support surfaces."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(v):
    return {str(q): float(np.percentile(v, q)) for q in (0, 5, 25, 50, 75, 95, 100)} if len(v) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = a.base
    inp = base / 'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3'
    refp = base / 'exo_compare_20261030/videomimic_gallery_08_v2/person_01/omni/reference.npz'
    meshp = Path('/data/yihang.bai/incoming/aliyun_hoi_result/exo/videomimic_gallery_08_v2/scene/mesh.npz')
    modelp = base / 'outputs/gallery08_p01_trajopt_20261030_2205/baseline_instrumented/compiled_model.mjb'
    meta = json.loads((inp / 'mjlab_motion.json').read_text())
    scene = json.loads((inp / 'scene.json').read_text())
    z = np.load(inp / 'mjlab_motion.npz', allow_pickle=False)
    ref = np.load(refp, allow_pickle=False)
    mesh = np.load(meshp, allow_pickle=False)
    rot = ref['world_from_source_rotation']
    floor = float(np.asarray(ref['floor_offset']).reshape(-1)[0])
    v = np.asarray(mesh['vertices'], dtype=float) @ rot.T
    v[:, :2] -= np.asarray(scene['xy_offset'])
    v[:, 2] -= floor
    f = np.asarray(mesh['faces'], dtype=int)
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    horizontal = np.abs(normals[:, 2]) / np.maximum(norms, 1e-20) > .85
    lo, hi = tri[:, :, :2].min(1), tri[:, :, :2].max(1)
    def inspect_xy(xy):
        xy = np.asarray(xy)
        radial = np.linalg.norm(v[:, :2] - xy, axis=1)
        idx = np.flatnonzero(np.all((xy >= lo-1e-8) & (xy <= hi+1e-8), axis=1))
        t = tri[idx]
        if len(t):
            one = t[:, 1, :2] - t[:, 0, :2]
            two = t[:, 2, :2] - t[:, 0, :2]
            q = xy - t[:, 0, :2]
            det = one[:, 0]*two[:, 1] - one[:, 1]*two[:, 0]
            good = np.abs(det) > 1e-12
            u = (q[:, 0]*two[:, 1] - q[:, 1]*two[:, 0]) / np.where(good, det, 1)
            w = (one[:, 0]*q[:, 1] - one[:, 1]*q[:, 0]) / np.where(good, det, 1)
            good &= (u >= -1e-6) & (w >= -1e-6) & (u+w <= 1+1e-6)
            heights = t[:, 0, 2] + u*(t[:, 1, 2]-t[:, 0, 2]) + w*(t[:, 2, 2]-t[:, 0, 2])
            hits = [dict(z=float(heights[i]), horizontal=bool(horizontal[idx[i]]), face=int(idx[i])) for i in np.flatnonzero(good)]
        else:
            hits=[]
        nearest = np.argsort(radial)[:min(10, len(v))]
        return dict(xy=xy.tolist(), triangle_hits=hits,
                    neighborhoods={str(r): dict(vertices=int(np.sum(radial<=r)), height_percentiles=stats(v[radial<=r,2])) for r in (.05,.10,.20,.35,.50)},
                    nearest_vertices=[dict(distance_xy_m=float(radial[i]),xyz=v[i].tolist()) for i in nearest])
    m = mj.MjModel.from_binary_path(str(modelp)); d = mj.MjData(m)
    jnames = meta['joint_names']; bnames = meta['body_names']; rootid=bnames.index('base_link')
    qadr = [m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'robot/'+n)] for n in jnames]
    geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    sample_frames = (0, 60, 140, 200, len(z['joint_pos'])-1)
    rows=[]
    for frame in sample_frames:
        mj.mj_resetData(m,d)
        d.qpos[:3]=z['body_pos_w'][frame,rootid]
        d.qpos[3:7]=z['body_quat_w'][frame,rootid]
        d.qpos[qadr]=z['joint_pos'][frame]
        mj.mj_forward(m,d)
        feet=[]
        for side in ('l','r'):
            gids=[i for i,n in enumerate(geom_names) if 'foot' in n and n.endswith('_'+side+'_box')]
            # Sole-box corners are physical collision geometry samples.
            samples=[]
            for gid in gids:
                signs=np.array([[x,y,zs] for x in (-1,1) for y in (-1,1) for zs in (-1,1)])
                corners=(signs*m.geom_size[gid]) @ d.geom_xmat[gid].reshape(3,3).T + d.geom_xpos[gid]
                corners=corners[np.argsort(corners[:,2])[:4]]
                lowest=corners[np.argmin(corners[:,2])]
                sample=inspect_xy(lowest[:2]);sample.update(geom=geom_names[gid],sole_point=lowest.tolist())
                h=[r['z'] for r in sample['triangle_hits'] if r['horizontal']]
                sample['sole_gap_to_highest_horizontal_mesh_m']=float(lowest[2]-max(h)) if h else None
                samples.append(sample)
            feet.append(dict(side=side,samples=samples))
        rows.append(dict(frame=frame, root=z['body_pos_w'][frame,rootid].tolist(), feet=feet))
    boxrows=[]
    for box in scene['boxes']:
        row=inspect_xy(box['pos'][:2]);row.update(name=box['name'],collision_top_z=box['pos'][2]+box['size'][2]/2)
        grid=[]
        for dx in np.linspace(-.45,.45,9)*box['size'][0]:
            for dy in np.linspace(-.45,.45,9)*box['size'][1]:
                hit=inspect_xy(np.asarray(box['pos'][:2])+[dx,dy])
                hs=[h['z'] for h in hit['triangle_hits'] if h['horizontal'] and -.05<=h['z']<=.35]
                grid.append(dict(xy=hit['xy'],mesh_surface_z=max(hs) if hs else None))
        covered=[h['mesh_surface_z'] for h in grid if h['mesh_surface_z'] is not None]
        top=row['collision_top_z']
        row['surface_grid_9x9']=grid
        row['grid_summary']=dict(total=81,mesh_covered=len(covered),
            within_15mm_of_box_top=sum(abs(h-top)<=.015 for h in covered),
            box_overstates_mesh_by_more_than_50mm=sum(top-h>.05 for h in covered),
            mesh_z_percentiles=stats(np.asarray(covered)))
        boxrows.append(row)
    out=dict(schema='hdmi.mesh_ground_audit.v1',read_only_inputs=True,
        mesh_path=str(meshp),mesh_sha256=sha(meshp),reference_path=str(refp),reference_sha256=sha(refp),
        scene_sha256=sha(inp/'scene.json'),source_sha256=sha(inp/'mjlab_motion.npz'),model_sha256=sha(modelp),
        transform=dict(rotation=rot.tolist(),floor_offset=floor,xy_offset=scene['xy_offset'],formula='vertices @ rotation.T; XY -= xy_offset; Z -= floor_offset'),
        vertices=len(v),faces=len(f),horizontal_faces=int(horizontal.sum()),
        bounds=[v.min(0).tolist(),v.max(0).tolist()],all_z_percentiles=stats(v[:,2]),
        frames=rows,box_center_mesh=boxrows,
        uncertainty='A ray without a triangle hit indicates missing mesh coverage, not proof of a zero-height floor.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    brief=[]
    for row in rows:
        brief.append(dict(frame=row['frame'],samples=[dict(geom=s['geom'],sole_z=s['sole_point'][2],gap=s['sole_gap_to_highest_horizontal_mesh_m'],triangle_hits=len(s['triangle_hits']),near_20cm=s['neighborhoods']['0.2']) for foot in row['feet'] for s in foot['samples']]))
    print(json.dumps(dict(bounds=out['bounds'],z=out['all_z_percentiles'],samples=brief),indent=2),flush=True)


if __name__=='__main__':main()
