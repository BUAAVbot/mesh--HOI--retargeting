"""Inspect support geometry and loads in frozen physics states, without simulation."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import mujoco
    model = mujoco.MjModel.from_binary_path(str(a.trial/'model.mjb'))
    data = mujoco.MjData(model)
    z = np.load(a.trial/'trace.npz', allow_pickle=False)
    c = np.load(a.trial/'contacts.npz', allow_pickle=False)
    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or 'world' for b in range(model.nbody)]
    geom_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in range(model.ngeom)]
    rows = []
    for requested in (0.,.25,.5,.75,1.,1.25,1.5,1.75,2.):
        i = min(int(np.searchsorted(z['time'],requested)), len(z['time'])-1)
        data.qpos[:] = z['qpos'][i]
        data.qvel[:] = z['qvel'][i]
        mujoco.mj_forward(model,data)
        frame = {'time_s':float(z['time'][i]),'actors':[]}
        k = max(0,i-1)
        for actor in range(2):
            prefix = f'p{actor+1}_'
            bodies = np.asarray([b for b,n in enumerate(names) if n.startswith(prefix)])
            mass = model.body_mass[bodies]
            com = np.sum(mass[:,None]*data.xipos[bodies],axis=0)/mass.sum()
            lower = {}
            for g,n in enumerate(geom_names):
                b = int(model.geom_bodyid[g])
                if not names[b].startswith(prefix) or 'foot' not in n or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                    continue
                mesh = int(model.geom_dataid[g])
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    start,count = int(model.mesh_vertadr[mesh]),int(model.mesh_vertnum[mesh])
                    points = model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                    lower[n] = float(np.min(points[:,2]))
                elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    extent = float(np.sum(np.abs(data.geom_xmat[g].reshape(3,3)[2])*model.geom_size[g]))
                    lower[n] = float(data.geom_xpos[g,2]-extent)
                elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    extent = float(abs(data.geom_xmat[g].reshape(3,3)[2,2])*model.geom_size[g,1]+model.geom_size[g,0])
                    lower[n] = float(data.geom_xpos[g,2]-extent)
                elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    lower[n] = float(data.geom_xpos[g,2]-model.geom_size[g,0])
            support=[]
            for j in np.flatnonzero(c['interval_index']==k):
                b1,b2 = map(int,c['body_ids'][j])
                mine1,mine2 = names[b1].startswith(prefix),names[b2].startswith(prefix)
                if mine1 == mine2:
                    continue
                mine,other = (b1,b2) if mine1 else (b2,b1)
                f = c['force_world_on_geom2_N'][j] * (-1 if mine1 else 1)
                if f[2] <= .01:
                    continue
                support.append({'body':names[mine],'other':names[other],
                                'geoms':[geom_names[int(g)] for g in c['geom_ids'][j]],
                                'point_world_m':c['position_world_m'][j].tolist(),
                                'force_on_actor_world_N':f.tolist()})
            force = np.asarray([v['force_on_actor_world_N'] for v in support]).reshape(-1,3)
            center = np.average(np.asarray([v['point_world_m'] for v in support]),weights=force[:,2],axis=0).tolist() if len(support) else None
            frame['actors'].append({'actor':actor+1,'mass_kg':float(mass.sum()),'com_world_m':com.tolist(),
                                    'foot_collision_min_world_z_m':lower,
                                    'positive_vertical_contact_force_N':float(force[:,2].sum()),
                                    'positive_vertical_load_center_world_m':center,
                                    'upward_contacts':support,
                                    'root_orientation_error_rad':float(z['root_orientation_error_rad'][k,actor]) if 'root_orientation_error_rad' in z else None})
        rows.append(frame)
    out={'status':'frozen_state_support_diagnostic','simulation_steps_run':0,
         'trial':str(a.trial),'trace_sha256':hashlib.sha256((a.trial/'trace.npz').read_bytes()).hexdigest(),
         'frames':rows,'load_center_note':'Weighted positive vertical contact forces; not a complete static stability proof.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{ 'time':r['time_s'],'actors':[{k:v for k,v in x.items() if k!='upward_contacts'} for x in r['actors']]} for r in rows],indent=2))


if __name__ == '__main__':
    main()
