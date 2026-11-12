"""Numeric joint-regressor targets from the supplied posed SMPL-X meshes.

The regression is a retargeting estimate, not an independent keypoint or force
measurement. No SMPL-X pickle or source-package executable is evaluated.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model-npz',type=Path)
    p.add_argument('--model-zip',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if (a.model_npz is None)==(a.model_zip is None):p.error('Choose exactly one numeric model source')
    import numpy as np
    if a.model_npz:
        model=np.load(a.model_npz,allow_pickle=False);model_source={'path':str(a.model_npz),'sha256':sha(a.model_npz)}
    else:
        with zipfile.ZipFile(a.model_zip)as z:
            raw=z.read('models/smplx/SMPLX_NEUTRAL.npz')
        model=np.load(io.BytesIO(raw),allow_pickle=False)
        model_source={'archive':str(a.model_zip),'sha256':sha(a.model_zip),'member':'models/smplx/SMPLX_NEUTRAL.npz',
                      'member_sha256':hashlib.sha256(raw).hexdigest()}
    reg=model['J_regressor']
    if reg.shape!=(55,10475)or not np.isfinite(reg).all():raise ValueError('Unexpected SMPL-X joint regressor')
    timeline=np.load(a.data/'timeline.npz',allow_pickle=False)
    times=timeline['pts_seconds'];frames=timeline['frame_ids']
    if times.shape!=(398,)or np.any(np.diff(times)<=0):raise ValueError('Expected full398 source timeline')
    out=dict(time_s=times-times[0],frame_ids=frames,smplx_joint_regressor=reg)
    source_hashes={}
    for person in(1,2):
        directory=a.data/f'human/person_{person:02d}'
        h=np.load(directory/'smplx_meshes_world.npz',allow_pickle=False)
        r=np.load(directory/'root_poses_world.npz',allow_pickle=False)
        vertices=h['vertices_world']
        if vertices.shape!=(398,10475,3)or not np.all(h['valid']):raise ValueError('Source mesh missing/nonmatching')
        out[f'person{person}_joints_world']=np.einsum('jv,tvc->tjc',reg,vertices,optimize=True)
        out[f'person{person}_root_xyz_wxyz']=np.c_[r['position_m'],r['quaternion_wxyz']]
        source_hashes[f'person_{person:02d}_mesh']=sha(directory/'smplx_meshes_world.npz')
        source_hashes[f'person_{person:02d}_root']=sha(directory/'root_poses_world.npz')
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out)
    report={'schema':'rgb_near_numeric_joint_targets.v1','frames':398,'actors':2,
            'method':'SMPL-X J_regressor applied to posed world mesh; approximate retargeting targets',
            'regressor_sha256':hashlib.sha256(reg.tobytes()).hexdigest(),'model_source':model_source,
            'source_file_sha256':source_hashes,'source_timeline_sha256':sha(a.data/'timeline.npz'),
            'output_sha256':sha(a.output),'script_sha256':sha(__file__),'physics_enabled':False}
    a.output.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
