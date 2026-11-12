"""Validate the reusable fallback against independent SAT and world-plane QP."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repro',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from rgb_near_convex_distance import ShoeHandConvexDistance
    source=json.loads(a.repro.read_text());m=mj.MjModel.from_xml_path(source['model']);d=mj.MjData(m);ref=np.load(source['reference'],allow_pickle=False);oracle=ShoeHandConvexDistance(m);rows=[]
    for row in source['rows']:
        frame=row['frame'];d.qpos[:]=np.r_[ref['qpos'][frame,0],ref['qpos'][frame,1],ref['object_target_pose'][frame]];mj.mj_forward(m,d)
        g=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,row['geom1']);h=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,row['geom2']);before=d.qpos.copy();result=oracle.query(d,g,h,precise=True)
        assert np.array_equal(before,d.qpos),'State modified'
        want=row['independent_convex_closest_point_QP']['distance_m'];error=abs(result['signed_gap_m']-want)
        assert result['precise_distance_success']and error<2e-7,(row,result,error)
        rows.append({'frame':frame,'geom1':row['geom1'],'geom2':row['geom2'],'original_mj_distance':row['after_fwdPosition'],'independent_world_plane_QP_distance_m':want,'fallback':result,'abs_distance_disagreement_m':error})
    primitives=[]
    cube=' '.join(str(z)for x in[-.01,.01]for y in[-.01,.01]for zz in[-.01,.01]for z in[x,y,zz])
    for shift,expected in[(.04,.02),(.02,0.),(.015,-.005)]:
        text=f'<mujoco><asset><mesh name="cube" vertex="{cube}"/></asset><worldbody><geom type="box" size=".01 .01 .01"/><geom type="mesh" mesh="cube" pos="{shift} 0 0"/></worldbody></mujoco>'
        tm=mj.MjModel.from_xml_string(text);td=mj.MjData(tm);mj.mj_forward(tm,td);got=ShoeHandConvexDistance(tm).query(td,0,1,precise=True)
        assert abs(got['signed_gap_m']-expected)<1e-8,(got,expected)
        primitives.append({'box_mesh_centroid_separation_m':shift,'expected_signed_gap_m':expected,'actual':got})
    out={'repro':str(a.repro),'repro_sha256':sha(a.repro),'module':'scripts/rgb_near_convex_distance.py','module_sha256':sha('scripts/rgb_near_convex_distance.py'),'all_cases_passed':True,'case_count':len(rows),'max_distance_disagreement_m':max(x['abs_distance_disagreement_m']for x in rows),'rows':rows,'analytic_separated_touching_overlap_tests':primitives}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}))
if __name__=='__main__':main()
