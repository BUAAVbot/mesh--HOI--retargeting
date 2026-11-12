"""Independent keyframe hand-shoe SAT audit; never modify the reference or model."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',default=[250,260,270,300,397]);a=p.parse_args()
    import numpy as np,mujoco as mj
    from rgb_near_convex_distance import ShoeHandConvexDistance
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False);oracle=ShoeHandConvexDistance(m)
    hands=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')and m.geom_contype[g]]
    shoe=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n)for n in ['shoe_sole_proxy','shoe_left_wall_proxy','shoe_right_wall_proxy','shoe_toe_wall_proxy']]
    records=[]
    for frame in a.frames:
        d.qpos[:]=np.r_[ref['qpos'][frame,0],ref['qpos'][frame,1],ref['object_target_pose'][frame]];mj.mj_forward(m,d);rows=[];native=[]
        for g in hands:
            vals=[]
            for h in shoe:
                v=oracle.query(d,g,h,precise=False);vals.append({'hand_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'shoe_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,h),'sat_gap_m':v['sat_gap_m']})
            nearest=min(vals,key=lambda v:v['sat_gap_m']);h=shoe[[x['shoe_geom']for x in vals].index(nearest['shoe_geom'])]
            precise=oracle.query(d,g,h,precise=True);nearest['precise']=precise;rows.extend(vals)
        for i in range(d.ncon):
            c=d.contact[i];pair={int(c.geom1),int(c.geom2)}
            if not(pair&set(hands)and pair&set(shoe)):continue
            g=int(c.geom1);h=int(c.geom2);v=oracle.query(d,g,h,precise=False)
            native.append({'geom1':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'geom2':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,h),'native_dist_m':float(c.dist),'sat_gap_m':v['sat_gap_m'],'native_contact_false_if_positive_sat':bool(v['sat_gap_m']>1e-6)})
        records.append({'frame':frame,'time_s':float(ref['time_s'][frame]),'pair_distances':rows,'penetrating_pairs':[r for r in rows if r['sat_gap_m']<-.0001],'native_contacts':native})
    report={'model':str(a.model),'model_sha256':sha(a.model),'reference':str(a.reference),'reference_sha256':sha(a.reference),'method':'Exact compiled box+mesh convex hulls SAT, nearest-pair convex QP; native contacts cross-checked independently.','frames':records}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'frame':f['frame'],'penetration_pairs':f['penetrating_pairs'],'native_contacts':f['native_contacts']}for f in records]),flush=True)
if __name__=='__main__':main()
