"""Reproduce zero distance queries against independent exact convex-polyhedral SAT."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize,LinearConstraint
    from itertools import product
    from fourth_convex_separation import ConvexSeparationOracle
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False);sat=ConvexSeparationOracle(m)
    proxies=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n)for n in ['shoe_sole_proxy','shoe_left_wall_proxy','shoe_right_wall_proxy','shoe_toe_wall_proxy']]
    hands=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')and m.geom_contype[g]]
    for g in proxies:
        v=np.asarray(list(product(*[[-float(z),float(z)]for z in m.geom_size[g]])))
        sat.cache[g]={'vertices':v,'normals':np.eye(3),'edges':np.eye(3)}
    def qp_distance(g,h):
        verts=[];norm=[];bias=[]
        for geom in [g,h]:
            v=sat._shape(geom)['vertices']@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom];eq=ConvexHull(v).equations
            eq=np.unique(np.round(eq,11),axis=0);verts.append(v);norm.append(eq[:,:3]);bias.append(eq[:,3])
        A=np.zeros((len(norm[0])+len(norm[1]),6));A[:len(norm[0]),:3]=norm[0];A[len(norm[0]):,3:]=norm[1]
        ub=-np.r_[bias[0],bias[1]];start=np.r_[verts[0].mean(0),verts[1].mean(0)]
        def fun(z):return .5*np.sum((z[:3]-z[3:])**2)
        def jac(z):return np.r_[z[:3]-z[3:],z[3:]-z[:3]]
        sol=minimize(fun,start,jac=jac,constraints=[LinearConstraint(A,-np.inf,ub)],method='SLSQP',options={'ftol':1e-13,'maxiter':200})
        return{'success':bool(sol.success),'distance_m':float(np.linalg.norm(sol.x[:3]-sol.x[3:])),'max_inequality_violation_m':float(max(0,np.max(A@sol.x-ub))),'witness_world':sol.x.tolist()}
    rows=[]
    for frame in [260,300,397]:
        q=np.r_[ref['qpos'][frame,0],ref['qpos'][frame,1],ref['object_target_pose'][frame]]
        for g in proxies:
            for h in hands:
                d.qpos[:]=q;mj.mj_fwdPosition(m,d);wit=np.zeros(6);before=float(mj.mj_geomDistance(m,d,g,h,.5,wit));gap,_=sat.query(d,g,h)
                if before!=0 and not(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,h).endswith('thumb_distal_0')):continue
                row={'frame':frame,'geom1':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'geom2':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,h),
                     'geom_types':[int(m.geom_type[g]),int(m.geom_type[h])],'after_fwdPosition':before,'fwdPosition_witness':wit.tolist(),'convex_polyhedral_SAT_gap_m':float(gap),
                     'sat_positive_means_proven_separated':bool(gap>1e-8)}
                # Verify required derived quantities and CCD setting, independently.
                mj.mj_forward(m,d);w=np.zeros(6);row['after_full_forward']=float(mj.mj_geomDistance(m,d,g,h,.5,w));row['full_forward_witness']=w.tolist()
                bit=int(mj.mjtDisableBit.mjDSBL_NATIVECCD);original=int(m.opt.disableflags)
                m.opt.disableflags=original&~bit;mj.mj_forward(m,d);row['nativeccd_enabled']=float(mj.mj_geomDistance(m,d,g,h,.5,np.zeros(6)))
                m.opt.disableflags=original|bit;mj.mj_forward(m,d);row['nativeccd_disabled']=float(mj.mj_geomDistance(m,d,g,h,.5,np.zeros(6)));m.opt.disableflags=original
                if before==0 or 'thumb_distal'in row['geom2']:row['independent_convex_closest_point_QP']=qp_distance(g,h)
                rows.append(row)
    report={'model':str(a.model),'model_sha256':sha(a.model),'reference':str(a.reference),'reference_sha256':sha(a.reference),'mujoco_version':mj.__version__,
            'original_disableflags':int(m.opt.disableflags),'original_enableflags':int(m.opt.enableflags),'ccd_iterations':int(m.opt.ccd_iterations),'ccd_tolerance':float(m.opt.ccd_tolerance),
            'nativeccd_disabled_in_source_model':bool(int(m.opt.disableflags)&int(mj.mjtDisableBit.mjDSBL_NATIVECCD)),
            'rows':rows,'method':['Mesh shapes use convex hull of compiled mesh_vert. Boxes use exact8vertices. SAT uses face normals and cross-edge axes, independent of MuJoCo GJK.',
                                 'Positive SAT is a proven separation-plane lower bound, not an exact Euclidean metric; six-variable convex QP independently solves closest points for separated pairs.',
                                 'Query zeros are recorded, not assumed to indicate contact. Full-forward and nativeCCD/MPR comparisons are explicit diagnostics only; no source model is saved or changed.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    bad=[r for r in rows if r['after_fwdPosition']==0 and r['convex_polyhedral_SAT_gap_m']>1e-6]
    print(json.dumps({'zero_but_proven_separated_pairs':len(bad),'mujoco_version':mj.__version__,'nativeccd_disabled_source':report['nativeccd_disabled_in_source_model'],'examples':bad[:4]}),flush=True)
if __name__=='__main__':main()
