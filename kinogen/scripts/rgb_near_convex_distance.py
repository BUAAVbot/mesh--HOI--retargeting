"""Read-only convex box/mesh distance fallback, independent of MuJoCo GJK.

Use one cached ShoeHandConvexDistance(model), then query(data, geom_a, geom_b,
precise=True). Positive precise gaps use a six-variable convex closest-point QP;
negative gaps are polyhedral SAT overlap. Never writes model/data state.
"""
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull
from scipy.optimize import minimize,LinearConstraint
from itertools import product

def _directions(rows):
    rows=np.asarray(rows,float);rows=rows[np.linalg.norm(rows,axis=1)>1e-10]
    rows/=np.linalg.norm(rows,axis=1)[:,None]
    for row in rows:
        nz=np.flatnonzero(abs(row)>1e-9)
        if len(nz)and row[nz[0]]<0:row*=-1
    rows=np.unique(np.round(rows,9),axis=0)
    return rows/np.linalg.norm(rows,axis=1)[:,None]

class ShoeHandConvexDistance:
    def __init__(self,model):
        self.model=model;self.cache={};self.calls=0

    def _shape(self,g):
        if g in self.cache:return self.cache[g]
        m=self.model;kind=int(m.geom_type[g])
        if kind==mj.mjtGeom.mjGEOM_BOX:
            v=np.asarray(list(product(*[[-float(z),float(z)]for z in m.geom_size[g]])))
        elif kind==mj.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[g]);i=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=m.mesh_vert[i:i+n].copy()
        else:raise ValueError('Fallback supports only exact box or compiled mesh convex hull, got '+str(kind))
        hull=ConvexHull(v);normal=hull.equations[:,:3];adj={}
        for fi,tri in enumerate(hull.simplices):
            for i,j in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:adj.setdefault(tuple(sorted((int(i),int(j)))),[]).append(fi)
        edges=[v[j]-v[i]for(i,j),fs in adj.items()if len(fs)!=2 or abs(float(normal[fs[0]]@normal[fs[1]]))<1-1e-9]
        eq=np.unique(np.round(hull.equations,11),axis=0)
        out={'vertices':v[hull.vertices],'normals':_directions(normal),'edges':_directions(edges),'local_planes':eq}
        self.cache[g]=out;return out

    def query(self,data,g1,g2,precise=True):
        self.calls+=1;A=self._shape(g1);B=self._shape(g2)
        RA=data.geom_xmat[g1].reshape(3,3);RB=data.geom_xmat[g2].reshape(3,3);pA=data.geom_xpos[g1];pB=data.geom_xpos[g2]
        va=A['vertices']@RA.T+pA;vb=B['vertices']@RB.T+pB
        ea=A['edges']@RA.T;eb=B['edges']@RB.T;cross=np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3);lens=np.linalg.norm(cross,axis=1);cross=cross[lens>1e-9]/lens[lens>1e-9,None]
        axes=np.r_[A['normals']@RA.T,B['normals']@RB.T,np.eye(3),cross]
        pa=axes@va.T;pb=axes@vb.T;gaps=np.maximum(pa.min(1)-pb.max(1),pb.min(1)-pa.max(1));idx=int(np.argmax(gaps));gap=float(gaps[idx])
        out={'backend':'compiled_convex_SAT','signed_gap_m':gap,'sat_gap_m':gap,'distance_kind':'separation_lower_bound'if gap>0 else'overlap_min_axis',
             'axis_world':axes[idx].tolist(),'proven_separated':bool(gap>1e-8),'precise_distance_success':False,'qp_distance_m':None,'qp_max_violation_m':None}
        if gap<=0 or not precise:return out
        na=A['local_planes'][:,:3]@RA.T;nb=B['local_planes'][:,:3]@RB.T
        ba=A['local_planes'][:,3]-na@pA;bb=B['local_planes'][:,3]-nb@pB
        mat=np.zeros((len(na)+len(nb),6));mat[:len(na),:3]=na;mat[len(na):,3:]=nb;ub=-np.r_[ba,bb]
        start=np.r_[va.mean(0),vb.mean(0)]
        def fun(z):return .5*np.sum((z[:3]-z[3:])**2)
        def jac(z):return np.r_[z[:3]-z[3:],z[3:]-z[:3]]
        sol=minimize(fun,start,jac=jac,constraints=[LinearConstraint(mat,-np.inf,ub)],method='SLSQP',options={'ftol':1e-13,'maxiter':200})
        dist=float(np.linalg.norm(sol.x[:3]-sol.x[3:]));violation=float(max(0,np.max(mat@sol.x-ub)))
        ok=bool(sol.success and violation<1e-8 and dist>=gap-1e-7)
        out.update(qp_distance_m=dist,qp_max_violation_m=violation,precise_distance_success=ok,qp_status=str(sol.message),qp_witness_world=sol.x.tolist())
        if ok:out.update(backend='compiled_convex_SAT_plus_QP',signed_gap_m=dist,distance_kind='separated_euclidean')
        return out
