"""Deterministic polyhedral separating-axis constraint for private hand queries.

Positive gap is a conservative lower bound on Euclidean separation, not an
exact distance. Negative gap indicates overlap of the queried convex solids.
All shapes are derived from the frozen compiled model. Native collision and
physical forces remain MuJoCo's responsibility; this module never mutates them.
"""
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull


def _directions(vectors):
    rows=np.asarray(vectors,float)
    rows=rows[np.linalg.norm(rows,axis=1)>1e-10]
    rows/=np.linalg.norm(rows,axis=1)[:,None]
    for row in rows:
        first=np.flatnonzero(abs(row)>1e-9)
        if len(first)and row[first[0]]<0:row*=-1
    rows=np.unique(np.round(rows,7),axis=0)
    return rows/np.linalg.norm(rows,axis=1)[:,None]


class ConvexSeparationOracle:
    def __init__(self,model):
        self.model=model;self.cache={};self.calls=0;self.cylinder_errors={}

    def _shape(self,g):
        if g in self.cache:return self.cache[g]
        m=self.model;kind=m.geom_type[g]
        if kind==mj.mjtGeom.mjGEOM_MESH:
            mesh=int(m.geom_dataid[g]);adr=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
            vertices=np.asarray(m.mesh_vert[adr:adr+count],float).copy()
        elif kind==mj.mjtGeom.mjGEOM_CYLINDER:
            # A circumscribed polygon is a conservative superset. The physical
            # cylinder is unchanged and its actual contacts still set force.
            count=32;radius=float(m.geom_size[g,0]);height=float(m.geom_size[g,1])
            angle=np.arange(count)*2*np.pi/count;outer=radius/np.cos(np.pi/count)
            vertices=np.asarray([[outer*np.cos(t),outer*np.sin(t),z]for z in [-height,height]for t in angle])
            self.cylinder_errors[m.geom(g).name]=outer-radius
        else:raise ValueError('Unsupported convex query primitive '+str(kind))
        hull=ConvexHull(vertices);normals=hull.equations[:,:3]
        adjacency={}
        for face,tri in enumerate(hull.simplices):
            for i,j in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
                adjacency.setdefault(tuple(sorted((int(i),int(j)))),[]).append(face)
        edges=[]
        for(i,j),faces in adjacency.items():
            if len(faces)!=2 or abs(float(normals[faces[0]]@normals[faces[1]]))<1-1e-7:
                edges.append(vertices[j]-vertices[i])
        result={'vertices':vertices[hull.vertices],'normals':_directions(normals),'edges':_directions(edges)}
        self.cache[g]=result;return result

    def query(self,kin,hand_geom,object_geom,velocity_indices=None):
        self.calls+=1;A=self._shape(hand_geom);B=self._shape(object_geom)
        RA=kin.geom_xmat[hand_geom].reshape(3,3);RB=kin.geom_xmat[object_geom].reshape(3,3)
        va=A['vertices']@RA.T+kin.geom_xpos[hand_geom];vb=B['vertices']@RB.T+kin.geom_xpos[object_geom]
        na=A['normals']@RA.T;nb=np.concatenate([B['normals']@RB.T,np.eye(3)])
        ea=A['edges']@RA.T;eb=B['edges']@RB.T
        uu=np.repeat(ea,len(eb),axis=0);vv=np.tile(eb,(len(ea),1));cross=np.cross(uu,vv);length=np.linalg.norm(cross,axis=1)
        mask=length>1e-8;uu=uu[mask];vv=vv[mask];length=length[mask];cross=cross[mask]/length[:,None]
        axes=np.concatenate([na,nb,cross]);pa=np.einsum('ij,kj->ik',axes,va);pb=np.einsum('ij,kj->ik',axes,vb)
        gap_positive=pa.min(1)-pb.max(1);gap_negative=pb.min(1)-pa.max(1)
        gap=np.maximum(gap_positive,gap_negative);best=float(gap.max())
        if velocity_indices is None:return best,None
        active=np.flatnonzero(gap>=best-1e-10);rows=[]
        for axis_index in active:
            n=axes[axis_index];sign=1. if gap_positive[axis_index]>=gap_negative[axis_index]else-1.
            if sign>0:
                av=va[pa[axis_index]<=pa[axis_index].min()+1e-10].mean(0)
                bv=vb[pb[axis_index]>=pb[axis_index].max()-1e-10].mean(0)
            else:
                av=va[pa[axis_index]>=pa[axis_index].max()-1e-10].mean(0)
                bv=vb[pb[axis_index]<=pb[axis_index].min()+1e-10].mean(0)
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
            mj.mj_jac(self.model,kin,jp,jr,av,int(self.model.geom_bodyid[hand_geom]))
            omega=jr[:,velocity_indices].T
            if axis_index<len(na):dn=np.cross(omega,n)
            elif axis_index<len(na)+len(nb):dn=np.zeros_like(omega)
            else:
                k=axis_index-len(na)-len(nb);raw=np.cross(np.cross(omega,uu[k]),vv[k])
                dn=(raw-(raw@n)[:,None]*n)/length[k]
            rows.append(sign*(n@jp[:,velocity_indices]+dn@(av-bv)))
        return best,np.mean(rows,axis=0)

    def report(self):
        return {'method':'Convex-polyhedral separating-axis signed gap; exact mesh convex hulls, circumscribed 32-gon for physical bottom cylinders.',
                'positive_value_is_conservative_euclidean_clearance_lower_bound':True,
                'cylinder_query_only_radial_superset_error_m':self.cylinder_errors,
                'cached_geoms':len(self.cache),'calls':self.calls,
                'cache_inputs':'Frozen compiled mesh_vert, geom type and size; no external mesh files.',
                'axis_jacobian':'Hand-face angular derivative and normalized cross-edge angular derivative included.',
                'world_aabb_axes_included_for_proven_lower_bound_pruning':True,
                'physical_collision_geometry_or_state_mutated':False}
