"""Shared NumPy URDF FK and body-surface geometry; original assets stay read-only."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
BODY_JOINT_NAMES=[f'{side}_{name}_joint'for side in ('left','right')for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]+[f'waist_{axis}_joint'for axis in ('yaw','roll','pitch')]+[f'{side}_{name}_joint'for side in ('left','right')for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]


def skew(v):
    x,y,z=v
    return np.asarray([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def origin(el):
    matrix=np.eye(4)
    if el is not None:
        matrix[:3,3]=np.fromstring(el.attrib.get('xyz','0 0 0'),sep=' ')
        matrix[:3,:3]=Rotation.from_euler('xyz',np.fromstring(el.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return matrix


class G1Kinematics:
    """Body state = xyz[0:3], base rotation-vector[3:6], body joints[6:35].

    World metres, Z up, +X robot forward, +Y robot left. Finger q has no
    influence on wrist/palm transforms. All FK transforms map link -> world.
    """

    def __init__(self,project_root=ROOT,proxy_faces_per_link=260,urdf_path=None,scene_path=None):
        self.root=Path(project_root)
        scene_file=Path(scene_path)if scene_path else self.root/'data/video_generation/candidate_scene/scene.json'
        if not scene_file.is_absolute():scene_file=self.root/scene_file
        self.scene=json.loads(scene_file.read_text())
        self.urdf_path=Path(urdf_path)if urdf_path else self.root/self.scene['robot']['urdf']
        if not self.urdf_path.is_absolute():self.urdf_path=self.root/self.urdf_path
        self.urdf_sha256=hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        xml=ET.parse(self.urdf_path).getroot()
        self.link_names=[link.attrib['name'] for link in xml.findall('link')]
        self.link_index={name:i for i,name in enumerate(self.link_names)}
        joints=list(xml.findall('joint'));ordered=[];known={'pelvis'}
        while joints:
            done=[]
            for joint in joints:
                if joint.find('parent').attrib['link'] in known:
                    ordered.append(joint);known.add(joint.find('child').attrib['link']);done.append(joint)
            if not done:raise RuntimeError('Unconnected URDF joints')
            joints=[j for j in joints if j not in done]
        self.body_joint_names=[j.attrib['name'] for j in ordered if j.attrib['name']in BODY_JOINT_NAMES]
        self.finger_joint_names=[j.attrib['name'] for j in ordered if j.attrib['type']!='fixed' and j.attrib['name']not in BODY_JOINT_NAMES]
        self.finger_joint_types={j.attrib['name']:j.attrib['type']for j in ordered if j.attrib['name']in self.finger_joint_names}
        if set(self.body_joint_names)!=set(BODY_JOINT_NAMES):raise RuntimeError('Expected unchanged 29 G1 body joints')
        self.body_index={name:i for i,name in enumerate(self.body_joint_names)}
        self.joints=[];self.ancestor=np.zeros((len(self.link_names),29),dtype=bool);self.body_limits=np.zeros((29,2))
        for j in ordered:
            name=j.attrib['name'];p=self.link_index[j.find('parent').attrib['link']];c=self.link_index[j.find('child').attrib['link']]
            axis=np.fromstring(j.find('axis').attrib.get('xyz','1 0 0'),sep=' ') if j.find('axis') is not None else np.zeros(3)
            if j.attrib['type']!='fixed':
                length=float(np.linalg.norm(axis))
                if not np.isfinite(length)or length<=1e-12:raise ValueError(f'Invalid nonfixed joint axis: {name}')
                axis=axis/length
            self.joints.append((name,p,c,origin(j.find('origin')),axis,j.attrib['type']))
            self.ancestor[c]=self.ancestor[p]
            if name in self.body_index:
                k=self.body_index[name];self.ancestor[c,k]=True;limit=j.find('limit');self.body_limits[k]=[float(limit.attrib['lower']),float(limit.attrib['upper'])]
        initial_joints=self.scene['robot'].get('joint_positions',self.scene['robot'].get('joint_positions_rad',{}))
        self.q0_fingers={name:float(initial_joints.get(name,0.)) for name in self.finger_joint_names}
        self.initial_state=np.r_[self.scene['robot']['base_position_xyz_m'],Rotation.from_quat(self.scene['robot']['base_quaternion_xyzw']).as_rotvec(),[self.scene['robot']['joint_positions_rad'][name] for name in self.body_joint_names]].astype(float)
        self.visuals=[];self.proxy_vertices=[];self.proxy_vertex_links=[];self.proxy_faces=[];self.proxy_face_links=[];offset=0
        import trimesh
        import open3d as o3d
        colors={m.attrib['name']:list(map(float,m.find('color').attrib['rgba'].split()))[:3] for m in xml.findall('material')}
        for link in xml.findall('link'):
            name=link.attrib['name'];index=self.link_index[name]
            for visual in link.findall('visual'):
                mesh_el=visual.find('geometry/mesh')
                if mesh_el is None:continue
                filename=mesh_el.attrib['filename']
                if filename.startswith('package://'):
                    package,relative=filename[len('package://'):].split('/',1)
                    package_roots={'g1_description':self.root/'dependencies/unitree-ros/robots/g1_description','rohand_urdf':self.root/'dependencies/rohand-urdf'}
                    if package not in package_roots:raise RuntimeError('Unknown URDF mesh package: '+package)
                    path=package_roots[package]/relative
                else:path=self.urdf_path.parent/filename
                mesh=trimesh.load_mesh(path,process=False)
                mesh.vertices*=np.fromstring(mesh_el.attrib.get('scale','1 1 1'),sep=' ');mesh.apply_transform(origin(visual.find('origin')))
                mat=visual.find('material');color=colors.get(mat.attrib.get('name','') if mat is not None else '',[.65]*3)
                if mat is not None and mat.find('color') is not None:color=list(map(float,mat.find('color').attrib['rgba'].split()))[:3]
                self.visuals.append({'link_name':name,'link_index':index,'vertices_local':np.asarray(mesh.vertices).copy(),'faces':np.asarray(mesh.faces).copy(),'color':color,'mesh_path':str(path),'mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                target=90 if '_hand_' in name else proxy_faces_per_link
                simple=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
                simple.remove_duplicated_vertices();simple.remove_degenerate_triangles();simple.remove_duplicated_triangles()
                if len(simple.triangles)>target:simple=simple.simplify_quadric_decimation(target)
                v=np.asarray(simple.vertices);f=np.asarray(simple.triangles)
                self.proxy_vertices.append(v.copy());self.proxy_vertex_links.append(np.full(len(v),index,dtype=np.int32));self.proxy_faces.append(f.copy()+offset);self.proxy_face_links.append(np.full(len(f),index,dtype=np.int32));offset+=len(v)
        self.proxy_vertices=np.concatenate(self.proxy_vertices);self.proxy_vertex_links=np.concatenate(self.proxy_vertex_links);self.proxy_faces=np.concatenate(self.proxy_faces).astype(np.uint32);self.proxy_face_links=np.concatenate(self.proxy_face_links)

    def fk(self,state,finger_q=None,return_axes=False):
        state=np.asarray(state);transforms=np.tile(np.eye(4),(len(self.link_names),1,1));base=transforms[self.link_index['pelvis']]
        base[:3,:3]=Rotation.from_rotvec(state[3:6]).as_matrix();base[:3,3]=state[:3]
        finger=self.q0_fingers if finger_q is None else {**self.q0_fingers,**finger_q}
        axes=np.zeros((29,3));centers=np.zeros((29,3))
        for name,p,c,fixed,axis,kind in self.joints:
            joint_world=transforms[p]@fixed;motion=np.eye(4)
            if kind!='fixed':
                angle=state[6+self.body_index[name]] if name in self.body_index else finger[name]
                if kind in ('revolute','continuous'):motion[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
                elif kind=='prismatic':motion[:3,3]=axis*angle
                else:raise RuntimeError('Unsupported motion joint: '+kind)
                if name in self.body_index:
                    k=self.body_index[name];axes[k]=joint_world[:3,:3]@axis;centers[k]=joint_world[:3,3]
            transforms[c]=joint_world@motion
        return (transforms,axes,centers) if return_axes else transforms

    def forward(self,base_xyz_m,base_quaternion_xyzw,body_joint_positions_rad,finger_joint_positions=None,**kwargs):
        finger=None
        if finger_joint_positions is None:finger_joint_positions=kwargs.get('finger_positions_rad')
        if finger_joint_positions is not None:
            finger=finger_joint_positions if isinstance(finger_joint_positions,dict) else dict(zip(self.finger_joint_names,finger_joint_positions))
        return self.fk(np.r_[base_xyz_m,Rotation.from_quat(base_quaternion_xyzw).as_rotvec(),body_joint_positions_rad],finger)

    def points(self,state,link_ids,local_points,with_jacobian=False):
        transforms,axes,centers=self.fk(state,return_axes=True)
        world=np.einsum('nij,nj->ni',transforms[link_ids,:3,:3],local_points)+transforms[link_ids,:3,3]
        if not with_jacobian:return world
        jac=np.zeros((len(world),3,35));jac[:,:,:3]=np.eye(3)
        r=np.asarray(state[3:6]);theta=np.linalg.norm(r);W=skew(r)
        Jr=np.eye(3)-.5*W+W@W/6 if theta<1e-6 else np.eye(3)-(1-np.cos(theta))/theta**2*W+(theta-np.sin(theta))/theta**3*(W@W)
        root_rotation=transforms[self.link_index['pelvis'],:3,:3];root_local=(world-state[:3])@root_rotation
        cross=np.zeros((len(world),3,3));cross[:,0,1]=-root_local[:,2];cross[:,0,2]=root_local[:,1];cross[:,1,0]=root_local[:,2];cross[:,1,2]=-root_local[:,0];cross[:,2,0]=-root_local[:,1];cross[:,2,1]=root_local[:,0]
        jac[:,:,3:6]=-np.einsum('ij,njk,kl->nil',root_rotation,cross,Jr)
        for joint in range(29):
            selected=self.ancestor[link_ids,joint]
            jac[selected,:,6+joint]=np.cross(axes[joint],world[selected]-centers[joint])
        return world,jac

    def proxy_world(self,state):
        return self.points(state,self.proxy_vertex_links,self.proxy_vertices)


def project_points(P,points,with_jacobian=False):
    homogeneous=np.column_stack((points,np.ones(len(points))))@P.T
    z=homogeneous[:,2];uv=homogeneous[:,:2]/z[:,None]
    if not with_jacobian:return uv
    jac=(P[None,:2,:3]*z[:,None,None]-homogeneous[:,:2,None]*P[None,2:3,:3])/z[:,None,None]**2
    return uv,jac
