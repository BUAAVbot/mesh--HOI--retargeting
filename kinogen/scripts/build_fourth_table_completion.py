"""Build a visible-surface + initial-support convex table completion hypothesis.

This does not claim the occluded true edge was measured. It preserves every
measured connected tabletop vertex and six initial support footprints, then
uses their minimal convex completion as a declared collision proxy.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
import mujoco as mj

R=Path(__file__).resolve().parents[1]
D=R/'data/fourth_sim_handoff'
A=R/'outputs/fourth_full_sequence/assets'
E=R/'outputs/fourth_full_sequence/table_surface_audit'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return ' '.join(format(float(x),'.17g')for x in np.asarray(v).reshape(-1))


def edge_x(polygon,y):
    values=[]
    for p,q in zip(polygon,np.roll(polygon,-1,axis=0)):
        if min(p[1],q[1])-1e-10<=y<=max(p[1],q[1])+1e-10:
            if abs(q[1]-p[1])<1e-10:values.extend([float(p[0]),float(q[0])])
            else:values.append(float(p[0]+(y-p[1])*(q[0]-p[0])/(q[1]-p[1])))
    return [min(values),max(values)]if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',type=Path,default=A/'vita_six_objects_newft_auxrotor_v1.xml')
    p.add_argument('--output',type=Path,default=A/'vita_six_objects_newft_source_table_v1.xml')
    p.add_argument('--initial-support-buffer-m',type=float,default=.015)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new model path')
    evidence=np.load(E/'table_surface_evidence.npz',allow_pickle=False)
    points=evidence['patch_vertices'];normal=evidence['plane_normal'];origin=evidence['plane_origin']
    normal/=np.linalg.norm(normal);plane_d=-float(normal@origin)
    u=np.asarray([1.,0.,-normal[0]/normal[2]]);u/=np.linalg.norm(u)
    v=np.cross(normal,u);basis=np.column_stack([u,v,normal])
    raw=np.load(D/'objects/poses_sim.npz',allow_pickle=False)
    asset_report=json.loads((A/'assets_report.json').read_text())
    support=[];initial_priors=[]
    for oi in range(6):
        position=raw['position_m'][0,oi].astype(float)
        position=position-normal*(normal@position+plane_d)
        radius=float(asset_report['objects'][oi]['collision']['base_radius_m'])+a.initial_support_buffer_m
        theta=np.linspace(0,2*np.pi,96,endpoint=False)
        ring=position+radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
        support.append(ring)
        initial_priors.append({'object_id':oi+1,'source_frame':0,'base_collision_radius_m':radius-a.initial_support_buffer_m,
                               'support_buffer_m':a.initial_support_buffer_m,'projected_bottom_center_world_m':position.tolist()})
    union=np.vstack([points,*support]);hull=ConvexHull(union[:,:2]);polygon=union[hull.vertices,:2]
    top=np.column_stack([polygon,(-plane_d-polygon@normal[:2])/normal[2]])
    centre=top.mean(axis=0);local=(top-centre)@basis
    vertices=np.vstack([local,local-np.asarray([0.,0.,.05])])
    quat=np.empty(4);mj.mju_mat2Quat(quat,basis.reshape(-1))
    xml=ET.parse(a.base_model).getroot();world=xml.find('worldbody');asset=xml.find('asset')
    table=world.find("geom[@name='table_support']")
    if table is None:raise ValueError('Expected existing table_support')
    old_table_attrs=dict(table.attrib)
    ET.SubElement(asset,'mesh',name='source_visible_table_convex_prism',vertex=fmt(vertices))
    table.set('type','mesh');table.set('mesh','source_visible_table_convex_prism')
    table.set('pos',fmt(centre));table.set('quat',fmt(quat));table.attrib.pop('size',None);table.set('density','0')
    ET.SubElement(world,'site',name='source_table_plane_frame',pos=fmt(centre),quat=fmt(quat),size='.001',rgba='0 0 0 0')
    custom=xml.find('custom')
    if custom is None:custom=ET.SubElement(xml,'custom')
    ET.SubElement(custom,'numeric',name='source_table_plane_abcd',data=fmt([*normal,plane_d]))
    a.output.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
    old=mj.MjModel.from_xml_path(str(a.base_model));new=mj.MjModel.from_xml_path(str(a.output))
    if(new.nq,new.nv,new.nu)!=(old.nq,old.nv,old.nu):raise ValueError('State dimensions changed')
    unchanged=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
               'jnt_type','jnt_pos','jnt_axis','jnt_range','dof_armature','dof_damping','dof_frictionloss',
               'actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_friction','geom_contype','geom_conaffinity']
    if any(not np.array_equal(getattr(old,k),getattr(new,k))for k in unchanged):raise ValueError('Non-table physical parameters changed')
    table_id=mj.mj_name2id(old,mj.mjtObj.mjOBJ_GEOM,'table_support')
    other=np.asarray([g for g in range(old.ngeom)if g!=table_id])
    for key in ['geom_type','geom_size','geom_pos','geom_quat','geom_bodyid']:
        if not np.array_equal(getattr(old,key)[other],getattr(new,key)[other]):raise ValueError('Non-table geometry changed '+key)
    # Validate source initial support with objects only, excluding an unrelated
    # uncontrolled robot free-fall from this isolated table-asset probe.
    probe=copy.deepcopy(xml);pw=probe.find('worldbody')
    for body in list(pw.findall('body')):
        if not body.get('name','').startswith('target_object_'):pw.remove(body)
    act=probe.find('actuator')
    if act is not None:probe.remove(act)
    assetp=probe.find('asset')
    for entry in list(assetp):
        if not entry.get('name','').startswith('target_object_')and entry.get('name')!='source_visible_table_convex_prism':assetp.remove(entry)
    probe_path=a.output.with_name(a.output.stem+'_objects_support_probe.xml')
    probe_path.write_text(ET.tostring(probe,encoding='unicode'),encoding='utf-8')
    m=mj.MjModel.from_xml_path(str(probe_path));data=mj.MjData(m);mj.mj_forward(m,data)
    contact=np.zeros((400,6),bool);max_pen=0.
    bids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,f'target_object_{i:02d}')for i in range(1,7)]
    for step in range(400):
        mj.mj_step(m,data)
        for ci in range(data.ncon):
            c=data.contact[ci];b=[int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])]
            gn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or''for g in [c.geom1,c.geom2]]
            max_pen=max(max_pen,-float(c.dist))
            if 'table_support'in gn:
                for oi,body in enumerate(bids):
                    if body in b:contact[step,oi]=True
    probes=[]
    for oi,body in enumerate(bids):
        j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{oi+1:02d}_free');q=int(m.jnt_qposadr[j]);vv=int(m.jnt_dofadr[j])
        residual=float(normal@data.qpos[q:q+3]+plane_d)
        speed=float(np.linalg.norm(data.qvel[vv:vv+3]));fraction=float(contact[-30:,oi].mean())
        passed=bool(abs(residual)<.025 and speed<.05 and fraction>=.5)
        probes.append({'object_id':oi+1,'final_bottom_world_m':data.qpos[q:q+3].tolist(),
                       'bottom_plane_residual_m':residual,'final_linear_speed_m_s':speed,
                       'final_0p15s_table_contact_fraction':fraction,'passed':passed})
    report={'schema':'fourth.source_visible_minimal_convex_support_completion.v1','model':str(a.output),'model_sha256':sha(a.output),
            'base_model':str(a.base_model),'base_model_sha256':sha(a.base_model),
            'source_evidence_sha256':sha(E/'table_surface_evidence.npz'),'source_table_audit_sha256':sha(E/'report.json'),
            'source_mesh_sha256':sha(D/'scene/visual_mesh.npz'),'source_initial_object_pose_sha256':sha(D/'objects/poses_sim.npz'),
            'boundary_uniquely_measured':False,'hypothesis':'Minimal convex completion of measured connected tabletop points and six initial stationary base-support priors.',
            'held_or_lifted_trajectories_used_to_enlarge_support':False,
            'plane_world_abcd':[float(x)for x in [*normal,plane_d]],'plane_frame_site':'source_table_plane_frame',
            'plane_custom_numeric':'source_table_plane_abcd','polygon_xy':polygon.tolist(),'thickness_m':.05,
            'measured_top_vertices_preserved':len(points),'initial_support_priors':initial_priors,
            'area_m2':float(hull.volume),'old_table_geometry':old_table_attrs,
            'x_intervals_at_world_y':{str(y):edge_x(polygon,y)for y in [1.47,1.60,1.70,1.80,1.90,2.,2.10]},
            'non_table_physical_parameters_exactly_preserved':unchanged,'nq':new.nq,'nv':new.nv,'nu':new.nu,
            'source_initial_six_objects_support_probe':{'duration_s':2.,'steps':400,'all_passed':all(x['passed']for x in probes),'objects':probes,'max_penetration_m':max_pen},
            'limitations':['No physical side edge is established by the sparse view-constrained mesh or raw cloud; this collider is a geometric completion hypothesis, not a measured true table boundary.',
                           'Convex completion fills unknown occluded regions; uncertainty should remain in downstream labels and old-box/new-completion comparisons.',
                           'The initial-support buffer is an explicit 1.5cm prior for six initially stationary objects, not measured furniture extent.',
                           'Table-only support probe does not prove body tracking, grasp or full-sequence interaction success.']}
    a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(a.output.with_suffix('.evidence.npz'),measured_top_vertices=points,initial_support_ring_points=np.stack(support),polygon_xy=polygon,plane_world_abcd=np.r_[normal,plane_d])
    print(json.dumps({k:report[k]for k in ['model','model_sha256','boundary_uniquely_measured','area_m2','plane_world_abcd','x_intervals_at_world_y','source_initial_six_objects_support_probe']}),flush=True)
    if not report['source_initial_six_objects_support_probe']['all_passed']:raise SystemExit(3)


if __name__=='__main__':main()
