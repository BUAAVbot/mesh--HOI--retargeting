#!/usr/bin/env python3
"""CPU reference geometry and measured image/camera consistency for a user video.

The generated video is not a calibrated camera capture. Geometry is a reference
prior; image registration and physical camera validity are reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "data/video_generation/candidate_scene")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/user_reconstruction")
    parser.add_argument("--points", type=int, default=650)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    import cv2
    import numpy as np
    import open3d as o3d
    import trimesh
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.scene_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    reference_path = args.scene_dir / "initial_two_view.png"
    reference = cv2.imread(str(reference_path))
    video_path = args.scene_dir / "generated.mp4"
    capture = cv2.VideoCapture(str(video_path))
    frame_count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    ok, first = capture.read()
    if reference is None or not ok or first.shape != reference.shape:
        raise RuntimeError("Cannot compare same-size reference and video first frame")
    frame_indices = np.unique(np.linspace(0, frame_count-1, 5).round().astype(int))
    frames = {}
    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok: frames[int(index)] = frame
    capture.release()

    urdf = ROOT / scene["robot"]["urdf"]
    if sha(urdf) != scene["robot"]["sha256"]:
        raise RuntimeError("URDF hash does not match authored scene")
    xml = ET.parse(urdf).getroot()
    transforms = {name:np.asarray(matrix,dtype=np.float64) for name,matrix in scene["robot"]["link_world_transforms"].items()}
    # Verify all transforms from recorded q0 by independent URDF FK.
    def origin(el):
        m=np.eye(4)
        if el is not None:
            m[:3,3]=np.fromstring(el.attrib.get("xyz","0 0 0"),sep=" ")
            m[:3,:3]=Rotation.from_euler("xyz",np.fromstring(el.attrib.get("rpy","0 0 0"),sep=" ")).as_matrix()
        return m
    fk={"pelvis":np.eye(4)}
    fk["pelvis"][:3,3]=scene["robot"]["base_position_xyz_m"]
    fk["pelvis"][:3,:3]=Rotation.from_quat(scene["robot"]["base_quaternion_xyzw"]).as_matrix()
    pending=list(xml.findall("joint"))
    while pending:
        done=[]
        for j in pending:
            parent,child=j.find("parent").attrib["link"],j.find("child").attrib["link"]
            if parent not in fk: continue
            motion=np.eye(4)
            if j.attrib["type"]!="fixed":
                if j.attrib["type"]!="revolute": raise RuntimeError("Unexpected joint type")
                motion[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find("axis").attrib["xyz"],sep=" ")*scene["robot"]["joint_positions_rad"][j.attrib["name"]]).as_matrix()
            fk[child]=fk[parent]@origin(j.find("origin"))@motion
            done.append(j)
        if not done: raise RuntimeError("Unconnected URDF")
        pending=[j for j in pending if j not in done]
    fk_error=max(float(np.abs(fk[name]-transforms[name]).max()) for name in transforms)
    if fk_error>1e-8: raise RuntimeError("Recorded q0 and link transforms disagree")
    mesh_hash_lookup={record["path"]:record["sha256"] for record in scene["robot"]["visual_mesh_sha256"]}
    vertices=[]; faces=[]; face_links=[]; face_classes=[]; names=[]; offset=0
    def add(mesh, tag, link=-1):
        nonlocal offset
        vertices.append(np.asarray(mesh.vertices)); faces.append(np.asarray(mesh.faces)+offset)
        face_links.append(np.full(len(mesh.faces),link,dtype=np.int32));face_classes.append(np.full(len(mesh.faces),tag,dtype=np.int32));offset+=len(mesh.vertices)
    for link in xml.findall("link"):
        name=link.attrib["name"]; names.append(name); link_index=len(names)-1
        for visual in link.findall("visual"):
            mesh_el=visual.find("geometry/mesh")
            if mesh_el is None: continue
            path=urdf.parent / mesh_el.attrib["filename"]
            relative=path.relative_to(ROOT).as_posix()
            if mesh_hash_lookup.get(relative)!=sha(path): raise RuntimeError("Visual mesh hash mismatch: "+relative)
            mesh=trimesh.load_mesh(path,process=False)
            mesh.vertices*=np.fromstring(mesh_el.attrib.get("scale","1 1 1"),sep=" ")
            mesh.apply_transform(transforms[name]@origin(visual.find("origin")))
            add(mesh,0,link_index)
    def box(size,center,tag):
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);add(mesh,tag)
    box(scene["box"]["size_xyz_m"],scene["box"]["center_xyz_m"],1)
    box(scene["table"]["top_size_xyz_m"],scene["table"]["top_center_xyz_m"],2)
    for x in (.19,.77):
        for y in (-.355,.355): box([.035,.035,.735],[x,y,.3675],2)
    floor_size=scene.get("floor",{}).get("size_xy_m",[4.,4.])
    box([*floor_size,.04],[0,0,-.02],3)
    v=np.concatenate(vertices).astype(np.float32); f=np.concatenate(faces).astype(np.uint32)
    face_link=np.concatenate(face_links);face_class=np.concatenate(face_classes)
    ray_scene=o3d.t.geometry.RaycastingScene(nthreads=6)
    ray_scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    box_corners=(np.asarray([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*np.asarray(scene["box"]["size_xyz_m"]))@Rotation.from_quat(scene["box"]["quaternion_xyzw"]).as_matrix().T+np.asarray(scene["box"]["center_xyz_m"])
    arrays={"box_corners_world":box_corners.astype(np.float32),"link_names":np.asarray(names),"q0_joint_names":np.asarray(list(scene["robot"]["joint_positions_rad"])),"q0_joint_positions_rad":np.asarray(list(scene["robot"]["joint_positions_rad"].values())),"sampled_frame_indices":frame_indices}
    fields={key:[] for key in ("camera_K","camera_R_wc","camera_t_wc","camera_P","P_video","robot_points_world","robot_points_pixels_xy","robot_points_video_initial_xy","robot_points_video_query_valid","robot_points_link_index","robot_points_initial_alignment_valid","robot_points_lk_error_px","robot_points_camera_depth_z_m","robot_points_triangle_index","robot_points_triangle_uv","box_corners_pixels_xy","box_corners_video_pixels_xy","static_pixels_xy","static_video_pixels_xy","static_camera_depth_z_m","static_points_world","static_class","image_homography_ref_to_video","reference_robot_mask","reference_box_mask","reference_static_mask","reference_depth_z_m","camera_affines_first_to_sampled_frame","calibration_valid")}
    reports=[]; rng=np.random.default_rng(20261020)

    def project(points,K,R,t):
        xyz=points@R.T+t
        uv=xyz@K.T
        return uv[:,:2]/uv[:,2:],xyz[:,2]
    def warp_points(points,H):
        p=np.concatenate((points,np.ones((len(points),1))),axis=1)@H.T
        return p[:,:2]/p[:,2:]
    def choose_pixels(mask,count,link_map=None):
        yy,xx=np.nonzero(mask)
        if len(xx)<count: raise RuntimeError("Too few visible mesh points")
        candidates=np.column_stack((xx,yy))
        # Even spatial coverage; then fill per-link details through a deterministic shuffle.
        permutation=rng.permutation(len(candidates)); candidates=candidates[permutation]
        selected=[]; selected_set=set()
        for step in (14,9,5,2,1):
            used=set((x//step,y//step) for x,y in selected)
            for x,y in candidates:
                key=(int(x)//step,int(y)//step)
                point=(int(x),int(y))
                if key not in used and point not in selected_set:
                    selected.append(point);used.add(key);selected_set.add(point)
                    if len(selected)==count:return np.asarray(selected,dtype=np.int32)
        return np.asarray(selected[:count],dtype=np.int32)

    for view,cam in enumerate(scene["cameras"]):
        h,w=cam["image_size_hw"]; x0=cam["pixel_slice_in_combined_xywh"][0]
        ref=reference[:,x0:x0+w]; current=first[:,x0:x0+w]
        gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY); target=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
        fx,fy=cam["focal_length_xy_px"];cx,cy=cam["principal_point_xy_px"]
        K=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
        D=np.diag([-1.,-1.,1.]);R=D@np.asarray(cam["R_pytorch3d_world_to_view_row_vectors"]).T;t=D@np.asarray(cam["T_pytorch3d_world_to_view_row_vectors"])
        P=K@np.column_stack((R,t));center=-R.T@t
        xx,yy=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
        directions=np.stack(((xx-cx)/fx,(yy-cy)/fy,np.ones_like(xx)),axis=-1)@R
        rays=np.concatenate((np.broadcast_to(center,directions.shape),directions),axis=-1).astype(np.float32)
        answer=ray_scene.cast_rays(o3d.core.Tensor(rays),nthreads=6)
        depth=answer["t_hit"].numpy();primitive=answer["primitive_ids"].numpy().astype(np.int64)
        valid=np.isfinite(depth);classes=np.full((h,w),-1,dtype=np.int32);linkmap=np.full((h,w),-1,dtype=np.int32)
        classes[valid]=face_class[primitive[valid]];linkmap[valid]=face_link[primitive[valid]]
        robot=classes==0;boxmask=classes==1;static=(classes==2)|(classes==3)
        reference_edges=cv2.Canny(gray,35,95)
        edge_distance=cv2.distanceTransform((reference_edges==0).astype(np.uint8),cv2.DIST_L2,3)
        robot_boundary=(cv2.morphologyEx(robot.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0)
        boundary_error=edge_distance[robot_boundary]
        silhouette_median=float(np.median(boundary_error));silhouette_p95=float(np.percentile(boundary_error,95))
        reference_geometry_valid=silhouette_median<=2. and silhouette_p95<=6.
        # Static registration excludes all robot/box pixels and their border.
        excluded=cv2.dilate(((robot|boxmask)*255).astype(np.uint8),np.ones((19,19),np.uint8))>0
        static_mask=((static&~excluded)*255).astype(np.uint8);static_mask[:130]=0
        sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.015)
        a,da=sift.detectAndCompute(gray,static_mask);b,db=sift.detectAndCompute(target,None)
        good=[]
        if da is not None and db is not None:
            good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.70*n.distance]
        src=np.asarray([a[m.queryIdx].pt for m in good],dtype=np.float32).reshape(-1,2);dst=np.asarray([b[m.trainIdx].pt for m in good],dtype=np.float32).reshape(-1,2)
        H=np.eye(3);inlier=np.zeros(len(good),dtype=bool)
        if len(good)>=4:
            affine,keep=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=6000,confidence=.999)
            if affine is not None:H[:2]=affine;inlier=keep[:,0].astype(bool)
        residual=np.linalg.norm(warp_points(src,H)-dst,axis=1) if len(src) else np.asarray([])
        identity_residual=np.linalg.norm(src-dst,axis=1) if len(src) else np.asarray([])
        n_inliers=int(inlier.sum());scale=float(np.sqrt(np.linalg.det(H[:2,:2])));rotation=float(np.rad2deg(np.arctan2(H[1,0],H[0,0])))
        median=float(np.median(residual[inlier])) if n_inliers else None;p95=float(np.percentile(residual[inlier],95)) if n_inliers else None
        registration_valid=bool(n_inliers>=12 and n_inliers/max(len(good),1)>=.4 and median<=2. and p95<=4.)
        # This is conditional compatibility, not recovery of camera truth.
        calibration_valid=bool(reference_geometry_valid and registration_valid and abs(scale-1)<.02 and abs(rotation)<1 and np.linalg.norm(H[:2,2])<12)
        eroded=cv2.erode(robot.astype(np.uint8),np.ones((3,3),np.uint8))>0
        seed=choose_pixels(eroded,args.points);sx,sy=seed[:,0],seed[:,1]
        reference_uv=seed.astype(np.float64)+.5
        world=center+directions[sy,sx]*depth[sy,sx,None]
        projected,pointdepth=project(world,K,R,t)
        max_projection_error=float(np.linalg.norm(projected-reference_uv,axis=1).max())
        mapped=warp_points(reference_uv,H).astype(np.float32)
        # Reference-to-video LK measures robot pose changes independently from the background H.
        tracked,status,lkerr=cv2.calcOpticalFlowPyrLK(gray,target,reference_uv.astype(np.float32).reshape(-1,1,2),mapped.reshape(-1,1,2).copy(),winSize=(19,19),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back,backstatus,_=cv2.calcOpticalFlowPyrLK(target,gray,tracked,None,winSize=(19,19),maxLevel=4)
        tracked=tracked[:,0];fb=np.linalg.norm(back[:,0]-reference_uv,axis=1);flow=np.linalg.norm(tracked-mapped,axis=1)
        correspondence=(status[:,0]>0)&(backstatus[:,0]>0)&(fb<2.)&(lkerr[:,0]<30)&(tracked[:,0]>2)&(tracked[:,0]<w-3)&(tracked[:,1]>2)&(tracked[:,1]<h-3)
        anchor=correspondence&(flow<3.)&calibration_valid
        video_seed=mapped.copy();video_seed[correspondence]=tracked[correspondence]
        # Only stable, texture-detectable ray hits are depth-scale calibration candidates.
        # The original large-floor raster has clipped near-white background
        # regions. A ray intersection alone does not make those pixels visible
        # metric observations; require a solid rendered surface interior.
        rendered_surface=cv2.erode((gray<230).astype(np.uint8),np.ones((5,5),np.uint8))>0
        static_candidate=static & ~excluded & (depth>.1)&(depth<12.) & rendered_surface
        static_candidate[:140]=False
        static_seed=choose_pixels(static_candidate,800);px,py=static_seed[:,0],static_seed[:,1]
        static_uv=static_seed.astype(np.float64)+.5
        static_world=center+directions[py,px]*depth[py,px,None]
        boxuv,_=project(box_corners,K,R,t)
        sampled_affines=[];drift=[]
        static_features=cv2.goodFeaturesToTrack(target,maxCorners=400,qualityLevel=.01,minDistance=9,mask=static_mask)
        for index in frame_indices:
            ht=np.eye(3)
            if static_features is not None and int(index) in frames:
                tg=cv2.cvtColor(frames[int(index)][:,x0:x0+w],cv2.COLOR_BGR2GRAY)
                flowpoints,goodstatus,_=cv2.calcOpticalFlowPyrLK(target,tg,static_features,None,winSize=(21,21),maxLevel=4)
                aa=static_features[:,0][goodstatus[:,0]>0];bb=flowpoints[:,0][goodstatus[:,0]>0]
                if len(aa)>10:
                    aff,ii=cv2.estimateAffinePartial2D(aa,bb,method=cv2.RANSAC,ransacReprojThreshold=2.5)
                    if aff is not None:
                        ht[:2]=aff;keep=ii[:,0]>0
                        drift.append({"frame":int(index),"inliers":int(keep.sum()),"median_static_motion_px":float(np.median(np.linalg.norm(bb[keep]-aa[keep],axis=1))),"median_fit_error_px":float(np.median(np.linalg.norm(warp_points(aa[keep],ht)-bb[keep],axis=1)))})
            sampled_affines.append(ht)
        # Visual diagnostic, with each reference point painted on its measured video correspondence.
        overlay=current.copy()
        for point,accept in zip(video_seed,anchor):cv2.circle(overlay,tuple(np.rint(point).astype(int)),1,(0,220,0) if accept else (0,100,255),-1)
        cv2.imwrite(str(args.output_dir/f"geometry_points_view{view}.jpg"),overlay)
        cv2.imwrite(str(args.output_dir/f"geometry_reference_mask_view{view}.png"),(robot*255).astype(np.uint8))
        per_link=[]
        for linkid in np.unique(linkmap[sy,sx]):
            m=linkmap[sy,sx]==linkid
            per_link.append({"link":names[int(linkid)],"points":int(m.sum()),"stable_anchor_count":int(anchor[m].sum()),"median_reference_to_video_nonrigid_displacement_px":float(np.median(flow[m]))})
        reports.append({"view":view,"name":cam["name"],"focal_length_xy_px":[fx,fy],"static_feature_matches":len(good),"static_inliers":n_inliers,"static_inlier_ratio":n_inliers/max(len(good),1),"first_frame_median_reprojection_error_px":median,"first_frame_p95_reprojection_error_px":p95,"first_frame_identity_median_error_px":float(np.median(identity_residual[inlier])) if n_inliers else None,"similarity_scale":scale,"rotation_degrees":rotation,"translation_xy_px":H[:2,2].tolist(),"background_registration_valid":registration_valid,"calibration_conditionally_compatible":calibration_valid,"reference_geometry_projection_roundtrip_max_error_px":max_projection_error,"reference_robot_silhouette_edge_median_error_px":silhouette_median,"reference_robot_silhouette_edge_p95_error_px":silhouette_p95,"reference_geometry_model_image_alignment_valid":reference_geometry_valid,"reference_robot_visible_pixels":int(robot.sum()),"robot_points":args.points,"robot_reference_to_video_correspondence_valid":int(correspondence.sum()),"robot_reference_world_anchor_valid":int(anchor.sum()),"robot_q0_not_video_ground_truth":True,"per_link":per_link,"sampled_static_background_drift":drift,"image_transform_ref_to_video":H.tolist(),"K":K.tolist(),"R_wc":R.tolist(),"t_wc":t.tolist(),"P":P.tolist(),"P_video":(H@P).tolist()})
        values={"camera_K":K,"camera_R_wc":R,"camera_t_wc":t,"camera_P":P,"P_video":H@P,"robot_points_world":world,"robot_points_pixels_xy":reference_uv,"robot_points_video_initial_xy":video_seed,"robot_points_video_query_valid":correspondence,"robot_points_link_index":linkmap[sy,sx],"robot_points_initial_alignment_valid":anchor,"robot_points_lk_error_px":fb,"robot_points_camera_depth_z_m":pointdepth,"robot_points_triangle_index":primitive[sy,sx],"robot_points_triangle_uv":answer["primitive_uvs"].numpy()[sy,sx],"box_corners_pixels_xy":boxuv,"box_corners_video_pixels_xy":warp_points(boxuv,H),"static_pixels_xy":static_uv,"static_video_pixels_xy":warp_points(static_uv,H),"static_camera_depth_z_m":depth[py,px],"static_points_world":static_world,"static_class":classes[py,px],"image_homography_ref_to_video":H,"reference_robot_mask":robot,"reference_box_mask":boxmask,"reference_static_mask":static,"reference_depth_z_m":depth,"camera_affines_first_to_sampled_frame":np.asarray(sampled_affines),"calibration_valid":calibration_valid}
        for key,value in values.items():fields[key].append(value)
    arrays.update({key:np.asarray(value) for key,value in fields.items()})
    arrays["static_surface_kind"]=arrays["static_class"]
    arrays["static_valid"]=np.isfinite(arrays["static_camera_depth_z_m"])&arrays["calibration_valid"][:,None]
    arrays.update(K=arrays["camera_K"],R=arrays["camera_R_wc"],t=arrays["camera_t_wc"],P=arrays["camera_P"],video_fps=np.asarray(fps),video_frame_count=np.asarray(frame_count),robot_points_world_are_reference_pose_only=np.asarray(True))
    npz=args.output_dir / "geometry.npz"
    np.savez_compressed(npz,**arrays)
    report={"status":"geometry_prepared_with_measured_conditional_calibration" if bool(arrays['calibration_valid'].all()) else "geometry_prepared_calibration_unverified_or_blocked","geometry_npz":str(npz),"geometry_sha256":sha(npz),"video_sha256":sha(video_path),"reference_image_sha256":sha(reference_path),"scene_json_sha256":sha(scene_path),"urdf_sha256":sha(urdf),"source_model":scene['robot']['model'],"hands":scene['robot']['hands'],"q0_joint_positions_rad":scene['robot']['joint_positions_rad'],"q0_fk_transform_max_error":fk_error,"video_frame_count":frame_count,"video_fps":fps,"camera_convention":"OpenCV column vectors: X_camera=R_wc X_world+t_wc, u right/v down/Z forward. World metres, Z up. PyTorch3D row coordinates Xp=Xworld@Rpy+Tpy become OpenCV via diag(-1,-1,1); R_wc=diag(-1,-1,1)@Rpy.T, t_wc=diag(-1,-1,1)@Tpy.","pixel_convention":"Continuous image coordinates, top-left image corner origin; sampled pixel centres are integer index+0.5. Each view is 640x720; add 640 to right-view u for combined-image coordinates.","depth_convention":"Camera Z in metres, not Euclidean ray length; ray directions are not unit normalized and have camera Z component 1.","image_alignment_policy":"H is measured reference-to-video first-frame 2D similarity, recorded separately from unchanged reference K/R/t. P_video=H@P is a conditional image projection, not proof of a physically calibrated generated video.","known_geometry_policy":"All robot 3D points describe authored reference q0. Video frame0 already changes hands/arms; only points with initial_alignment_valid may serve as approximate static 3D anchors. Valid LK correspondences on other links identify material candidates, not unchanged world positions.","static_depth_policy":"Static points are nearest CPU mesh-ray hits on table/ground only. Their metric Z may calibrate predicted relative depth only where background alignment and static-video checks hold. Appearance changes/artifacts and invented occluded geometry are not evidence of metric accuracy.","views":reports,"array_fields":{key:{"shape":list(value.shape),"dtype":str(value.dtype)} for key,value in arrays.items()},"limitations":["Generated video is not a calibrated sensor recording","Reference q0 differs from video arms/hands at frame0","ROHand paper assembly is unavailable; public G1 three-finger model only","Static background image consistency does not prove exact camera or object geometry","Table/ground depth calibration remains conditional on static geometry preservation"]}
    reportpath=ROOT / "reports/user-geometry.json";reportpath.parent.mkdir(exist_ok=True)
    reportpath.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report['status'],"npz":str(npz),"views":[{key:r[key] for key in ('view','static_feature_matches','static_inliers','first_frame_median_reprojection_error_px','similarity_scale','translation_xy_px','calibration_conditionally_compatible','robot_reference_world_anchor_valid')} for r in reports]},indent=2),flush=True)


if __name__=="__main__":main()
