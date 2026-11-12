#!/usr/bin/env python3
"""Render an auditable public-G1 candidate scene; not the paper's ROHand assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/video_generation/candidate_scene")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run on the server; the local workstation is for image/video preview only")
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    import trimesh
    import torch
    from PIL import Image
    from pytorch3d.renderer import (
        BlendParams, HardPhongShader, MeshRasterizer, PerspectiveCameras,
        PointLights, RasterizationSettings, TexturesVertex, look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda:0")
    urdf = ROOT / "dependencies/unitree-ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
    tree = ET.parse(urdf).getroot()
    links = {el.attrib["name"]: el for el in tree.findall("link")}
    joints = list(tree.findall("joint"))
    q = {j.attrib["name"]: 0. for j in joints if j.attrib["type"] != "fixed"}
    limits = {}
    for joint in joints:
        if joint.attrib["name"] in q:
            limit = joint.find("limit")
            limits[joint.attrib["name"]] = [float(limit.attrib.get("lower", -3.14)), float(limit.attrib.get("upper", 3.14))]
    for side in ("left", "right"):
        q.update({f"{side}_hip_pitch_joint": -.10, f"{side}_knee_joint": .20, f"{side}_ankle_pitch_joint": -.10})
        q.update({f"{side}_shoulder_pitch_joint": -.45, f"{side}_elbow_joint": .80})

    def origin(element):
        matrix = np.eye(4)
        if element is not None:
            matrix[:3, 3] = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ")
            matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        return matrix

    def fk(values, base_z):
        result = {"pelvis": np.eye(4)}
        result["pelvis"][2, 3] = base_z
        pending = list(joints)
        while pending:
            progressed = False
            for joint in list(pending):
                parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
                if parent not in result:
                    continue
                transform = origin(joint.find("origin"))
                if joint.attrib["type"] != "fixed":
                    axis = np.fromstring(joint.find("axis").attrib.get("xyz", "1 0 0"), sep=" ")
                    motion = np.eye(4)
                    if joint.attrib["type"] in ("revolute", "continuous"):
                        motion[:3, :3] = Rotation.from_rotvec(axis * values[joint.attrib["name"]]).as_matrix()
                    elif joint.attrib["type"] == "prismatic":
                        motion[:3, 3] = axis * values[joint.attrib["name"]]
                    else:
                        raise ValueError("Unsupported URDF joint type")
                    transform = transform @ motion
                result[child] = result[parent] @ transform
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise RuntimeError("URDF is not a connected pelvis-rooted tree")
        return result

    materials = {el.attrib["name"]: list(map(float, el.find("color").attrib["rgba"].split()))[:3] for el in tree.findall("material")}
    visuals = []
    asset_hashes = []
    for name, link in links.items():
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                raise ValueError("Candidate renderer requires explicit mesh visuals")
            path = urdf.parent / mesh_element.attrib["filename"]
            mesh = trimesh.load_mesh(path, process=False)
            scale = np.fromstring(mesh_element.attrib.get("scale", "1 1 1"), sep=" ")
            mesh.vertices *= scale
            color_element = visual.find("material")
            color = materials.get(color_element.attrib.get("name", ""), [.65, .65, .65]) if color_element is not None else [.65, .65, .65]
            if color_element is not None and color_element.find("color") is not None:
                color = list(map(float, color_element.find("color").attrib["rgba"].split()))[:3]
            visuals.append((name, mesh, origin(visual.find("origin")), color))
            asset_hashes.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})

    transforms = fk(q, 0.)
    minimum_z = min(float(trimesh.transform_points(mesh.vertices, transforms[name] @ local)[:, 2].min()) for name, mesh, local, color in visuals)
    base_z = -minimum_z + .002
    ik_checks = []
    # An approach pose with deliberately visible gaps at the box sides.
    # Position IK authors q0 only. It is not a grasp/contact or dynamics solver.
    for side, sign in (("left", 1), ("right", -1)):
        arm = [f"{side}_{name}_joint" for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        target = np.asarray([.41, sign * .245, .90])
        initial = np.asarray([q[name] for name in arm])
        palm_name = f"{side}_hand_palm_link"
        palm_local = np.asarray([.06, 0., 0., 1.])

        def residual(angles):
            trial = dict(q)
            trial.update(zip(arm, angles))
            world = fk(trial, base_z)
            palm_position = (world[palm_name] @ palm_local)[:3]
            return np.concatenate(((palm_position - target) * 10., (angles - initial) * .015))

        solution = least_squares(residual, initial, bounds=(np.asarray([limits[n][0] for n in arm]), np.asarray([limits[n][1] for n in arm])), max_nfev=100)
        q.update(zip(arm, map(float, solution.x)))
        achieved = (fk(q, base_z)[palm_name] @ palm_local)[:3]
        ik_checks.append({"side": side, "target_palm_point_xyz_m": target.tolist(), "palm_point_local_xyz_m": palm_local[:3].tolist(), "achieved_xyz_m": achieved.tolist(), "position_error_m": float(np.linalg.norm(achieved-target))})

    transforms = fk(q, base_z)
    geometry = []
    for name, mesh, local, color in visuals:
        item = mesh.copy()
        item.apply_transform(transforms[name] @ local)
        geometry.append((item, color))

    def add_box(size, center, color):
        item = trimesh.creation.box(extents=size)
        item.unmerge_vertices()
        item.apply_translation(center)
        geometry.append((item, color))

    box_size, box_center = [.28, .34, .24], [.41, 0., .90]
    add_box([4., 4., .04], [0., 0., -.02], [.78, .81, .84])
    for grid_coordinate in np.arange(-1.5, 1.501, .25):
        add_box([3., .002, .0005], [0., grid_coordinate, .00025], [.67, .71, .75])
        add_box([.002, 3., .0005], [grid_coordinate, 0., .00025], [.67, .71, .75])
    add_box([.70, .85, .045], [.48, 0., .7575], [.40, .30, .21])
    for x in (.19, .77):
        for y in (-.355, .355):
            add_box([.035, .035, .735], [x, y, .3675], [.25, .28, .31])
    add_box(box_size, box_center, [.68, .43, .20])
    add_box([.282, .035, .001], [.41, 0., 1.0208], [.31, .22, .13])
    add_box([.001, .035, .24], [.2692, 0., .90], [.31, .22, .13])

    vertices, faces, colors, offset = [], [], [], 0
    for mesh, color in geometry:
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        colors.append(np.tile(np.asarray(color), (len(mesh.vertices), 1)))
        offset += len(mesh.vertices)
    mesh = Meshes(verts=[torch.tensor(np.concatenate(vertices), device=device, dtype=torch.float32)], faces=[torch.tensor(np.concatenate(faces), device=device, dtype=torch.int64)], textures=TexturesVertex(verts_features=[torch.tensor(np.concatenate(colors),device=device,dtype=torch.float32)]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images, camera_records = [], []
    for name, eye in (("left", [2.35, -1.90, 1.52]), ("right", [2.35, 1.90, 1.52])):
        at, up = [.26, 0., .72], [0., 0., 1.]
        R, T = look_at_view_transform(eye=[eye], at=[at], up=[up], device=device)
        camera = PerspectiveCameras(device=device, focal_length=((950., 950.),), principal_point=((320., 360.),), image_size=((720, 640),), in_ndc=False, R=R, T=T)
        rasterizer = MeshRasterizer(cameras=camera, raster_settings=RasterizationSettings(image_size=(720,640), blur_radius=0., faces_per_pixel=1, bin_size=64, max_faces_per_bin=200000, z_clip_value=.05, cull_to_frustum=True))
        shader = HardPhongShader(device=device,cameras=camera,lights=PointLights(device=device, location=((2., -2., 4.),),ambient_color=((.52,.52,.52),),diffuse_color=((.60,.60,.60),),specular_color=((.05,.05,.05),)),blend_params=BlendParams(background_color=(.95,.96,.97)))
        with torch.inference_mode():
            fragments = rasterizer(mesh)
            rgba = shader(fragments, mesh)
        image = Image.fromarray((rgba[0,:,:,:3].clamp(0,1).cpu().numpy()*255).astype(np.uint8))
        image.save(args.output_dir / f"view_{name}.png")
        images.append(image)
        np.save(args.output_dir / f"view_{name}_camera_depth_m.npy", fragments.zbuf[0,:,:,0].cpu().numpy())
        camera_records.append({"name":name,"image_size_hw":[720,640],"focal_length_xy_px":[950.,950.],"principal_point_xy_px":[320.,360.],"eye_world_xyz_m":eye,"at_world_xyz_m":at,"up_world":up,"R_pytorch3d_world_to_view_row_vectors":R[0].cpu().tolist(),"T_pytorch3d_world_to_view_row_vectors":T[0].cpu().tolist(),"K_pytorch3d_4x4":camera.get_projection_transform().get_matrix()[0].cpu().tolist(),"world_to_view_row_matrix4x4":camera.get_world_to_view_transform().get_matrix()[0].cpu().tolist(),"pixel_slice_in_combined_xywh":[0 if name=='left' else 640,0,640,720]})
    combined = Image.new("RGB", (1280,720))
    combined.paste(images[0], (0,0)); combined.paste(images[1], (640,0))
    combined.save(args.output_dir / "initial_two_view.png")
    metadata = {
        "scene_id":"candidate_scene_g1_stock_three_finger_box_approach",
        "status":"candidate_geometric_render_for_review",
        "paper_exact_scene":False,
        "purpose":"First-frame reference for two-view table-box lifting video test",
        "renderer":"PyTorch3D triangle rasterization of public Unitree URDF visuals; no AI image generation",
        "coordinate_system":"URDF world: metres; Z up; X robot forward; Y robot left",
        "robot":{"urdf":str(urdf.relative_to(ROOT)),"sha256":digest(urdf),"source_repository":"unitreerobotics/unitree_ros","source_commit":"5994d4faef0a9cadd3287f8de0199a67eeb2a259","model":"g1_29dof_with_hand_rev_1_0","hands":"Public URDF integrated three-finger hand, 7 revolute joints per hand; not ROHand","rohand_assembly_performed":False,"base_position_xyz_m":[0.,0.,base_z],"base_quaternion_xyzw":[0.,0.,0.,1.],"joint_positions_rad":q,"joint_limits_rad":limits,"all_joint_limits_satisfied":all(limits[name][0]<=value<=limits[name][1] for name,value in q.items()),"link_world_transforms":{name:matrix.tolist() for name,matrix in transforms.items()},"visual_mesh_sha256":asset_hashes},
        "box":{"size_xyz_m":box_size,"center_xyz_m":box_center,"quaternion_xyzw":[0.,0.,0.,1.],"appearance":"brown cardboard with narrow dark tape strip","resting_on_table":True},
        "table":{"top_size_xyz_m":[.70,.85,.045],"top_center_xyz_m":[.48,0.,.7575],"top_surface_z_m":.78},
        "floor":{"surface_z_m":0.,"size_xy_m":[4.,4.],"visible_grid_spacing_m":.25,"foot_lowest_visual_clearance_m":.002},
        "cameras":camera_records,"combined_image_size_wh":[1280,720],"camera_model":"Pinhole, no lens distortion. R/T follow documented PyTorch3D row-vector convention; +X camera left, +Y camera up, +Z into scene.",
        "pose_authoring":{"method":"URDF FK and bounded arm position IK for an approach pose","palm_targets":ik_checks},
        "limitations":["Uses public G1 integrated three-finger hands, not the KinoGen G1-ROHand assembly","Candidate camera placement and object geometry are authored here; paper calibration/assets unavailable","Approach pose only; fingers are not wrapped around box","No grasp, collision-free, contact-force, balance or dynamics validation","Geometric reference only; generated-video motion consistency must be checked separately"],
        "gpu":torch.cuda.get_device_name(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"mesh_vertices":int(offset),"mesh_triangles":int(sum(len(item[0].faces) for item in geometry)),
    }
    metadata["output_files"]=[{"name":p.name,"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.suffix in (".png",".npy")]
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    (args.output_dir / "scene.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    report = ROOT / "reports/candidate-scene.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":metadata['status'],"output_dir":str(args.output_dir),"hands":metadata['robot']['hands'],"mesh_triangles":metadata['mesh_triangles'],"pose_checks":ik_checks},indent=2),flush=True)


if __name__ == "__main__":
    main()
