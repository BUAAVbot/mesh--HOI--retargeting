"""Render/export the optimized articulated URDF and box; audit actual mesh silhouettes."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/user_reconstruction_v2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-test", action="store_true")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/video_generation/candidate_scene/scene_rohand.json")
    parser.add_argument("--assembly-preview", action="store_true")
    parser.add_argument("--keyframe-preview", type=Path, help="Actual sparse hand candidate NPZ; QA only, not a full sequence")
    parser.add_argument("--preview-frames",type=int,nargs='+')
    parser.add_argument("--hand-face-limit",type=int,default=0,help="0 preserves every original hand triangle; positive limits are diagnostic simplifications")
    parser.add_argument("--hand-file", type=Path)
    parser.add_argument("--body-file", type=Path)
    parser.add_argument("--variant", choices=("v4",), default="v4")
    args = parser.parse_args()
    for attribute in ('body_file','hand_file','keyframe_preview'):
        path=getattr(args,attribute)
        if path is not None:setattr(args,attribute,path.resolve()if path.is_absolute()else(ROOT/path).resolve())
    output_dir = ROOT / ("outputs/user_reconstruction_" + args.variant) if args.variant in ("v3", "v4") else OUT
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Use the coordinated physical GPU7 for rendering only")
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    import av
    import cv2
    import numpy as np
    import open3d as o3d
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from scipy.spatial.transform import Rotation
    from user_body_kinematics import G1Kinematics
    cv2.setNumThreads(4)
    torch.cuda.set_per_process_memory_fraction(.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = ROOT / "outputs/previews"
    previews.mkdir(parents=True, exist_ok=True)
    kin = G1Kinematics(ROOT, scene_path=args.scene)
    scene = kin.scene
    # Audit original hand surfaces, independently of the decimated render mesh.
    hand_audit_surfaces = []
    non_hand_audit_surfaces = []
    for visual in kin.visuals:
        if "_hand_ro_" not in visual["link_name"]:
            vertices=np.asarray(visual['vertices_local'],dtype=np.float64)
            faces=np.asarray(visual['faces'],dtype=np.int64)
            non_hand_audit_surfaces.append((visual['link_index'],visual['link_name'],np.unique(vertices,axis=0),vertices[faces].mean(axis=1)))
            continue
        raw_vertices = np.asarray(visual["vertices_local"], dtype=np.float64)
        raw_faces = np.asarray(visual["faces"], dtype=np.int64)
        unique_vertices = np.unique(raw_vertices, axis=0)
        centers = raw_vertices[raw_faces].mean(axis=1)
        hand_audit_surfaces.append((visual["link_index"], unique_vertices, centers))

    def hand_collision_audit(transforms, frame, box_data):
        inverse_rotation = Rotation.from_quat(box_data["quaternion_xyzw"][frame]).as_matrix()
        center = box_data["center_xyz_m"][frame]
        half_size = np.asarray(scene["box"]["size_xyz_m"]) / 2
        minimum_vertex = minimum_center = np.inf
        vertex_count = penetrated_vertices = center_count = penetrated_centers = 0
        for link, raw_vertices, centers in hand_audit_surfaces:
            transform = transforms[link]
            for kind, points in (("vertex", raw_vertices), ("center", centers)):
                world = points @ transform[:3, :3].T + transform[:3, 3]
                local_box = (world - center) @ inverse_rotation
                signed_axes = np.abs(local_box) - half_size
                sdf = np.linalg.norm(np.maximum(signed_axes, 0), axis=1) + np.minimum(signed_axes.max(axis=1), 0)
                if kind == "vertex":
                    minimum_vertex = min(minimum_vertex, float(sdf.min()))
                    vertex_count += len(points)
                    penetrated_vertices += int(np.count_nonzero(sdf < -.005))
                else:
                    minimum_center = min(minimum_center, float(sdf.min()))
                    center_count += len(points)
                    penetrated_centers += int(np.count_nonzero(sdf < -.005))
        return {"hand_vertices_inside_box_deeper_than_5mm_fraction": penetrated_vertices / vertex_count,
                "hand_triangle_centers_inside_box_deeper_than_5mm_fraction": penetrated_centers / center_count,
                "hand_box_max_vertex_penetration_m": float(max(0, -minimum_vertex)),
                "hand_box_max_triangle_center_penetration_m": float(max(0, -minimum_center)),
                "hand_box_max_surface_sample_penetration_m": float(max(0, -min(minimum_vertex, minimum_center))),
                "hand_box_min_surface_sample_signed_distance_m": float(min(minimum_vertex, minimum_center)),
                "hand_audit_original_vertex_count": vertex_count,
                "hand_audit_original_triangle_center_count": center_count}

    def non_hand_box_audit(transforms,frame,box_data):
        rotation=Rotation.from_quat(box_data['quaternion_xyzw'][frame]).as_matrix()
        center=box_data['center_xyz_m'][frame];half=np.asarray(scene['box']['size_xyz_m'])/2
        minimum=np.inf;worst=None
        for link,name,vertices,centers in non_hand_audit_surfaces:
            T=transforms[link];points=np.concatenate((vertices,centers))@T[:3,:3].T+T[:3,3]
            axes=np.abs((points-center)@rotation)-half
            distance=np.linalg.norm(np.maximum(axes,0),axis=1)+np.minimum(axes.max(axis=1),0)
            low=float(distance.min())
            if low<minimum:minimum,worst=low,name
        return {'non_hand_box_max_surface_sample_penetration_m':float(max(0,-minimum)),
                'non_hand_box_min_surface_sample_signed_distance_m':float(minimum),'non_hand_box_nearest_link':worst}
    vertices, faces, normals, colors, vertex_links, face_classes = [], [], [], [], [], []
    offset = 0

    def add(v, f, color, link_id, class_id, target=None):
        nonlocal offset
        mesh = trimesh.Trimesh(vertices=np.asarray(v).copy(), faces=np.asarray(f).copy(), process=False)
        if target and len(mesh.faces) > target:
            simple = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
            simple.remove_duplicated_vertices()
            simple.remove_degenerate_triangles()
            simple = simple.simplify_quadric_decimation(target)
            mesh = trimesh.Trimesh(vertices=np.asarray(simple.vertices), faces=np.asarray(simple.triangles), process=False)
        vertices.append(np.asarray(mesh.vertices)); faces.append(np.asarray(mesh.faces) + offset)
        normals.append(np.asarray(mesh.vertex_normals)); colors.append(np.tile(color, (len(mesh.vertices), 1)))
        vertex_links.append(np.full(len(mesh.vertices), link_id, np.int32))
        face_classes.append(np.full(len(mesh.faces), class_id, np.int32)); offset += len(mesh.vertices)

    for visual in kin.visuals:
        add(visual["vertices_local"], visual["faces"], visual["color"], visual["link_index"], 1,
            target=(args.hand_face_limit or None) if "_hand_ro_" in visual["link_name"] else 1800)
    robot_vertex_count = offset
    robot_face_count = sum(len(x) for x in faces)
    box_link, static_link = len(kin.link_names), len(kin.link_names) + 1

    def primitive(size, center, color, link, kind):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        add(mesh.vertices, mesh.faces, color, link, kind)

    primitive(scene["box"]["size_xyz_m"], [0, 0, 0], [.68, .43, .20], box_link, 2)
    primitive(scene["table"]["top_size_xyz_m"], scene["table"]["top_center_xyz_m"], [.40, .30, .21], static_link, 3)
    for x in (.19, .77):
        for y in (-.355, .355):
            primitive([.035, .035, .735], [x, y, .3675], [.25, .28, .31], static_link, 3)
    primitive([3, 3, .002], [0, 0, -.002], [.83, .85, .87], static_link, 4)
    local = np.concatenate(vertices).astype(np.float32)
    triangles = np.concatenate(faces).astype(np.int32)
    local_normals = np.concatenate(normals).astype(np.float32)
    rgb_colors = np.concatenate(colors).astype(np.float32)
    ids = np.concatenate(vertex_links)
    classes = np.concatenate(face_classes)
    device = "cuda"
    tri = torch.as_tensor(triangles, device=device).contiguous()
    vlocal = torch.as_tensor(local, device=device)
    nlocal = torch.as_tensor(local_normals, device=device)
    v_ids = torch.as_tensor(ids.astype(np.int64), device=device)
    v_color = torch.as_tensor(rgb_colors, device=device)
    class_gpu = torch.as_tensor(classes, device=device)
    ctx = dr.RasterizeCudaContext()
    flip = True

    def transformed(transforms):
        mats = torch.as_tensor(transforms, dtype=torch.float32, device=device)[v_ids]
        world = torch.einsum("nij,nj->ni", mats[:, :3, :3], vlocal) + mats[:, :3, 3]
        normal = torch.einsum("nij,nj->ni", mats[:, :3, :3], nlocal)
        light = torch.tensor([.4, -.5, .76], dtype=torch.float32, device=device)
        shade = .45 + .55 * torch.abs((normal * light).sum(-1)).clamp(0, 1)
        color = (v_color * shade[:, None]).clamp(0, 1)
        return world, color

    def render(P, world, color, height=720, width=640, orientation=True, face_selection=None, return_links=False):
        projection = torch.as_tensor(P, dtype=torch.float32, device=device)
        hom = torch.cat((world, torch.ones((len(world), 1), device=device)), dim=1) @ projection.T
        z = hom[:, 2]
        near, far = .05, 30.
        clip = torch.stack((2 * hom[:, 0] / width - z, z - 2 * hom[:, 1] / height,
                            (far + near) / (far - near) * z - 2 * far * near / (far - near), z), dim=1)
        render_tri = tri if face_selection is None else tri[face_selection].contiguous()
        render_classes = class_gpu if face_selection is None else class_gpu[face_selection]
        rast, _ = dr.rasterize(ctx, clip[None].contiguous(), render_tri, resolution=[height, width], grad_db=False)
        image, _ = dr.interpolate(color[None].contiguous(), rast, render_tri)
        face_ids = rast[0, :, :, 3].long() - 1
        valid = face_ids >= 0
        label = torch.where(valid, render_classes[face_ids.clamp(min=0)], 0)
        pixel_links = None
        if return_links:
            face_links = v_ids[render_tri[:, 0].long()]
            pixel_links = torch.where(valid, face_links[face_ids.clamp(min=0)], -1)
        image = torch.where(valid[:, :, None], image[0], torch.full_like(image[0], .94))
        if orientation and flip:
            image, label = image.flip(0), label.flip(0)
            if pixel_links is not None:
                pixel_links = pixel_links.flip(0)
        result = ((image.detach().cpu().numpy() * 255).round().astype(np.uint8), label.detach().cpu().numpy().astype(np.uint8))
        return result + (pixel_links.detach().cpu().numpy().astype(np.int16),) if return_links else result

    def transform_array(base, quat, body, finger, box_center, box_quat):
        t = np.tile(np.eye(4), (len(kin.link_names) + 2, 1, 1))
        t[:len(kin.link_names)] = kin.forward(base, quat, body, finger)
        t[box_link, :3, :3] = Rotation.from_quat(box_quat).as_matrix()
        t[box_link, :3, 3] = box_center
        return t

    def look_at_projection(eye, target, up, width, height, focal_scale=1.05):
        eye, target, up = (np.asarray(x, dtype=float) for x in (eye, target, up))
        z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack((x, y, z)); K = np.asarray([[height * focal_scale, 0, width / 2], [0, height * focal_scale, height / 2], [0, 0, 1]])
        return K @ np.column_stack((R, -R @ eye))

    with np.load(ROOT / "outputs/user_reconstruction/geometry.npz", allow_pickle=False) as data:
        geometry = {key: data[key] for key in data.files}
    q0 = kin.initial_state
    t0 = transform_array(q0[:3], Rotation.from_rotvec(q0[3:6]).as_quat(), q0[6:], kin.q0_fingers,
                         scene["box"]["center_xyz_m"], scene["box"]["quaternion_xyzw"])
    world, color = transformed(t0)
    raw_image, raw_label = render(geometry["camera_P"][0], world, color, orientation=False)
    reference = geometry["reference_robot_mask"][0]

    def iou(a, b, valid=None):
        if valid is not None:
            a, b = a & valid, b & valid
        return float((a & b).sum() / max((a | b).sum(), 1))

    direct = iou(raw_label == 1, reference)
    inverted = iou(raw_label[::-1] == 1, reference)
    flip = inverted > direct
    orientation_iou = max(direct, inverted)
    if orientation_iou < .80:
        raise RuntimeError(f"URDF projection/rasterization reference check failed: {direct}, {inverted}")
    image, _ = render(geometry["camera_P"][0], world, color)
    reference_prefix = "user_" + args.variant if args.variant in ("v3", "v4") else "user_v2"
    cv2.imwrite(str(previews / (reference_prefix + "_renderer_reference.jpg")), image[:, :, ::-1])
    renderer_check = {"status": "passed", "reference_robot_silhouette_iou": orientation_iou,
                      "vertical_flip": bool(flip), "render_faces": len(triangles), "render_vertices": len(local),
                      "scene": str(args.scene), "urdf_sha256": kin.urdf_sha256,
                      "reference_image_hand_model_matches": "rohand" not in kin.urdf_path.name,
                      "geometry_simplification": f"Body per-link quadric decimation at most1800 faces; hand per-link limit={args.hand_face_limit} (0 means every original triangle). Source meshes unchanged."}
    (ROOT / (f"reports/user-{args.variant}-renderer.json" if args.variant in ("v3", "v4") else "reports/user-v2-renderer.json")).write_text(json.dumps(renderer_check, indent=2) + "\n", encoding="utf-8")
    if args.reference_test:
        print(json.dumps(renderer_check), flush=True)
        return

    if args.assembly_preview:
        from user_rohand_coupling import ROHandCoupling
        coupling = ROHandCoupling(ROOT)
        hand_face_masks = []
        hand_P = []
        for side, sign in (("left", 1), ("right", -1)):
            link_ids = [i for i, name in enumerate(kin.link_names) if name.startswith(side + "_hand_ro_")]
            hand_face_masks.append(torch.as_tensor(np.isin(ids[triangles[:, 0]], link_ids), device=device))
            wrist = t0[kin.link_index[side + "_wrist_yaw_link"]]
            center = wrist[:3, 3] + wrist[:3, :3] @ np.asarray([.16, 0, 0])
            eye = center + wrist[:3, :3] @ np.asarray([.07, -sign * .36, .07])
            up = wrist[:3, :3] @ np.asarray([0, 0, 1.])
            hand_P.append(look_at_projection(eye, center, up, 480, 480, 1.12))
        sheet = np.full((1040, 1440, 3), [16, 24, 39], np.uint8)
        for col, fraction in enumerate((0., .4, .8)):
            controls = np.asarray(coupling.upper) * fraction
            transforms = transform_array(q0[:3], Rotation.from_rotvec(q0[3:6]).as_quat(), q0[6:], coupling.expand_one(controls),
                                         scene["box"]["center_xyz_m"], scene["box"]["quaternion_xyzw"])
            world, color = transformed(transforms)
            for row, side in enumerate(("left", "right")):
                image, _ = render(hand_P[row], world, color, 480, 480, face_selection=hand_face_masks[row])
                sheet[row*520+40:row*520+520, col*480:col*480+480] = image
                cv2.putText(sheet, f'{side.upper()} | 5 fingers | coupled control {fraction:.0%}', (col*480+8, row*520+25), 0, .58, (235, 240, 250), 1, cv2.LINE_AA)
        path = previews / "user_g1_five_finger_hands.jpg"
        cv2.imwrite(str(path), sheet[:, :, ::-1])
        demo = previews / "user_g1_five_finger_demo.mp4"
        with av.open(str(demo), "w") as container:
            stream = container.add_stream("libx264", rate=24); stream.width = 960; stream.height = 560; stream.pix_fmt = "yuv420p"; stream.options = {"crf":"21"}
            for frame in range(72):
                fraction = .8 * np.sin(np.pi * frame / 71) ** 2
                controls = np.asarray(coupling.upper) * fraction
                transforms = transform_array(q0[:3], Rotation.from_rotvec(q0[3:6]).as_quat(), q0[6:], coupling.expand_one(controls),
                                             scene["box"]["center_xyz_m"], scene["box"]["quaternion_xyzw"])
                world, color = transformed(transforms)
                canvas = np.full((560, 960, 3), [16, 24, 39], np.uint8)
                for side in range(2):
                    image, _ = render(hand_P[side], world, color, 480, 480, face_selection=hand_face_masks[side])
                    canvas[70:550, side*480:side*480+480] = image
                cv2.putText(canvas, 'G1 + FIVE-FINGER ROHAND | LEFT / RIGHT', (12, 25), 0, .72, (235, 240, 250), 1, cv2.LINE_AA)
                cv2.putText(canvas, '6 actuators per hand; official nonlinear linkage | FK demonstration, not a fitted grasp', (12, 53), 0, .53, (245, 190, 90), 1, cv2.LINE_AA)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        print(json.dumps({"status":"five_finger_assembly_preview_exported", "image":str(path), "video":str(demo), "urdf_sha256":kin.urdf_sha256}), flush=True)
        return

    if args.keyframe_preview:
        def read_npz(path):
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        partial_body = read_npz(args.body_file or OUT / "body_partial_keyframes.npz")
        partial_hand = read_npz(args.keyframe_preview)
        camera = read_npz(OUT / "camera_refinement.npz")
        box = read_npz(OUT / "box_trajectory.npz")
        input_data = read_npz(ROOT / "outputs/user_reconstruction/input.npz")
        from user_hand_observations_v3 import make_observation
        landmarks = read_npz(OUT/"body_landmark_observations.npz")
        segmentation = read_npz(ROOT/"outputs/user_reconstruction/segmentation.npz")
        preview_metrics=[]
        hand_frames = list(map(int, partial_hand["frame_indices"]))
        frames = args.preview_frames or hand_frames
        body_frames = list(map(int, partial_body["frame_indices"]))
        body_names = list(map(str, partial_body["joint_names"]))
        finger_names = list(map(str, partial_hand["joint_names"]))
        body_values = partial_body["joint_positions_rad"]
        if "body_joint_positions_rad_refined" in partial_hand:
            body_values = partial_hand["body_joint_positions_rad_refined"]
            body_names = list(map(str, partial_hand["body_joint_names"]))
        canvas = np.full((len(frames) * 400, 1280, 3), [16, 24, 39], np.uint8)
        P_free = look_at_projection([2.4, -2.4, 1.55], [.30, 0, .75], [0, 0, 1], 640, 360, 1.5)
        for row, frame in enumerate(frames):
            bi = body_frames.index(frame)
            hi = hand_frames.index(frame)
            qi = hi if "body_joint_positions_rad_refined" in partial_hand else bi
            q = body_values[qi, [body_names.index(name) for name in kin.body_joint_names]]
            finger = partial_hand["joint_positions"][hi, [finger_names.index(name) for name in kin.finger_joint_names]]
            transforms = transform_array(partial_body["base_xyz_m"][bi], partial_body["base_quaternion_xyzw"][bi], q,
                                         dict(zip(kin.finger_joint_names, finger)), box["center_xyz_m"][frame], box["quaternion_xyzw"][frame])
            world, color = transformed(transforms)
            tiles = []
            for view in range(2):
                _, label, pixel_links = render(camera["P_refined"][view], world, color,return_links=True)
                original = input_data["rgb"][frame, view].copy()
                for kind, tint in ((1, (30, 220, 240)), (2, (240, 210, 40))):
                    contours, _ = cv2.findContours((label == kind).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(original, contours, -1, tint, 2, cv2.LINE_AA)
                side=1-view; wi=2+4*view; pi=wi+1
                if landmarks['valid'][frame,view,wi]and landmarks['valid'][frame,view,pi]:
                    observed=make_observation(input_data['rgb'][frame,view],segmentation['robot'][frame,view],segmentation['box'][frame,view],~input_data['watermark_invalid'][frame,view],landmarks['positions_xy'][frame,view,pi],landmarks['positions_xy'][frame,view,wi])
                    if observed is not None:
                        hand_ids=[i for i,name in enumerate(kin.link_names)if name.startswith(('left','right')[side]+'_hand_ro_')]
                        predicted=np.isin(pixel_links,hand_ids)
                        preview_metrics.append({'frame':frame,'view':view,'side':('left','right')[side],
                            'source_pixels':int(observed['hand_mask'].sum()),'candidate_visible_pixels_in_domain':int((predicted&observed['domain']).sum()),
                            'visible_iou_in_rgb_domain':iou(predicted,observed['hand_mask'],observed['domain'])})
                        x0,y0,x1,y1=map(int,observed['bounds']);crop=original[y0:y1,x0:x1]
                        cv2.imwrite(str(previews/f'user_v4_hand_crop_{args.keyframe_preview.stem}_{frame:03d}_{side}.jpg'),cv2.resize(crop,(640,520),interpolation=cv2.INTER_NEAREST)[:,:,::-1])
                tiles.append(original)
            video = cv2.resize(np.concatenate(tiles, axis=1), (640, 360), interpolation=cv2.INTER_AREA)
            free, _ = render(P_free, world, color, height=360, width=640)
            canvas[row*400+40:row*400+400, :640] = video
            canvas[row*400+40:row*400+400, 640:] = free
            cv2.putText(canvas, f'Frame {frame:03d} | actual optimized keyframe | cyan: five-finger URDF mesh', (10, row*400+25), 0, .64, (235, 240, 250), 1, cv2.LINE_AA)
        destination = previews / f"user_v4_keyframe_full_mesh_{args.keyframe_preview.stem}.jpg"
        cv2.imwrite(str(destination), canvas[:, :, ::-1])
        preview_report={'status':'actual_complete_scene_zbuffer_keyframe_qa','frames':frames,'inputs':{str(p.relative_to(ROOT)):{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for p in (args.body_file,args.keyframe_preview)},'visible_hand_metrics':preview_metrics,'image':str(destination.relative_to(ROOT)),'image_sha256':sha(destination),'strict_metric_reconstruction_valid':False}
        (ROOT/f'reports/user-v4-preview-{args.keyframe_preview.stem}.json').write_text(json.dumps(preview_report,indent=2)+'\n')
        print(json.dumps({"status":"actual_keyframe_mesh_qa_only", "frames":frames, "path":str(destination), "sha256":sha(destination),"visible_hand_metrics":preview_metrics}), flush=True)
        return

    paths = {name: OUT / (name + ".npz") for name in ("body_trajectory", "hand_trajectory", "box_trajectory", "camera_refinement")}
    paths["body_trajectory"] = args.body_file or (output_dir / "body_trajectory.npz" if args.variant in ("v3", "v4") else OUT / ("body_trajectory_contact.npz" if args.variant == "contact" else "body_trajectory.npz"))
    paths["hand_trajectory"] = args.hand_file or (output_dir / "hand_trajectory.npz" if args.variant in ("v3", "v4") else OUT / ("hand_trajectory_contact.npz" if args.variant == "contact" else "hand_trajectory_wrist_refined.npz"))
    trajectory_path = output_dir / ("full_trajectory_contact.npz" if args.variant == "contact" else "full_trajectory.npz")
    mesh_frame_prefix = "scene_contact_frame_" if args.variant == "contact" else "scene_frame_"
    arrays = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            arrays[name] = {key: data[key] for key in data.files}
    body, hand, box, camera = (arrays[name] for name in ("body_trajectory", "hand_trajectory", "box_trajectory", "camera_refinement"))
    body_names = list(map(str, body["joint_names"]))
    finger_names = list(map(str, hand["joint_names"]))
    body_q = body["joint_positions_rad"][:, [body_names.index(name) for name in kin.body_joint_names]]
    original_body_q = body_q.copy()
    wrist_refinement_used = "body_joint_positions_rad_refined" in hand
    if wrist_refinement_used:
        refine_names = list(map(str, hand["body_joint_names"]))
        body_q = hand["body_joint_positions_rad_refined"][:, [refine_names.index(name) for name in kin.body_joint_names]]
        nonwrist = [i for i, name in enumerate(kin.body_joint_names) if "wrist_" not in name]
        assert np.array_equal(body_q[:, nonwrist], original_body_q[:, nonwrist]), "Hand stage silently changed non-wrist body joints"
        assert str(hand["source_body_sha256"]) == sha(paths["body_trajectory"]), "Wrist refinement has a stale source body"
    finger_values = hand["joint_positions"] if "joint_positions" in hand else hand["joint_positions_rad"]
    finger_q = finger_values[:, [finger_names.index(name) for name in kin.finger_joint_names]]
    if not body_q.shape == (121, 29) or not finger_q.shape == (121, len(kin.finger_joint_names)):
        raise RuntimeError("Incomplete optimized motion")
    assert np.isfinite(body_q).all() and np.isfinite(finger_q).all()
    assert ((body_q >= kin.body_limits[:, 0] - 1e-6) & (body_q <= kin.body_limits[:, 1] + 1e-6)).all()
    xml = ET.parse(kin.urdf_path).getroot()
    finger_limits = {j.attrib["name"]: ([float(j.find("limit").attrib[x]) for x in ("lower", "upper")] if j.find("limit") is not None else [-np.inf,np.inf])
                     for j in xml.findall("joint") if j.attrib["name"] in finger_names}
    bounds = np.asarray([finger_limits[name] for name in kin.finger_joint_names])
    assert ((finger_q >= bounds[:, 0] - 1e-6) & (finger_q <= bounds[:, 1] + 1e-6)).all()
    from user_rohand_coupling import ROHandCoupling
    coupling = ROHandCoupling(ROOT)
    controls = hand["actuator_positions"]
    control_names = list(map(str, hand["actuator_names"]))
    controls = controls[:, [control_names.index(name) for name in coupling.actuator_names]]
    expanded = coupling.expand_array(controls)
    expanded = expanded[:, [coupling.joint_names.index(name) for name in kin.finger_joint_names]]
    assert np.allclose(finger_q, expanded, atol=1e-8), "ROHand linkage does not match official coupled actuator controls"
    P = camera["P_refined"]
    with np.load(ROOT / "outputs/user_reconstruction/input.npz", allow_pickle=False) as data:
        rgb, invalid, times = data["rgb"], data["watermark_invalid"], data["timestamps"]
    with np.load(ROOT / "outputs/user_reconstruction/segmentation.npz", allow_pickle=False) as data:
        observed_robot, observed_box = data["robot"], data["box"]
    all_transforms = np.stack([transform_array(body["base_xyz_m"][f], body["base_quaternion_xyzw"][f], body_q[f],
                                             dict(zip(kin.finger_joint_names, finger_q[f])), box["center_xyz_m"][f],
                                             box["quaternion_xyzw"][f]) for f in range(121)])
    visible_hand_baseline = None
    visibility_landmarks = None
    if args.variant in ("v3", "v4"):
        from user_hand_observations_v3 import make_observation
        from scipy.spatial import cKDTree
        baseline_body_path = ROOT / "outputs/user_reconstruction_v3/body_trajectory.npz" if args.variant == "v4" else OUT / "body_trajectory_contact.npz"
        baseline_hand_path = ROOT / "outputs/user_reconstruction_v3/hand_trajectory.npz" if args.variant == "v4" else OUT / "hand_trajectory_contact.npz"
        baseline_body_hash = "8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c" if args.variant == "v4" else "159cabb128e6f2d2f1f182e747c77f51ab6b25c79cfc22df9e49a1de46fb6f66"
        baseline_hand_hash = "6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1" if args.variant == "v4" else "4f8f7c16df560d785c515fa8462c7de7243e93e1211c351ed6028b20a93ca9a1"
        if sha(baseline_body_path) != baseline_body_hash:
            raise RuntimeError("Frozen visibility baseline body changed")
        if sha(baseline_hand_path) != baseline_hand_hash:
            raise RuntimeError("Frozen visibility baseline hand changed")
        with np.load(baseline_body_path, allow_pickle=False) as data:
            frozen_body = {key: data[key] for key in data.files}
        with np.load(baseline_hand_path, allow_pickle=False) as data:
            frozen_hand = {key: data[key] for key in data.files}
        with np.load(OUT / "body_landmark_observations.npz", allow_pickle=False) as data:
            visibility_landmarks = {key: data[key] for key in data.files}
        if list(map(str, frozen_hand["body_joint_names"])) != kin.body_joint_names:
            raise RuntimeError("Visibility audit body ordering differs")
        frozen_finger_names = list(map(str, frozen_hand["joint_names"]))
        frozen_finger_q = frozen_hand["joint_positions"][:, [frozen_finger_names.index(name) for name in kin.finger_joint_names]]
        visible_hand_baseline = np.stack([
            transform_array(frozen_body["base_xyz_m"][f], frozen_body["base_quaternion_xyzw"][f],
                            frozen_hand["body_joint_positions_rad_refined"][f], dict(zip(kin.finger_joint_names, frozen_finger_q[f])),
                            box["center_xyz_m"][f], box["quaternion_xyzw"][f]) for f in range(121)])

    def visible_hand_metrics(frame, view, links, baseline_links):
        side = 1 - view
        wrist_index, palm_index = 2 + 4 * view, 3 + 4 * view
        source = visibility_landmarks
        if not (source["valid"][frame, view, wrist_index] and source["valid"][frame, view, palm_index]):
            return None
        obs = make_observation(rgb[frame, view], observed_robot[frame, view], observed_box[frame, view], ~invalid[frame, view],
                               source["positions_xy"][frame, view, palm_index], source["positions_xy"][frame, view, wrist_index])
        if obs is None:
            return None
        hand_ids = [i for i, name in enumerate(kin.link_names) if name.startswith(("left", "right")[side] + "_hand_ro_")]
        domain = obs["domain"]
        interior = cv2.erode(domain.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

        def external_boundary(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boundary = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(boundary, [c for c in contours if len(c) >= 5], -1, 1, 1)
            return boundary.astype(bool)

        source_edge = external_boundary(obs["hand_mask"]) & interior
        yy, xx = np.nonzero(source_edge)
        heldout = (xx // 6 + yy // 6) % 5 == 0
        points = np.column_stack((xx[heldout], yy[heldout]))
        record = {"side": ("left", "right")[side], "source_pixels": int(obs["hand_mask"].sum()),
                  "heldout_external_edge_points": len(points), "observation_available": True}
        images = []
        for name, pixel_ids in (("frozen_v3" if args.variant == "v4" else "frozen_v2", baseline_links), ("candidate", links)):
            predicted = np.isin(pixel_ids, hand_ids)
            edge = external_boundary(predicted)
            py, px = np.nonzero(edge & interior)
            distances = cKDTree(np.column_stack((px, py))).query(points)[0] if len(px) and len(points) else np.asarray([])
            record[name] = {"visible_mask_iou_in_source_domain": iou(predicted, obs["hand_mask"], domain),
                            "external_heldout_mean_px": float(distances.mean()) if len(distances) else None,
                            "external_heldout_p90_px": float(np.percentile(distances, 90)) if len(distances) else None,
                            "visible_hand_pixels_in_domain": int((predicted & domain).sum()),
                            "missing_visible_model_contour": bool(len(points) and not len(px))}
            image = rgb[frame, view].copy()
            image[edge & obs["roi"]] = (255, 150, 30) if name.startswith("frozen_") else (30, 230, 255)
            x0, y0, x1, y1 = map(int, obs["bounds"])
            images.append(cv2.resize(image[y0:y1, x0:x1], (320, 260), interpolation=cv2.INTER_NEAREST))
        if frame in (0, 27, 54, 60, 120):
            sheet = np.full((300, 640, 3), [16, 24, 39], np.uint8)
            sheet[40:, :320], sheet[40:, 320:] = images
            cv2.putText(sheet, f'Frame {frame:03d} {record["side"]}: {"V3" if args.variant == "v4" else "V2"} visible mesh | {args.variant.upper()} visible mesh', (8, 24), 0, .45, (235, 240, 250), 1)
            cv2.imwrite(str(previews / f'user_{args.variant}_visible_hand_{record["side"]}_{frame:03d}.jpg'), sheet[:, :, ::-1])
        return record
    for f in range(121):
        assert np.allclose(all_transforms[f, :len(kin.link_names)], kin.forward(body["base_xyz_m"][f], body["base_quaternion_xyzw"][f], body_q[f], finger_q[f]), atol=1e-7)
    finger_units = dict(zip(coupling.joint_names, coupling.joint_units))
    np.savez_compressed(trajectory_path, timestamps=times, joint_names=np.asarray(kin.body_joint_names + kin.finger_joint_names),
                        joint_positions=np.concatenate((body_q, finger_q), axis=1),
                        joint_units=np.asarray(["rad"]*29+[finger_units[name] for name in kin.finger_joint_names]),
                        hand_actuator_names=np.asarray(coupling.actuator_names), hand_actuator_positions=controls,
                        hand_actuator_units=np.asarray(coupling.actuator_units), body_joint_positions_rad=body_q, base_xyz_m=body["base_xyz_m"],
                        body_joint_positions_rad_before_wrist_refinement=original_body_q,
                        wrist_refinement_used=np.asarray(wrist_refinement_used),
                        base_quaternion_xyzw=body["base_quaternion_xyzw"], link_names=np.asarray(kin.link_names),
                        link_world_transforms=all_transforms[:, :len(kin.link_names)],
                        box_center_xyz_m=box["center_xyz_m"], box_quaternion_xyzw=box["quaternion_xyzw"],
                        body_observed=body.get("joint_image_supported", body.get("observed", np.zeros((121, 29), bool))),
                        hand_joint_observed=hand.get("observed", np.zeros((121, 50), bool)),
                        hand_actuator_observed=hand.get("actuator_observed", np.zeros((121, 12), bool)),
                        body_valid=body.get("candidate_valid", body.get("valid", np.zeros(121, bool))), hand_valid=hand.get("valid", np.zeros(121, bool)),
                        box_valid=box.get("valid", np.zeros(121, bool)), strict_metric_reconstruction_valid=np.asarray(False))
    from export_rigid_glb import export_rigid_animation
    animated_path = output_dir / ("scene_animated_contact.glb" if args.variant == "contact" else "scene_animated.glb")
    animation_audit = export_rigid_animation(
        animated_path, local, triangles, local_normals, rgb_colors, ids, all_transforms,
        kin.link_names + ["known_box_candidate", "static_table_and_floor"], times,
        {"run_variant": args.variant, "strict_metric_reconstruction_valid": False,
         "physical_hardware_validated": False, "urdf_sha256": kin.urdf_sha256,
         "source_video_sha256": sha(ROOT / "data/video_generation/candidate_scene/generated.mp4"),
         "inputs": {key: sha(path) for key, path in paths.items()}})
    print(json.dumps({"animation_export": animation_audit}), flush=True)

    def free_camera(eye, width, height):
        eye = np.asarray(eye); target = np.asarray([.28, 0, .77]); up = np.asarray([0, 0, 1.])
        z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack((x, y, z)); K = np.asarray([[height * 1.05, 0, width / 2], [0, height * 1.05, height / 2], [0, 0, 1]])
        return K @ np.column_stack((R, -R @ eye))

    free_P = [free_camera(eye, w, h) for eye, w, h in (([2.4, -2.5, 1.6], 640, 640), ([3.2, 0, 1.3], 320, 320), ([.3, -3.0, 1.3], 320, 320))]
    preview_prefix = "user_" + args.variant if args.variant in ("v3", "v4") else ("user_v2_contact" if args.variant == "contact" else "user_v2")
    overlay_path = previews / (preview_prefix + "_full_overlay.mp4")
    mesh_path = previews / (preview_prefix + "_full_mesh.mp4")
    metrics = []
    with av.open(str(overlay_path), "w") as overlay_container, av.open(str(mesh_path), "w") as mesh_container:
        overlay_stream = overlay_container.add_stream("libx264", rate=24); overlay_stream.width = 1280; overlay_stream.height = 820; overlay_stream.pix_fmt = "yuv420p"; overlay_stream.options = {"crf": "21", "preset": "medium"}
        mesh_stream = mesh_container.add_stream("libx264", rate=24); mesh_stream.width = 960; mesh_stream.height = 748; mesh_stream.pix_fmt = "yuv420p"; mesh_stream.options = {"crf": "21", "preset": "medium"}
        for frame in range(121):
            world, color = transformed(all_transforms[frame]); views = []; values = []
            visibility_values = []
            if visible_hand_baseline is not None:
                frozen_world, frozen_color = transformed(visible_hand_baseline[frame])
            collision = hand_collision_audit(all_transforms[frame], frame, box)
            collision.update(non_hand_box_audit(all_transforms[frame],frame,box))
            hand_image_sides = hand.get("image_fit_valid", np.zeros((121, 2), bool))[frame]
            quality_caption = (f'Box strict: {"PASS" if box["valid"][frame] else "FAIL"} | '
                               f'Hand image L/R: {"PASS" if hand_image_sides[0] else "FAIL"}/{"PASS" if hand_image_sides[1] else "FAIL"} | '
                               f'Hand/box penetration: {collision["hand_box_max_surface_sample_penetration_m"] * 1000:.1f} mm')
            ground_failed = False
            if args.variant in ("v3", "v4"):
                minimum_z = float(body["raw_all_robot_min_z_m"][frame])
                ground_failed = minimum_z < -.005
                extra_caption = (f'Ground: {"FAIL" if ground_failed else "PASS"} ({minimum_z * 1000:.1f} mm) | '
                                 f'Candidate: {"PASS" if body["candidate_valid"][frame] else "CONDITIONAL"} | '
                                 f'Body/box penetration: {collision["non_hand_box_max_surface_sample_penetration_m"] * 1000:.2f} mm')
            baseline_t = transform_array(q0[:3], Rotation.from_rotvec(q0[3:6]).as_quat(), q0[6:], kin.q0_fingers,
                                         box["center_xyz_m"][frame], box["quaternion_xyzw"][frame])
            baseline_world, baseline_color = transformed(baseline_t)
            for view in range(2):
                if visible_hand_baseline is not None:
                    _, label, pixel_links = render(P[view], world, color, return_links=True)
                    _, _, frozen_pixel_links = render(P[view], frozen_world, frozen_color, return_links=True)
                    visibility_values.append(visible_hand_metrics(frame, view, pixel_links, frozen_pixel_links))
                else:
                    _, label = render(P[view], world, color)
                _, baseline_label = render(P[view], baseline_world, baseline_color)
                original = rgb[frame, view].copy()
                for kind, tint in ((1, (30, 220, 240)), (2, (240, 210, 40))):
                    contours, _ = cv2.findContours((label == kind).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(original, contours, -1, tint, 1, cv2.LINE_AA)
                values.append({"view": view, "robot_visible_silhouette_iou": iou(label == 1, observed_robot[frame, view], ~invalid[frame, view]),
                               "baseline_q0_robot_visible_silhouette_iou": iou(baseline_label == 1, observed_robot[frame, view], ~invalid[frame, view]),
                               "box_visible_silhouette_iou": iou(label == 2, observed_box[frame, view], ~invalid[frame, view])})
                views.append(original)
            overlay = np.full((820, 1280, 3), [16, 24, 39], np.uint8); overlay[:720] = np.concatenate(views, axis=1)
            cv2.putText(overlay, f'Articulated candidate | frame {frame:03d}/120 | cyan: URDF, yellow: box', (12, 745), 0, .62, (225, 235, 245), 1, cv2.LINE_AA)
            cv2.putText(overlay, quality_caption, (12, 775), 0, .59, (255, 185, 85), 1, cv2.LINE_AA)
            cv2.putText(overlay,extra_caption,(12,804),0,.59,(255,70,70)if ground_failed or collision['non_hand_box_max_surface_sample_penetration_m']>.005 else(255,185,85),1,cv2.LINE_AA)
            canvas = np.full((748, 960, 3), [16, 24, 39], np.uint8)
            for index, (p, h, w, row, col) in enumerate(zip(free_P, (640, 320, 320), (640, 320, 320), (60, 60, 380), (0, 640, 640))):
                image, _ = render(p, world, color, height=h, width=w)
                canvas[row:row+h, col:col+w] = image
            cv2.putText(canvas, 'OPTIMIZED G1 + HANDS + BOX | CONDITIONAL 3D CANDIDATE', (10, 24), 0, .61, (235, 240, 250), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'Frame {frame:03d}/120 | 29 body + 12 coupled hand controls | conditional candidate', (10, 48), 0, .55, (255, 185, 85), 1, cv2.LINE_AA)
            cv2.putText(canvas, quality_caption, (10, 718), 0, .43, (255,185,85), 1, cv2.LINE_AA)
            cv2.putText(canvas,extra_caption,(10,739),0,.43,(255,70,70)if ground_failed or collision['non_hand_box_max_surface_sample_penetration_m']>.005 else(255,185,85),1,cv2.LINE_AA)
            for container, stream, image in ((overlay_container, overlay_stream, overlay), (mesh_container, mesh_stream, canvas)):
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            xyz_cpu = world.detach().cpu().numpy()
            metrics.append({"frame": frame, "views": values,
                            "visible_hand_geometry_comparison": visibility_values,
                            **collision,
                            "robot_mesh_min_world_z_m": float(xyz_cpu[:robot_vertex_count, 2].min())})
            if frame in ((0, 26, 54, 60, 120) if args.variant in ("v3", "v4") else (0, 60, 120)):
                cv2.imwrite(str(previews / f"{preview_prefix}_full_mesh_{frame:03d}.jpg"), canvas[:, :, ::-1])
            if frame in (0, 60, 120):
                xyz = world.detach().cpu().numpy()
                mesh = trimesh.Trimesh(vertices=xyz, faces=triangles, vertex_colors=(rgb_colors * 255).astype(np.uint8), process=False)
                mesh.export(output_dir / f"{mesh_frame_prefix}{frame:03d}.glb")
            if frame % 30 == 0:
                print(f'full mesh renderer frame {frame}/120', flush=True)
        for container, stream in ((overlay_container, overlay_stream), (mesh_container, mesh_stream)):
            for packet in stream.encode(): container.mux(packet)
    for path in (overlay_path, mesh_path):
        with av.open(str(path)) as container:
            assert sum(1 for _ in container.decode(video=0)) == 121
    visibility_summary = {}
    for side in ("left", "right"):
        rows = [value for row in metrics for value in row["visible_hand_geometry_comparison"] if value is not None and value["side"] == side]
        if not rows:
            continue
        summary = {"evaluated_frames": len(rows)}
        for candidate_name in ("frozen_v3" if args.variant == "v4" else "frozen_v2", "candidate"):
            values = [row[candidate_name] for row in rows]
            errors = [value["external_heldout_mean_px"] for value in values if value["external_heldout_mean_px"] is not None]
            summary[candidate_name] = {"median_visible_mask_iou": float(np.median([value["visible_mask_iou_in_source_domain"] for value in values])),
                                       "median_external_heldout_mean_px": float(np.median(errors)) if errors else None,
                                       "missing_visible_model_contour_frames": sum(value["missing_visible_model_contour"] for value in values)}
        visibility_summary[side] = summary
    implementation_paths = [Path(__file__), ROOT/"scripts/export_rigid_glb.py", ROOT/"scripts/user_body_kinematics.py", ROOT/"scripts/user_rohand_coupling.py", ROOT/"scripts/build_g1_five_finger.py"]
    if args.variant in ("v3", "v4"):
        implementation_paths.append(ROOT/"scripts/user_hand_observations_v3.py")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "articulated_scene_candidate_exported",
              "training": False, "physical_gpu_rendering": 7, "renderer_reference_check": renderer_check,
              "frames": 121, "robot_body_joints": 29, "robot_finger_motions": 50, "independent_hand_controls": 12, "fingers_per_hand": 5,
              "independent_full_mesh_audit": {"robot_iou_median_per_view": [float(np.median([r["views"][v]["robot_visible_silhouette_iou"] for r in metrics])) for v in range(2)],
                                               "baseline_q0_robot_iou_median_per_view": [float(np.median([r["views"][v]["baseline_q0_robot_visible_silhouette_iou"] for r in metrics])) for v in range(2)],
                                               "box_iou_median_per_view": [float(np.median([r["views"][v]["box_visible_silhouette_iou"] for r in metrics])) for v in range(2)],
                                               "max_hand_box_vertex_penetration_m": float(max(r["hand_box_max_vertex_penetration_m"] for r in metrics)),
                                               "max_hand_box_surface_sample_penetration_m": float(max(r["hand_box_max_surface_sample_penetration_m"] for r in metrics)),
                                               "max_non_hand_box_surface_sample_penetration_m":float(max(r['non_hand_box_max_surface_sample_penetration_m']for r in metrics)),
                                               "non_hand_box_failed_frames_gt5mm":[r['frame']for r in metrics if r['non_hand_box_max_surface_sample_penetration_m']>.005],
                                               "median_hand_vertices_box_penetration_gt5mm_fraction": float(np.median([r["hand_vertices_inside_box_deeper_than_5mm_fraction"] for r in metrics]))},
              "wrist_refinement_used": wrist_refinement_used,
              "animated_glb_audit": animation_audit,
              "hand_collision_geometry": "all original hand visual meshes: deduplicated vertices and every triangle center; independent of render decimation",
              "visible_hand_comparison_policy": "Complete-scene z-buffer identifies actually visible hand links. Compare frozen V3 and V4 candidate external hand contours and mask IoU in the same RGB-only observation domain. No projected mesh selects source pixels. Spatial edge holdout is diagnostic; image-domain/model uncertainty remains.",
              "visible_hand_comparison_summary": visibility_summary,
              "strict_metric_reconstruction_valid": False, "urdf_sha256": kin.urdf_sha256,
              "source_video_sha256": sha(ROOT / "data/video_generation/candidate_scene/generated.mp4"),
              "implementation_files": {str(path.relative_to(ROOT)):sha(path) for path in implementation_paths},
              "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for key, path in paths.items()},
              "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size} for path in [trajectory_path, overlay_path, mesh_path, animated_path] + sorted(output_dir.glob(mesh_frame_prefix + "*.glb"))],
              "per_frame_mesh_audit": metrics,
              "limitations": ["Known URDF mesh and fixed known cuboid are model-based reconstructions, not recovered arbitrary video surfaces.",
                              "Five-finger ROHand geometry and project-authored wrist adapters are not physically verified to match generated hands exactly.",
                              "Occluded and axial joint ambiguities remain; parameter availability is not proof of observation.",
                              "All original vertices and triangle centers are audited, but this is not a continuous triangle intersection proof.",
                              "No simulation, dynamics, torque feasibility or official KinoGen full pipeline validation."]}
    report["run_variant"] = args.variant
    if args.variant in ("v3", "v4"):
        report["visible_hand_audit_inputs"] = {str(path.relative_to(ROOT)): {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for path in (baseline_body_path, baseline_hand_path, OUT/"body_landmark_observations.npz")}
    report_path = ROOT / (f"reports/user-full-scene-{args.variant}.json" if args.variant in ("v3", "v4") else ("reports/user-full-scene-contact.json" if args.variant == "contact" else "reports/user-full-scene.json"))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "mesh_audit": report["independent_full_mesh_audit"], "outputs": report["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
