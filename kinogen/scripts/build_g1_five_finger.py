"""Assemble a project-local G1 + bilateral five-finger ROHand model from verified assets."""
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from user_rohand_coupling import ROHandCoupling
    scene_path = ROOT / "data/video_generation/candidate_scene/scene.json"
    original_scene = json.loads(scene_path.read_text(encoding="utf-8"))
    original_urdf = ROOT / original_scene["robot"]["urdf"]
    assert digest(original_urdf) == original_scene["robot"]["sha256"]
    directory = ROOT / "data/robots/g1_rohand_five_finger"
    directory.mkdir(parents=True, exist_ok=True)
    urdf_path = directory / "g1_rohand.urdf"
    model = ET.parse(original_urdf).getroot()
    model.attrib["name"] = "g1_29dof_bilateral_rohand_five_finger"
    removed = []
    for item in list(model):
        if item.tag in ("link", "joint") and "_hand_" in item.attrib.get("name", ""):
            removed.append(item.attrib["name"])
            model.remove(item)
    source_files = {str(original_urdf.relative_to(ROOT)): digest(original_urdf)}
    mesh_records = {}

    def relocate_meshes(element, package_root):
        for mesh in element.iter("mesh"):
            filename = mesh.attrib["filename"]
            if filename.startswith("package://rohand_urdf_ros1/"):
                path = ROOT / "dependencies/rohand-urdf" / filename.split("package://rohand_urdf_ros1/", 1)[1]
            else:
                path = package_root / filename
            path = path.resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise FileNotFoundError(str(path))
            mesh.attrib["filename"] = os.path.relpath(path, directory).replace("\\", "/")
            mesh_records[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)

    relocate_meshes(model, original_urdf.parent)
    mounts = []
    for side, sign in (("left", 1), ("right", -1)):
        source = ROOT / f"dependencies/rohand-urdf/urdf/rohand_{side}.urdf"
        source_files[str(source.relative_to(ROOT))] = digest(source)
        hand = ET.parse(source).getroot()
        prefix = side + "_hand_ro_"
        for item in list(hand):
            item = copy.deepcopy(item)
            if item.tag not in ("link", "joint"):
                continue
            item.attrib["name"] = prefix + item.attrib["name"]
            for ref in item.iter():
                if ref.tag in ("parent", "child"):
                    ref.attrib["link"] = prefix + ref.attrib["link"]
                if ref.tag == "mimic":
                    ref.attrib["joint"] = prefix + ref.attrib["joint"]
            relocate_meshes(item, source.parent)
            model.append(item)
        mount = ET.SubElement(model, "joint", name=prefix + "mount_joint", type="fixed")
        ET.SubElement(mount, "parent", link=side + "_wrist_yaw_link")
        ET.SubElement(mount, "child", link=prefix + "base_link")
        xyz = [.0415, sign * .003, 0.]
        rpy = [sign * np.pi / 2, 0., np.pi]
        ET.SubElement(mount, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
        mounts.append({"side": side, "parent": side + "_wrist_yaw_link", "child": prefix + "base_link",
                       "translation_m": xyz, "rpy_rad": rpy,
                       "rotation_matrix": Rotation.from_euler("xyz", rpy).as_matrix().tolist(),
                       "basis_policy": "ROHand -X finger direction maps to G1 +X; left/right index side maps to +Z; flexion toward opposite inward Y directions",
                       "mechanical_adapter_verified": False,
                       "note": "Project-authored fixed virtual wrist adapter. No hardware mounting, load/torque or author assembly certification."})
    links = {x.attrib["name"]: x for x in model.findall("link")}
    joints = list(model.findall("joint"))
    assert len(links) == len(model.findall("link"))
    assert len({j.attrib["name"] for j in joints}) == len(joints)
    assert not any("_hand_" in name and "_hand_ro_" not in name for name in links)
    parents = [j.find("parent").attrib["link"] for j in joints]
    children = [j.find("child").attrib["link"] for j in joints]
    assert set(parents + children).issubset(links)
    assert len(children) == len(set(children))
    assert set(links) - set(children) == {"pelvis"}
    coupling = ROHandCoupling(ROOT)
    controls = np.zeros(12)
    hand_q = coupling.expand_one(controls)
    body_q = {key: val for key, val in original_scene["robot"]["joint_positions_rad"].items() if "_hand_" not in key}
    assert len(body_q) == 29 and len(hand_q) == 50
    values = {**body_q, **hand_q}
    transforms = {"pelvis": np.eye(4)}
    transforms["pelvis"][:3, :3] = Rotation.from_quat(original_scene["robot"]["base_quaternion_xyzw"]).as_matrix()
    transforms["pelvis"][:3, 3] = original_scene["robot"]["base_position_xyz_m"]
    pending = list(joints)
    while pending:
        done = []
        for joint in pending:
            p, c = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
            if p not in transforms:
                continue
            origin = joint.find("origin"); fixed = np.eye(4)
            if origin is not None:
                fixed[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                fixed[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            motion = np.eye(4); kind = joint.attrib["type"]
            if kind != "fixed":
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" "); axis /= np.linalg.norm(axis)
                q = values[joint.attrib["name"]]
                if kind == "prismatic": motion[:3, 3] = axis * q
                else: motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
            transforms[c] = transforms[p] @ fixed @ motion
            done.append(joint)
        if not done: raise RuntimeError("Unconnected assembly")
        pending = [j for j in pending if j not in done]
    old_tf = original_scene["robot"]["link_world_transforms"]
    unchanged_error = max(float(np.max(np.abs(transforms[name] - np.asarray(old_tf[name])))) for name in transforms if name in old_tf)
    assert unchanged_error < 1e-8
    for side in ("left", "right"):
        for digit in ("th", "if", "mf", "rf", "lf"):
            assert side + "_hand_ro_" + digit + "_distal_link" in links
    ET.indent(model, space="  ")
    ET.ElementTree(model).write(urdf_path, encoding="utf-8", xml_declaration=True)
    scene = copy.deepcopy(original_scene)
    scene.update(scene_id="candidate_scene_g1_rohand_five_finger", status="five_finger_model_assembled_for_video_reconstruction", paper_exact_scene=False)
    historical_keys = ("renderer", "gpu", "cuda_visible_devices", "mesh_vertices", "mesh_triangles", "output_files", "pose_authoring")
    scene["original_reference_scene_metadata"] = {key: scene.pop(key) for key in historical_keys if key in scene}
    mixed_units = {**{k:"rad" for k in body_q}, **dict(zip(coupling.joint_names, coupling.joint_units))}
    mixed_limits = {}
    for joint in joints:
        name = joint.attrib["name"]
        if joint.attrib["type"] == "fixed": continue
        limit = joint.find("limit")
        mixed_limits[name] = {"type":joint.attrib["type"], "unit":mixed_units[name],
                              "lower":float(limit.attrib["lower"]) if limit is not None else None,
                              "upper":float(limit.attrib["upper"]) if limit is not None else None,
                              "dependent_official_coupling": name in hand_q and name not in coupling.actuator_names}
    scene["robot"].update(urdf=str(urdf_path.relative_to(ROOT)).replace("\\", "/"), sha256=digest(urdf_path),
                          model="g1_29dof_bilateral_rohand_five_finger", hands="Bilateral OYMotion ROHand: five physical fingers per hand, six coupled actuators per side",
                          rohand_assembly_performed=True, paper_exact_assembly_verified=False,
                          joint_positions_rad=body_q, joint_positions=values,
                          joint_units=mixed_units, joint_limits=mixed_limits,
                          joint_limits_rad={k:v for k,v in original_scene["robot"]["joint_limits_rad"].items() if k in body_q},
                          hand_actuator_names=coupling.actuator_names, hand_actuator_positions=controls.tolist(),
                          hand_actuator_units=coupling.actuator_units,
                          link_world_transforms={name: matrix.tolist() for name, matrix in transforms.items()},
                          visual_mesh_sha256=[{"path": name, "sha256": checksum} for name, checksum in mesh_records.items()], wrist_mounts=mounts)
    # Preserve the original renderer/reference metadata as history, not a claim
    # that the old three-finger reference image depicts the new robot assembly.
    scene["reference_image_robot_model_matches"] = False
    scene["limitations"] = ["Virtual fixed wrist adapters are project-authored, not hardware-qualified.",
                            "Existing reference/video cameras retain their conditional validation; original reference image used stock three-finger hands.",
                            "Five-finger topology agrees with generated appearance but exact link shapes and hand dimensions may differ.",
                            "ROHand linkage uses official six-actuator nonlinear coupling, not 25 independent DOF per hand."]
    new_scene_path = ROOT / "data/video_generation/candidate_scene/scene_rohand.json"
    new_scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    report = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "status": "g1_bilateral_five_finger_rohand_assembled",
              "urdf": {"path": str(urdf_path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(urdf_path)},
              "scene": {"path": str(new_scene_path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(new_scene_path)},
              "fingers_per_hand": 5, "body_independent_dof": 29, "hand_independent_controls": 12,
              "hand_expanded_urdf_motions": 50, "total_independent_controls": 41,
              "link_count": len(links), "joint_count": len(joints), "unchanged_body_fk_max_error": unchanged_error,
              "original_stock_hand_links_and_joints_removed": removed,
              "source_files": source_files, "source_snapshots_modified": False,
              "wrist_mounts": mounts, "coupling_domain_review": coupling.domain_review(),
              "hardware_mount_validated": False, "physics_validated": False,
              "existing_reference_image_modified": False, "training": False}
    (ROOT / "reports/user-g1-five-finger.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in ("status", "urdf", "scene", "fingers_per_hand", "total_independent_controls", "link_count", "unchanged_body_fk_max_error")}), flush=True)


if __name__ == "__main__":
    main()
