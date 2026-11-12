"""Write a portable glTF animation for a rigid articulated mesh at actual sample times."""
import json
from pathlib import Path
import struct


def export_rigid_animation(path, vertices, triangles, normals, colors, vertex_links,
                           transforms, node_names, times, metadata):
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    path = Path(path)
    times = np.asarray(times, dtype=np.float32)
    transforms = np.asarray(transforms, dtype=np.float64)
    if len(times) != len(transforms) or not np.all(np.diff(times) > 0):
        raise ValueError("Animation requires all actual strictly increasing sample times")
    binary = bytearray()
    gltf = {"asset": {"version": "2.0", "generator": "KinoGen project conditional rigid-mesh export"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
            "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0,
                          "roughnessFactor": .75}, "doubleSided": True}],
            "buffers": [], "bufferViews": [], "accessors": [],
            "animations": [{"name": "Actual 121-frame conditional trajectory", "samplers": [], "channels": []}],
            "extras": metadata}

    def accessor(values, kind, target=None, bounds=False):
        values = np.ascontiguousarray(values)
        if values.dtype not in (np.dtype("float32"), np.dtype("uint32")):
            raise TypeError("Unexpected glTF accessor dtype")
        while len(binary) % 4:
            binary.append(0)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": values.nbytes}
        if target is not None:
            view["target"] = target
        binary.extend(values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes())
        gltf["bufferViews"].append(view)
        record = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5126 if values.dtype.kind == "f" else 5125,
                  "count": len(values), "type": kind}
        if bounds:
            flat = values.reshape(len(values), -1)
            record.update(min=flat.min(axis=0).tolist(), max=flat.max(axis=0).tolist())
        gltf["accessors"].append(record)
        return len(gltf["accessors"]) - 1

    time_accessor = accessor(times, "SCALAR", bounds=True)
    animated_nodes = []
    for link, name in enumerate(node_names):
        selected = np.flatnonzero(vertex_links == link)
        if not len(selected):
            continue
        selected_faces = triangles[np.all(vertex_links[triangles] == link, axis=1)]
        mapping = np.full(len(vertices), -1, dtype=np.int64)
        mapping[selected] = np.arange(len(selected))
        faces = mapping[selected_faces].astype(np.uint32)
        if not len(faces) or faces.max() >= len(selected):
            raise ValueError("Mesh crosses rigid-link nodes")
        primitive = {"attributes": {"POSITION": accessor(np.asarray(vertices[selected], dtype=np.float32), "VEC3", 34962, True),
                                    "NORMAL": accessor(np.asarray(normals[selected], dtype=np.float32), "VEC3", 34962),
                                    "COLOR_0": accessor(np.asarray(colors[selected], dtype=np.float32), "VEC3", 34962)},
                     "indices": accessor(faces.reshape(-1), "SCALAR", 34963), "material": 0, "mode": 4}
        gltf["meshes"].append({"name": name, "primitives": [primitive]})
        positions = transforms[:, link, :3, 3].astype(np.float32)
        quaternions = Rotation.from_matrix(transforms[:, link, :3, :3]).as_quat().astype(np.float32)
        for frame in range(1, len(times)):
            if quaternions[frame - 1] @ quaternions[frame] < 0:
                quaternions[frame] *= -1
        node_id = len(gltf["nodes"])
        gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1,
                              "translation": positions[0].tolist(), "rotation": quaternions[0].tolist()})
        gltf["scenes"][0]["nodes"].append(node_id)
        if not np.allclose(transforms[:, link], transforms[0, link], atol=1e-12):
            animation = gltf["animations"][0]
            for channel, values, kind in (("translation", positions, "VEC3"), ("rotation", quaternions, "VEC4")):
                output = accessor(values, kind)
                animation["samplers"].append({"input": time_accessor, "output": output, "interpolation": "LINEAR"})
                animation["channels"].append({"sampler": len(animation["samplers"]) - 1,
                                               "target": {"node": node_id, "path": channel}})
            animated_nodes.append(name)
    roots = gltf["scenes"][0]["nodes"]
    root_id = len(gltf["nodes"])
    gltf["nodes"].append({"name": "Source_Z_up_to_glTF_Y_up", "children": roots,
                          "rotation": [-2 ** -.5, 0, 0, 2 ** -.5]})
    gltf["scenes"][0]["nodes"] = [root_id]
    gltf["extras"]["coordinates"] = "Source meters/Z-up; a fixed scene-root rotation converts to glTF Y-up"
    gltf["buffers"] = [{"byteLength": len(binary)}]
    encoded_json = json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode()
    encoded_json += b" " * (-len(encoded_json) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(encoded_json) + 8 + len(binary)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total))
        stream.write(struct.pack("<I4s", len(encoded_json), b"JSON"))
        stream.write(encoded_json)
        stream.write(struct.pack("<I4s", len(binary), b"BIN\0"))
        stream.write(binary)
    # Read back serialized buffers and reconstruct every stored rigid transform.
    data = path.read_bytes()
    json_size = struct.unpack_from("<I", data, 12)[0]
    decoded = json.loads(data[20:20 + json_size])
    bin_start = 20 + json_size + 8

    def read(index):
        a = decoded["accessors"][index]
        view = decoded["bufferViews"][a["bufferView"]]
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(data, dtype="<f4", offset=bin_start + view.get("byteOffset", 0),
                             count=a["count"] * width).reshape(a["count"], width)

    if not np.array_equal(read(time_accessor)[:, 0], times):
        raise RuntimeError("Serialized animation times changed")
    maximum_error = 0.
    channels = decoded["animations"][0]["channels"]
    samplers = decoded["animations"][0]["samplers"]
    for node_id, node in enumerate(decoded["nodes"]):
        if "mesh" not in node:
            continue
        link = node_names.index(node["name"])
        positions = np.tile(node["translation"], (len(times), 1))
        quaternions = np.tile(node["rotation"], (len(times), 1))
        for channel in channels:
            if channel["target"]["node"] != node_id:
                continue
            values = read(samplers[channel["sampler"]]["output"])
            if channel["target"]["path"] == "translation":
                positions = values
            else:
                quaternions = values
        restored = np.tile(np.eye(4), (len(times), 1, 1))
        restored[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
        restored[:, :3, 3] = positions
        maximum_error = max(maximum_error, float(np.max(np.abs(restored - transforms[:, link]))))
    if maximum_error > 2e-6:
        raise RuntimeError("Serialized animation does not match fitted transforms")
    # An independent general-purpose glTF loader must accept the complete scene.
    loaded = trimesh.load(path, force="scene")
    if sum(len(m.faces) for m in loaded.geometry.values()) != len(triangles):
        raise RuntimeError("Independent glTF loader lost mesh triangles")
    return {"path": str(path), "frames": len(times), "nodes": len(gltf["nodes"]),
            "animated_nodes": len(animated_nodes), "channels": len(gltf["animations"][0]["channels"]),
            "serialized_transform_max_abs_error": maximum_error, "independent_trimesh_load_passed": True,
            "sampling_note": "Stored transforms match every actual 24 FPS sample. Between-sample world-link TRS interpolation is for preview, not a physics simulation."}
