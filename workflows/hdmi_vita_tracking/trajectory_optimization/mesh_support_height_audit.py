"""Query original reconstructed triangle heights in the exact render frame."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--transform-reference", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--support-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.mesh, allow_pickle=False) as m:
        vertices, faces = np.asarray(m["vertices"], float), np.asarray(m["faces"], int)
    with np.load(args.transform_reference, allow_pickle=False) as t:
        rotation = np.asarray(t["world_from_source_rotation"], float)
        floor = float(np.asarray(t["floor_offset"]).reshape(-1)[0])
    scene = json.loads(args.scene.read_text())
    vertices = vertices @ rotation.T
    vertices[:, :2] -= np.asarray(scene["xy_offset"])
    vertices[:, 2] -= floor
    triangle = vertices[faces]
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    support = json.loads(args.support_audit.read_text())
    rows = []
    for source in (support["frames"][0], support["frames"][-1]):
        row = dict(frame=source["frame"], feet={})
        for side, foot in source["feet"].items():
            point = np.asarray(foot["closest_plane"]["foot_point"])
            x, y = point[:2]
            good = np.abs(den) > 1e-12
            u = np.zeros_like(den)
            v = np.zeros_like(den)
            u[good] = ((b[good, 1] - c[good, 1]) * (x - c[good, 0]) + (c[good, 0] - b[good, 0]) * (y - c[good, 1])) / den[good]
            v[good] = ((c[good, 1] - a[good, 1]) * (x - c[good, 0]) + (a[good, 0] - c[good, 0]) * (y - c[good, 1])) / den[good]
            inside = good & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
            z = u[inside] * a[inside, 2] + v[inside] * b[inside, 2] + (1 - u[inside] - v[inside]) * c[inside, 2]
            distance = np.linalg.norm(vertices[:, :2] - point[:2], axis=-1)
            near = {}
            for radius in (.05, .15, .35):
                values = vertices[distance < radius, 2]
                near[str(radius)] = dict(count=len(values), z_quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else None)
            below = z[z <= point[2] + .005]
            row["feet"][side] = dict(sole_point=point.tolist(), mesh_z_exact_xy=z.tolist(),
                                     highest_mesh_below_sole=float(below.max()) if len(below) else None,
                                     vertical_gap_to_mesh=float(point[2] - below.max()) if len(below) else None,
                                     nearest_vertex_xy_distance_m=float(distance.min()),
                                     nearest_vertex=vertices[distance.argmin()].tolist(), neighborhoods=near)
        rows.append(row)
    result = dict(transform=dict(rotation=rotation.tolist(), floor_offset=floor, xy_offset=scene["xy_offset"]),
                  vertices=len(vertices), triangles=len(faces), queries=rows,
                  note="Original visual reconstruction only; holes and surfaces do not certify collision geometry")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
