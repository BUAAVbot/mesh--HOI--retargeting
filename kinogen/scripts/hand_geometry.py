"""CPU visual geometry for legacy G1 hands and coupled five-finger ROHand."""
from pathlib import Path
import xml.etree.ElementTree as ET


class HandGeometry:
    def __init__(self, urdf, side):
        import numpy as np
        import trimesh
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        self.side = side
        self.palm = side + "_hand_palm_link"
        tree = ET.parse(urdf).getroot()
        self.joints = []
        self.names, self.lower, self.upper = [], [], []
        for joint in tree.findall("joint"):
            name = joint.attrib["name"]
            if name.startswith(side + "_hand_") and joint.attrib["type"] == "revolute":
                element = joint.find("origin")
                transform = np.eye(4)
                if element is not None:
                    transform[:3, 3] = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ")
                    transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                limit = joint.find("limit")
                self.joints.append((joint.find("parent").attrib["link"], joint.find("child").attrib["link"], transform, axis))
                self.names.append(name)
                self.lower.append(float(limit.attrib["lower"]))
                self.upper.append(float(limit.attrib["upper"]))
        if len(self.joints) != 7:
            raise ValueError(f"Expected 7 {side} finger joints, got {len(self.joints)}")
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        self.links = [self.palm] + [j[1] for j in self.joints]
        self.vertices = {}
        self.tip_local = {}
        for link in tree.findall("link"):
            name = link.attrib["name"]
            if name not in self.links:
                continue
            visual = link.find("visual")
            mesh = visual.find("geometry/mesh")
            geometry = trimesh.load_mesh(Path(urdf).parent / mesh.attrib["filename"], process=True)
            vertices = np.asarray(geometry.vertices, float) * np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
            origin = visual.find("origin")
            if origin is not None:
                vertices = vertices @ Rotation.from_euler("xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix().T + np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            hull = ConvexHull(vertices)
            hull_vertices = vertices[hull.vertices]
            self.vertices[name] = hull_vertices[np.linspace(0, len(hull_vertices)-1, min(len(hull_vertices), 100)).astype(int)]
            if name.endswith(("thumb_2_link", "middle_1_link", "index_1_link")):
                distance = np.linalg.norm(vertices, axis=1)
                self.tip_local[name] = vertices[distance >= np.percentile(distance, 90)].mean(axis=0)

    def world_geometry(self, angles, palm_transform):
        import numpy as np
        from scipy.spatial.transform import Rotation
        transforms = {self.palm: palm_transform}
        for angle, (parent, child, origin, axis) in zip(angles, self.joints):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transforms[child] = transforms[parent] @ origin @ motion
        points = {link: vertices @ transforms[link][:3, :3].T + transforms[link][:3, 3]
                  for link, vertices in self.vertices.items()}
        tips = np.stack([point @ transforms[link][:3, :3].T + transforms[link][:3, 3]
                         for link, point in self.tip_local.items()])
        return points, tips, transforms


class ROHandGeometry:
    """Five-finger visual geometry with the official nonlinear actuator linkage."""
    def __init__(self, project_root, side):
        import numpy as np
        import trimesh
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        from user_rohand_coupling import ROHandCoupling
        self.root, self.side = Path(project_root), side
        self.coupling = ROHandCoupling(project_root)
        self.base_name = side + "_hand_ro_base_link"
        xml = ET.parse(self.root/f"dependencies/rohand-urdf/urdf/rohand_{side}.urdf").getroot()
        self.joints, self.vertices, self.tip_local, self.contact_vertices, self.dense_contact_vertices = [], {}, {}, {}, {}
        offset=0 if side=='left' else 6
        self.lower=np.asarray(self.coupling.lower[offset:offset+6]);self.upper=np.asarray(self.coupling.upper[offset:offset+6])
        self.names=self.coupling.actuator_names[offset:offset+6]
        def origin(element):
            tf=np.eye(4)
            if element is not None:
                tf[:3,3]=np.fromstring(element.attrib.get('xyz','0 0 0'),sep=' ')
                tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
            return tf
        for joint in xml.findall('joint'):
            axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ');axis/=np.linalg.norm(axis)
            self.joints.append((joint.attrib['name'],joint.attrib['type'],joint.find('parent').attrib['link'],joint.find('child').attrib['link'],origin(joint.find('origin')),axis))
        for link in xml.findall('link'):
            name=link.attrib['name']
            visual=link.find('visual');mesh_el=visual.find('geometry/mesh')
            path=self.root/'dependencies/rohand-urdf'/mesh_el.attrib['filename'].split('package://rohand_urdf_ros1/')[-1]
            mesh=trimesh.load_mesh(path,process=True);vertices=np.asarray(mesh.vertices,float)
            vertices*=np.fromstring(mesh_el.attrib.get('scale','1 1 1'),sep=' ')
            tf=origin(visual.find('origin'));vertices=vertices@tf[:3,:3].T+tf[:3,3]
            # Surface samples for every actual visual, including palm/base and
            # internal linkage meshes. Root independently audits every vertex.
            centers=vertices[np.asarray(mesh.faces)].mean(axis=1)
            surface=np.concatenate((vertices,centers))
            self.dense_contact_vertices[name]=surface.copy()
            self.contact_vertices[name]=surface[np.linspace(0,len(surface)-1,min(len(surface),256)).astype(int)]
            # Internal sliders/linkages are not independent fingertip outlines.
            if 'slider' in name or 'connecting' in name:continue
            hull=ConvexHull(vertices);points=vertices[hull.vertices]
            self.vertices[name]=points[np.linspace(0,len(points)-1,min(len(points),80)).astype(int)]
            if 'distal' in name:
                distance=np.linalg.norm(vertices,axis=1);self.tip_local[name]=vertices[distance>=np.percentile(distance,92)].mean(axis=0)

    def world_geometry(self, actuators, base_transform):
        import numpy as np
        from scipy.spatial.transform import Rotation
        values=self.coupling.expand_side(actuators,self.side)
        prefix=self.side+'_hand_ro_';transforms={'base_link':base_transform}
        for name,kind,parent,child,origin,axis in self.joints:
            motion=np.eye(4);position=values[prefix+name]
            if kind=='prismatic':motion[:3,3]=axis*position
            else:motion[:3,:3]=Rotation.from_rotvec(axis*position).as_matrix()
            transforms[child]=transforms[parent]@origin@motion
        points={link:vertices@transforms[link][:3,:3].T+transforms[link][:3,3] for link,vertices in self.vertices.items()}
        tips=np.stack([point@transforms[link][:3,:3].T+transforms[link][:3,3] for link,point in self.tip_local.items()])
        return points,tips,{prefix+link:tf for link,tf in transforms.items()}

    def contact_surface(self, transforms, dense=False):
        import numpy as np
        prefix=self.side+'_hand_ro_'
        return np.concatenate([p@transforms[prefix+link][:3,:3].T+transforms[prefix+link][:3,3]
                               for link,p in (self.dense_contact_vertices if dense else self.contact_vertices).items()])


def project(points, projection):
    import numpy as np
    camera = np.column_stack((points, np.ones(len(points)))) @ projection.T
    return camera[:, :2] / np.maximum(camera[:, 2:], 1e-8), camera[:, 2]


def polygon_samples(points, number=12):
    import cv2
    import numpy as np
    polygon = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(float)
    if len(polygon) < 3:
        return np.repeat(polygon[:1], number, axis=0)
    segments = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0, lengths.cumsum()]
    values = np.arange(number) / number * cumulative[-1]
    index = np.clip(np.searchsorted(cumulative, values, side="right")-1, 0, len(lengths)-1)
    return polygon[index] + segments[index] * ((values-cumulative[index])/np.maximum(lengths[index], 1e-8))[:, None]


def box_signed_distance(points, transform, size):
    import numpy as np
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    q = np.abs(local) - size/2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def box_occludes(points, projection, transform, size):
    import numpy as np
    camera = -np.linalg.solve(projection[:, :3], projection[:, 3])
    origin = (camera - transform[:3, 3]) @ transform[:3, :3]
    direction = (points - camera) @ transform[:3, :3]
    direction = np.where(np.abs(direction) < 1e-10, 1e-10, direction)
    first = (-size/2 - origin) / direction
    last = (size/2 - origin) / direction
    near = np.minimum(first, last).max(axis=-1)
    far = np.maximum(first, last).min(axis=-1)
    return (far >= np.maximum(near, 0)) & (near > 0) & (near < .985)
