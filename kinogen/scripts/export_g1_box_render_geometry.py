"""Export actual spawned visual geometry, retaining every articulated hand link."""
import json
from pathlib import Path
import numpy as np

def export(raw, robot, obj, output):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage=raw.sim.stage
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    target=Path(output)/"render-geometry"
    target.mkdir(exist_ok=True)
    origin=raw.scene.env_origins[0].detach().cpu().numpy()
    rows=[]
    robot_root=stage.GetPrimAtPath("/World/envs/env_0/Robot")
    by_name={p.GetName():p for p in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(robot.body_names).issubset(by_name), (set(robot.body_names)-set(by_name))
    def color(prim,default):
        attr=UsdGeom.Gprim(prim).GetDisplayColorAttr()
        val=attr.Get() if attr else None
        if val: return list(val[0])+[1.]
        mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            for p in Usd.PrimRange(mat.GetPrim()):
                shader=UsdShade.Shader(p)
                inp=shader.GetInput("diffuseColor") if shader else None
                if inp and inp.Get() is not None:
                    return list(inp.Get())+[1.]
        return default
    def meshes(root,group,body=None):
        inv=cache.GetLocalToWorldTransform(root).GetInverse()
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if any(x in str(prim.GetPath()).lower() for x in ("collision","/looks/")):
                continue
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
                continue
            image=UsdGeom.Imageable(prim)
            if image.ComputeVisibility()==UsdGeom.Tokens.invisible: continue
            if prim.IsA(UsdGeom.Mesh):
                schema=UsdGeom.Mesh(prim)
                points=schema.GetPointsAttr().Get()
                counts=schema.GetFaceVertexCountsAttr().Get()
                indices=schema.GetFaceVertexIndicesAttr().Get()
                if not points or not counts: continue
                faces=[]; offset=0
                for n in counts:
                    face=indices[offset:offset+n]
                    faces.extend([[int(face[0]),int(face[k]),int(face[k+1])] for k in range(1,n-1)])
                    offset+=n
            else:
                size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
                points=[Gf.Vec3f(x*size,y*size,z*size) for x,y,z in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
                faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5]]
            transform=cache.GetLocalToWorldTransform(prim)
            transform=transform*inv
            verts=np.array([list(transform.Transform(Gf.Vec3d(*p))) for p in points])
            filename=f"mesh_{len(rows):04d}.obj"
            with open(target/filename,"w") as f:
                for vertex in verts: f.write("v "+" ".join(map(str,vertex))+"\n")
                for face in faces: f.write("f "+" ".join(str(i+1) for i in face)+"\n")
            default=[.64,.67,.70,1.] if group=="robot" else ([.75,.08,.04,1.] if group=="object" else [.58,.42,.27,1.])
            rows.append(dict(file=filename,group=group,body=body,source_prim=str(prim.GetPath()),rgba=color(prim,default),vertices=len(verts),triangles=len(faces)))
    for name in robot.body_names:
        meshes(by_name[name],"robot",name)
    obj_path=obj.root_physx_view.prim_paths[0]
    meshes(stage.GetPrimAtPath(obj_path),"object")
    table_roots=[stage.GetPrimAtPath(raw.scene["table"].root_physx_view.prim_paths[0])]
    assert len(table_roots)==1, [str(p.GetPath()) for p in table_roots]
    meshes(table_roots[0],"table")
    manifest=dict(source="Actual live Isaac stage visuals",rows=rows,robot_body_names=list(robot.body_names),
        table_root=(raw.scene["table"].data.root_pos_w[0].detach().cpu().numpy()-origin).tolist(),
        table_quat=raw.scene["table"].data.root_quat_w[0].detach().cpu().tolist(),
        all_hand_links_preserved=all(any(r["body"]==n for r in rows) for n in robot.body_names if "hand_" in n and n.endswith("_link")))
    assert manifest["all_hand_links_preserved"], "Missing visual hand geometry"
    assert any(r["group"]=="object" for r in rows)
    assert any(r["group"]=="table" for r in rows)
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest
