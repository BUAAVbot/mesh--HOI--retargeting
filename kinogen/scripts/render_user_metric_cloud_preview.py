#!/usr/bin/env python3
"""Fixed metre-scale workspace crop for candidate point-cloud visual review."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]


def render_preview(clouds,path):
    import numpy as np
    from PIL import Image,ImageDraw
    bounds=np.asarray([[-.6,1.2],[-.8,.8],[-.2,1.6]])
    image=Image.new('RGB',(1440,1040),'#101827');draw=ImageDraw.Draw(image)
    counts=[]
    for column,(frame,points,colors) in enumerate(clouds):
        keep=np.isfinite(points).all(axis=1)&((points>=bounds[:,0])&(points<=bounds[:,1])).all(axis=1)
        p=points[keep];c=colors[keep];counts.append({'frame':int(frame),'full_cloud_points':len(points),'shown_in_fixed_workspace_crop':len(p)})
        for row,(horizontal,label) in enumerate(((1,'Front: Y/Z'),(0,'Side: X/Z'))):
            ox,oy=480*column,520*row;center=(bounds[horizontal].sum()/2,bounds[2].sum()/2);scale=420/max(np.ptp(bounds[horizontal]),np.ptp(bounds[2]))
            def screen(h,z):return (int(ox+240+(h-center[0])*scale),int(oy+290-(z-center[1])*scale))
            for z in (0.,.5,1.,1.5):
                a=screen(bounds[horizontal,0],z);b=screen(bounds[horizontal,1],z)
                draw.line((a,b),fill='#334155',width=1);draw.text((ox+5,a[1]-6),f'{z:g}m',fill='#94a3b8')
            if len(p):
                order=np.argsort(p[:,0] if horizontal==1 else -p[:,1])
                for i in order:
                    x,y=screen(p[i,horizontal],p[i,2]);color=tuple(map(int,c[i]))
                    if ox+30<x<ox+475 and oy+70<y<oy+500:draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
            draw.text((ox+12,oy+10),f'Frame {frame:03d} | {label}',fill='white')
            draw.text((ox+12,oy+28),'CANDIDATE depth; not metric ground truth',fill='#cbd5e1')
            draw.text((ox+12,oy+46),'Fixed workspace crop; full PLY retained',fill='#cbd5e1')
    path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=94)
    return {'world_crop_xyz_m':bounds.tolist(),'same_scale_and_bounds_for_all_frames':True,'full_ply_files_unchanged':True,'counts':counts}


def main():
    import numpy as np
    out=ROOT/'outputs/user_reconstruction';clouds=[]
    for frame in (0,60,120):
        path=out/f'cloud_frame_{frame:03d}.ply'
        with path.open('rb') as f:
            header=[]
            while True:
                line=f.readline().decode('ascii').strip();header.append(line)
                if line=='end_header':break
            count=int(next(line.split()[-1] for line in header if line.startswith('element vertex ')))
            data=np.frombuffer(f.read(),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('view_id','u1')],count=count)
        clouds.append((frame,np.column_stack([data[n] for n in ('x','y','z')]),np.column_stack([data[n] for n in ('red','green','blue')])))
    path=ROOT/'outputs/previews/user_metric_pointcloud.jpg';meta=render_preview(clouds,path)
    reportpath=ROOT/'reports/user-metric-depth.json';report=json.loads(reportpath.read_text())
    report['pointcloud_preview_crop']=meta
    for record in report['previews']:
        if record['path']==str(path):record.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    reportpath.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
