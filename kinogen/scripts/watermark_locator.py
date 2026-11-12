"""Locate the observed moving overlay using masked, signed high-pass templates."""
import cv2
import numpy as np


class WatermarkLocator:
    def __init__(self, frames):
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        median = np.median(gray, axis=0).astype(np.float32)
        self.templates = []
        for source_frame,roi,kind in ((120,(42,35,205,60),'observed_logo'),):
            source = gray[source_frame].astype(np.float32)
            high = source - cv2.GaussianBlur(source, (0, 0), 3.)
            x,y,w,h = roi
            residual = np.abs(source-median)[y:y+h,x:x+w]
            template = high[y:y+h,x:x+w]
            mask = ((residual > 4) & (np.abs(template) > 2)).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((3,3),np.uint8))
            for scale in (.7, .85, 1., 1.15, 1.3):
                size = (round(w*scale),round(h*scale))
                self.templates.append((cv2.resize(template,size),cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST),scale,kind))

    def locate(self, frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        high = gray-cv2.GaussianBlur(gray,(0,0),3.)
        candidates=[]
        for template,mask,scale,kind in self.templates:
            response=cv2.matchTemplate(high,template,cv2.TM_CCORR_NORMED,mask=mask)
            response=np.nan_to_num(response,nan=-1.,posinf=-1.,neginf=-1.)
            _,score,_,(x,y)=cv2.minMaxLoc(response)
            candidates.append({'score':float(score),'xyxy':[x,y,x+template.shape[1],y+template.shape[0]],'scale':scale,'kind':kind})
        return sorted(candidates,key=lambda v:v['score'],reverse=True)


if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    cap=cv2.VideoCapture(str(root/'data/video_generation/candidate_scene/generated.mp4'))
    frames=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frames.append(f)
    locator=WatermarkLocator(frames)
    rows=[]
    for i in (0,12,24,36,48,60,72,84,96,108,120):
        candidates=locator.locate(frames[i])
        rows.append({'frame':i,'candidates':candidates[:1]})
    print(json.dumps(rows),flush=True)
