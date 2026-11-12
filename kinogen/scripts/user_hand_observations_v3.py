"""Image-only hand evidence with explicit robot/box occlusion ownership.

No model projection participates in pixel selection. Near-hand ROI and wrist
come from the existing approximate manual/LK image observations.
"""
def make_observation(image,robot,box,valid,palm,wrist):
 import cv2,numpy as np
 if not np.isfinite(palm).all() or not np.isfinite(wrist).all():return None
 x,y=palm;x0,x1=max(0,int(x)-40),min(640,int(x)+40);y0,y1=max(0,int(y)-34),min(720,int(y)+31)
 roi=np.zeros((720,640),bool);roi[y0:y1,x0:x1]=True
 yy,xx=np.indices((720,640));direction=(palm-wrist)/max(np.linalg.norm(palm-wrist),1)
 domain=roi&valid&(((xx-wrist[0])*direction[0]+(yy-wrist[1])*direction[1])>2.)
 hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);metal=(hsv[:,:,1]<65)&(hsv[:,:,2]>45)&(hsv[:,:,2]<245)
 # Gray metal connected to the source robot may recover SAM cut boundaries,
 # while brown cardboard cannot claim source robot pixels as occluders.
 nearby=cv2.dilate(robot.astype(np.uint8),np.ones((3,3),np.uint8))>0
 hand=metal&nearby&domain
 hand=cv2.morphologyEx(hand.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
 count,labels=cv2.connectedComponents(hand.astype(np.uint8));seed=robot&hand
 keep=[i for i in range(1,count)if np.sum(seed&(labels==i))>=5]
 hand=np.isin(labels,keep)&domain
 if hand.sum()<40:return None
 silhouette=cv2.morphologyEx(hand.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
 gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);edges=cv2.Canny(gray,40,110)>0
 border=cv2.dilate((~domain).astype(np.uint8),np.ones((5,5),np.uint8))>0
 observed=(silhouette|(edges&hand))&domain&~border
 ys,xs=np.nonzero(observed);all_points=np.column_stack((xs,ys)).astype(float)
 if len(xs)<12:return None
 held=(xs//6+ys//6)%5==0;train=observed.copy();train[ys[held],xs[held]]=False
 points=all_points[~held];points=points[np.linspace(0,len(points)-1,min(72,len(points))).astype(int)]
 return dict(distance=cv2.distanceTransform((~train).astype(np.uint8),cv2.DIST_L2,3),outside=cv2.distanceTransform((~hand).astype(np.uint8),cv2.DIST_L2,3),points=points,training_all_points=all_points[~held],heldout_points=all_points[held],all_points=all_points,edge_mask=observed,hand_mask=hand,domain=domain,valid=valid,robot=robot,box=box,roi=roi,contact_hint=False,palm=np.asarray(palm),wrist=np.asarray(wrist),bounds=np.asarray([x0,y0,x1,y1]))
