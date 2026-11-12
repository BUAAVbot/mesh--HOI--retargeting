#!/usr/bin/env python3
"""SHA-bound full-scene visibility diagnostic; does not alter frozen v3 audit."""
from pathlib import Path
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    from verify_user_reconstruction_v3 import sha,record,require
    sp=ROOT/'reports/user-full-scene-v3.json';ap=ROOT/'reports/user-v3-independent-audit.json'
    s=json.loads(sp.read_text());a=json.loads(ap.read_text())
    require(a['scene_exporter_diagnostics']['source']['sha256']==sha(sp),'Scene report differs from frozen CPU audit')
    for b in s['outputs']:require(sha(ROOT/b['path'])==b['sha256'],'Rendered artifact changed')
    rows=[{'frame':r['frame'],**v}for r in s['per_frame_mesh_audit']for v in r.get('visible_hand_geometry_comparison',[])if v is not None]
    missing=[{'frame':r['frame'],'side':r['side'],'source_pixels':r['source_pixels'],'heldout_external_edge_points':r['heldout_external_edge_points'],'candidate_visible_pixels_in_domain':r['candidate']['visible_hand_pixels_in_domain']}for r in rows if r['candidate']['missing_visible_model_contour']]
    sides={}
    for side in ('left','right'):
        rr=[r for r in rows if r['side']==side];summary={'observed_near_view_frames':len(rr)}
        for kind in ('frozen_v2','candidate'):
            ee=[r[kind]['external_heldout_mean_px']for r in rr if r[kind]['external_heldout_mean_px']is not None]
            summary[kind]={'median_visible_mask_iou':statistics.median([r[kind]['visible_mask_iou_in_source_domain']for r in rr])if rr else None,'median_external_heldout_mean_px_when_contour_exists':statistics.median(ee)if ee else None,'frames_with_external_error':len(ee),'missing_visible_model_contour_frames':[r['frame']for r in rr if r[kind]['missing_visible_model_contour']]}
        sides[side]=summary
    f54=[r for r in rows if r['frame']==54 and r['side']=='right']
    require(len(f54)==1,'Expected actual frame54 right hand observation')
    report={'status':'visible_hand_image_consistency_failed'if missing else'visible_hand_missing_contour_check_passed','source_scene_report':record(sp),'frozen_cpu_audit':record(ap),'verification_script':record(Path(__file__)),'frame54_right':f54[0],'summary_by_side':sides,'renderer_original_summary':s['visible_hand_comparison_summary'],'missing_visible_hand_observations':missing,'visible_hand_presence_consistency_passed':not missing,'necessary_geometric_checks_passed':a['audited_necessary_physical_checks_passed'],'overall_reconstruction_success':False,'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'scope':'This separately freezes v3 diagnostic evidence. Source observations are RGB-only; source ROI/pixel selection never uses predicted model. Visibility comes from the complete scene z-buffer including box/table and robot self-occlusion. It is SHA-bound exporter evidence, not a new rendering or optimization.','interpretation':'A projected per-link proxy can fit RGB while being occluded in the actual scene. Zero hand-box penetration and small proxy residuals therefore do not establish reconstructed visible hands or a valid grasp. Missing predicted external contours with observed RGB edges must remain image consistency failures.','median_warning':'External-error medians omit absent model contours; missing counts/lists are reported explicitly and cannot be treated as zero error or hidden by an improved median.','cause_scope':'Back-face/occlusion is a plausible diagnosis, but this report establishes missing visible model support; it does not infer uniquely observed3D hand positions from RGB.'}
    p=ROOT/'reports/user-v3-visible-hand-diagnostic.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(p),'frame54_right':f54[0],'summary_by_side':sides,'visible_hand_presence_consistency_passed':not missing},indent=2))

if __name__=='__main__':main()
