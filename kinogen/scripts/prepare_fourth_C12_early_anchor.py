from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_o6_C11_actual_segment.py').read_text()
point=" all_contact=all(Fdigit[v]>.05 for v in required)"
replacement=""" pair_dot=min((float(n1@n2)for v1,n1,f1 in normals for v2,n2,f2 in normals if v1=='thumb'and v2!='thumb'and min(f1,f2)>.05),default=1.)
 pair_anchor=Fdigit['thumb']>.1 and sum(Fdigit[v]for v in digits if v!='thumb')>.2 and pair_dot<=-.25
 if locked_palm is None and pair_anchor:locked_palm=(arm.palm(d,1).copy(),d.xmat[arm.hand_bids[1]].reshape(3,3).copy())
 all_contact=all(Fdigit[v]>.05 for v in required)"""
assert point in text
text=text.replace(point,replacement)
text=text.replace(" if recent:lost_since=None", " if pair_anchor or recent:lost_since=None")
text=text.replace("'palm_loading_policy':'Lock actual world palm/R at first actual full-wrench feasible contact; release latch after150ms real contact loss'",
                  "'palm_loading_policy':'Anchor actual world palm/R after opposing thumb-bank low-preload contact; this does not certify force closure or permit lifting; full actual LP and measured-force proof retained'")
target=ROOT/'scripts/run_fourth_o6_C12_early_anchor.py'
target.write_text(text)
print(target)
