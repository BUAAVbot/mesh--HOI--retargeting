from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/fourth_single_plate_benchmark_setup.py').read_text()
source=source.replace('from fourth_sonic_controller import Controller', 'from fourth_sonic_controller import Controller\nfrom fourth_ground_reaction_pd_controller import GroundReactionPD')
needle="    spec = int(mj.mjtState.mjSTATE_INTEGRATION)"
insertion="""    ground_controller = None
    if body_control == 'pd_groundaware':
        ground_controller = GroundReactionPD(m, q0, sonic.aids, sonic.qa, sonic.va)
        sonic.action = ground_controller.action
        sonic.ground_controller = ground_controller
"""
assert needle in source
source=source.replace(needle,insertion+needle)
source=source.replace("    (output/'settle_report.json').write_text", "    if ground_controller is not None:\n        (output/'ground_reaction_control.json').write_text(json.dumps(ground_controller.report(),indent=2)+'\\n')\n    (output/'settle_report.json').write_text")
target=ROOT/'scripts/fourth_single_plate_benchmark_groundaware.py'
target.write_text(source)
print(target)
