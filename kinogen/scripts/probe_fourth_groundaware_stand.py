from pathlib import Path
import argparse
from fourth_single_plate_benchmark_groundaware import prepare
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--setup',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
state=prepare(a.setup,a.output,None,'pd_groundaware')
print('GROUND-AWARE TWO SECOND FREE-ROOT STAND PASSED',flush=True)
