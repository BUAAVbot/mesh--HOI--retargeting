import sys, json, glob
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

def read(path):
    ds = datastore.DataStore(); ds.open_for_scan(path)
    rows = []
    while True:
        try:
            d = ds.scan_data()
        except Exception as e:
            break
        if d is None: break
        r = pb.Record(); r.ParseFromString(d)
        if r.WhichOneof("record_type") == "history":
            row = {}
            for it in r.history.item:
                k = it.key or "/".join(it.nested_key)
                try: row[k] = json.loads(it.value_json)
                except Exception: pass
            rows.append(row)
    return rows

keys = sys.argv[2].split(",") if len(sys.argv) > 2 else None
for run in sys.argv[1].split(","):
    f = glob.glob(f"{run}/wandb/offline-run-*/run-*.wandb")
    if not f: print(run, "no wandb"); continue
    rows = read(sorted(f)[-1])
    print("==", run, "rows", len(rows))
    if not rows: continue
    last = rows[-1]
    if keys is None:
        print(sorted(last.keys())); continue
    for i in sorted(set([0, len(rows)//4, len(rows)//2, len(rows)-1])):
        r = rows[i]
        print(i, {k: (round(v,4) if isinstance(v,(int,float)) else v) for k, v in r.items() if any(s in k for s in keys)})
