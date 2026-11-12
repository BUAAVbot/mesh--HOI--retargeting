"""Build an explicit download plan from locally saved official Hub metadata."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hub_files(repo, component, folder, kind="models", include=None):
    metadata = json.loads((ROOT / "manifests/hub" / (repo.replace("/", "--") + ".json")).read_text(encoding="utf-8"))
    if metadata.get("gated"):
        raise ValueError(f"Refusing to plan gated model without explicit authorized access: {repo}")
    for file in metadata["siblings"]:
        name = file["rfilename"]
        if name == ".gitattributes" or (include and not include(name)):
            continue
        lfs = file.get("lfs") or {}
        yield {
            "name": (folder + "--" + name).replace("/", "--").replace(".", "_"),
            "component": component,
            "url": f"https://huggingface.co/{'spaces/' if kind == 'spaces' else ''}{repo}/resolve/{metadata['sha']}/{name}",
            "path": f"weights/{folder}/{name}",
            "bytes": file.get("size"),
            "sha256": lfs.get("sha256"),
            "role": "Official public artifact; KinoGen checkpoint variant remains unconfirmed",
        }


def main():
    plan = []
    for repo, component, folder in [
        ("nvidia/Kimodo-G1-RP-v1", "Kimodo", "kimodo-g1-rp-v1"),
        ("nvidia/Kimodo-G1-SEED-v1", "Kimodo", "kimodo-g1-seed-v1"),
        ("depth-anything/Video-Depth-Anything-Large", "Video Depth Anything", "video-depth-anything-large"),
        ("nvidia/FoundationPose", "FoundationPose ONNX export (not original PyTorch checkpoint)", "foundationpose-onnx"),
    ]:
        plan.extend(hub_files(repo, component, folder))
    plan.extend(hub_files("rolpotamias/WiLoR", "WiLoR", "wilor", "spaces", lambda name: name.startswith("pretrained_models/") or name == "mano_data/mano_mean_params.npz"))
    for filename, url in [
        ("tapnextpp_ckpt.pt", "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"),
        ("tapnextpp_512.ckpt", "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"),
    ]:
        plan.append({"name": filename.replace(".", "_"), "component": "TAPNext++", "path": f"weights/tapnextpp/{filename}", "url": url, "role": "Official 256/512 variants; KinoGen resolution choice unconfirmed"})
    output = ROOT / "configs/downloads.json"
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Planned {len(plan)} files; known total {sum(item.get('bytes') or 0 for item in plan) / 2**30:.2f} GiB plus TAPNext++")


if __name__ == "__main__":
    main()
