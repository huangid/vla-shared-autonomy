"""Merge kNN-format .npy demo files (and their .tasks.json sidecars) into one.

Used to mix correction episodes with a slice of the original demos: finetuning on
corrections alone measurably degrades the policy (see docs/RESULTS.md section 9), so
both arms of the D_human / D_blend comparison need the same base-demo ballast.

Episodes are renumbered sequentially; each sidecar entry keeps its own `frames` path,
so npy_to_lerobot still finds the right images per episode.

    python scripts/merge_npy_datasets.py \\
        --inputs logs/data/corrections_human.npy logs/data/randomblock_demos.npy \\
        --take all 100 --seed 0 \\
        --output logs/data/mixed_human.npy

`--take` accepts "all" or an episode count per input; a count samples WITHOUT
replacement using `--seed`, so passing the same seed to every merge draws the same
base episodes — necessary, or the two arms differ by more than the labels under test.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input .npy files, in order.")
    p.add_argument("--take", nargs="+", default=None,
                   help="Per input: 'all' or an episode count. Default: all of each.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed; keep it fixed across merges.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    takes = args.take or ["all"] * len(args.inputs)
    if len(takes) != len(args.inputs):
        raise SystemExit(f"--take has {len(takes)} entries for {len(args.inputs)} inputs")

    out, out_meta, n = {}, {}, 0
    for src, take in zip(args.inputs, takes):
        data = np.load(src, allow_pickle=True).item()
        meta_path = Path(str(src) + ".tasks.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        keys = sorted(data)
        if take != "all":
            k = int(take)
            if k > len(keys):
                raise SystemExit(f"{src} has {len(keys)} episodes, --take {k} requested")
            keys = sorted(random.Random(args.seed).sample(keys, k))

        for old in keys:
            out[n] = data[old]
            m = meta.get(str(old))
            if m is not None:
                out_meta[n] = {**m, "merged_from": str(src), "merged_episode": int(old)}
            n += 1
        print(f"  {src}: took {len(keys)} of {len(data)} episodes")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out, allow_pickle=True)
    Path(str(args.output) + ".tasks.json").write_text(
        json.dumps({str(k): v for k, v in sorted(out_meta.items())}, indent=2))
    frames = sum(len(v[next(iter(v))]) for v in out.values())
    print(f"Wrote {len(out)} episodes ({frames} frames) to {args.output}")
    if len(out_meta) != len(out):
        print(f"[warn] only {len(out_meta)}/{len(out)} episodes carry sidecar metadata; "
              f"npy_to_lerobot --images needs it for the task string and frame paths")


if __name__ == "__main__":
    main()
