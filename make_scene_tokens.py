#!/usr/bin/env python3
"""Write a contiguous slice of nuScenes scene tokens to a text file."""
from __future__ import annotations

import argparse
import os

from nuscenes.nuscenes import NuScenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a nuScenes scene-token slice file.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--start", type=int, required=True, help="Zero-based scene start index.")
    parser.add_argument("--count", type=int, required=True, help="Number of scene tokens to write.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.count < 1:
        raise ValueError("--count must be >= 1")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    end = min(args.start + args.count, len(nusc.scene))
    scenes = nusc.scene[args.start:end]
    if not scenes:
        raise ValueError(
            f"No scenes selected from index {args.start}; dataset has {len(nusc.scene)} scenes."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(scene["token"] + "\n")

    print(
        f"Wrote {len(scenes)} scene tokens "
        f"(1-based scenes {args.start + 1}-{args.start + len(scenes)}) to {args.output}"
    )


if __name__ == "__main__":
    main()
