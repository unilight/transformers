#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    last = last[-args.num :]
    print("average over", last)
    avg = None

    if args.backend == "pytorch":
        import torch

        # sum
        for path in last:
            states = torch.load(path, map_location=torch.device("cpu"))
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    if "weight" in k or "bias" in k:
                        avg[k] += states[k]

        # average
        for k in avg.keys():
            if "weight" in k or "bias" in k:
                if avg[k] is not None:
                    avg[k] /= args.num

        torch.save(avg, args.out)

    else:
        raise ValueError("Incorrect type of backend")


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default="pytorch", type=str)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
