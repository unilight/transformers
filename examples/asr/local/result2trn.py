#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from io import open
import json
import logging
import sys

from tqdm.auto import tqdm, trange


def get_parser():
    parser = argparse.ArgumentParser(
        description="Parse asr result to ref and hyp trn files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result_file", type=str, help="Text file")
    parser.add_argument("--refs", type=str, help="Output ref file")
    parser.add_argument("--hyps", type=str, help="Output hyp file")
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)

    with open(args.result_file, "r") as f:
        lines = [
            line
            for line in f.read().splitlines()
            if (len(line) > 0 and not line.isspace())
        ]

    with open(args.refs, "w") as f_refs, open(args.hyps, "w") as f_hyps:
        for i, line in enumerate(lines):
            words = line.split(" ")[-1]
            if line.startswith("Original"):
                f_refs.write("{} (dummy_{})\n".format(' '.join(words), int(i/2)))
            elif line.startswith("Predition"):
                f_hyps.write("{} (dummy_{})\n".format(' '.join(words), int(i/2)))
