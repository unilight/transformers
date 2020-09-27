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
        description="Split text file for KenLM training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text_file", type=str, help="Text file")
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
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

    with open(args.text_file, "r") as f:
        lines = [
            line
            for line in f.read().splitlines()
            if (len(line) > 0 and not line.isspace())
        ]

    for line in lines:
        print(' '.join(line))
