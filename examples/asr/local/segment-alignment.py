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
        description="Further segment word level alignment into char level alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alignment_file", type=str, help="Alignment file")
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

    with open(args.alignment_file, "r") as f:
        lines = [
            line
            for line in f.read().splitlines()
            if (len(line) > 0 and not line.isspace())
        ]

    # Formet: BAC009S0002W0122 1 0.970 1.090 楼市
    with open(args.out, "w") as f:
        for line in tqdm(lines):
            fileid, number, start, duration, word, _ = line.split(" ")
            word_length = len(word)
            if word_length == 1:
                f.write(line + "\n")
            else:
                char_duration = float(duration) / word_length
                for i in range(word_length):
                    f.write(
                        "{} {} {:.3f} {:.3f} {}\n".format(
                            fileid,
                            number,
                            float(start) + i * char_duration,
                            char_duration,
                            word[i],
                        )
                    )
