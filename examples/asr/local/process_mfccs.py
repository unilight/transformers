#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from io import open
import json
import logging
import sys

import h5py
from itertools import zip_longest
from kaldiio import ReadHelper, WriteHelper
import numpy as np
import os
from tqdm.auto import tqdm, trange


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process the MFCCs using an alignment file and save them in hdf5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alignment_file", type=str, help="Alignment file")
    parser.add_argument("--mfcc_scp", type=str, help="Scp file of MFCCs")
    parser.add_argument("--frame_shift", type=float, default=0.01, help="Frame shift of MFCC")
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument("--output_dir", type=str,help="The output directory")
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

    # read alignments first
    alignments = {}
    with open(args.alignment_file, "r") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    for line in lines:
        fileid, number, start, duration, word = line.split(" ")[:5]
        if fileid not in alignments:
            alignments[fileid] = []
        alignments[fileid].append([start, duration])


    # Read MFCCs from scp, and use alignment to split to word-level
    # Return a dict: {key: [mfccs for token 1, mfccs for token 2, ...]}
    with ReadHelper("scp:" + args.mfcc_scp) as reader:
        for fileid, mfccs in reader:
            num_steps = len(alignments[fileid])
            num_frames = mfccs.shape[0]
            mfcc_list = []
            masks = np.zeros((num_steps, num_frames), np.int8)
            for step, (start, duration) in enumerate(alignments[fileid]):
                start_idx = int(float(start) / args.frame_shift)
                end_idx = start_idx + int(float(duration) / args.frame_shift)
                current_mfccs = mfccs[start_idx:end_idx]
                masks[step][start_idx:end_idx] = 1
                mfcc_list.append(current_mfccs)

            """ Process the MFCC list and pad each MFCC block to the same length.
                The final processed MFCC array has shape (text_length, max_frames, feature_dim)
            """
            max_length = max([_mfccs.shape[0] for _mfccs in mfcc_list])
            padded_mfccs_list = [np.pad(_mfccs, ((0, max_length - _mfccs.shape[0]), (0, 0)), mode="constant") for _mfccs in mfcc_list]
            padded_mfccs = np.stack(padded_mfccs_list, axis=0)

            """ Save """
            with h5py.File(os.path.join(args.output_dir, fileid + ".hdf5"), "w") as f:
                f.create_dataset("raw_mfccs", data=mfccs)
                f.create_dataset("padded_mfccs", data=padded_mfccs)
                f.create_dataset("masks", data=masks)
