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
    raw_mfccs_dict = {}
    with ReadHelper("scp:" + args.mfcc_scp) as reader:
        for fileid, mfccs in reader:
            mfcc_list = []
            for start, duration in alignments[fileid]:
                start_idx = int(float(start) / args.frame_shift)
                end_idx = start_idx + int(float(duration) / args.frame_shift)
                mfcc_list.append(mfccs[start_idx:end_idx])
            raw_mfccs_dict[fileid] = mfcc_list

    # Process the MFCC list.
    # Each processed MFCC array has shape (text_length, max_frames, feature_dim)}
    processed_mfccs_dict = {}
    for fileid, raw_mfccs in tqdm(raw_mfccs_dict.items()):
        # """ 1. Take the last word to the first word. (current frame) """
        # reversed_mfccs = [mfccs[-1]] + mfccs[:-1]

        """ 2. Pad to the same length. """
        max_length = max([mfccs.shape[0] for mfccs in raw_mfccs])
        padded_mfccs_list = [np.pad(mfccs, ((0, max_length - mfccs.shape[0]), (0, 0)), mode="constant") for mfccs in raw_mfccs]
        padded_mfccs = np.stack(padded_mfccs_list, axis=0)

        # """ 3. Add one more empty frame (correspond to [SEP]) """
        # appended_mfccs = np.pad(padded_mfccs, ((0, 1), (0, 0), (0, 0)), mode="constant")

        """ Save """
        with h5py.File(os.path.join(args.output_dir, fileid + ".hdf5"), "w") as f:
            f.create_dataset("mfccs", data=padded_mfccs)
            #f.create_dataset("mfccs", data=appended_mfccs)

