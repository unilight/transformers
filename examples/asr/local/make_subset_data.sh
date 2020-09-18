#!/bin/bash

# Make subset files located in data direcoty.

# Copyright 2020 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)


if [ $# -ne 3 ]; then
    echo "Usage: $0 <text_txt> <num_split> <dst_dir>"
    echo "e.g.: $0 data/train_nodev 16 data/train_nodev/split16"
    exit 1
fi

set -eu

text_txt=$1
num_split=$2
dst_dir=$3

[ ! -e "${dst_dir}" ] && mkdir -p "${dst_dir}"

num_utts=$(wc -l < "${text_txt}")

split_scps=""
for i in $(seq 1 "${num_split}"); do
    split_scps+=" ${dst_dir}/${i}.scp"
done
utils/split_scp.pl "${text_txt}" ${split_scps}
echo "Successfully make subsets."
