#!/bin/bash

# Training
#~/VC/Experiments/wav2letter/kenlm/build/bin/lmplz -o 3 --verbose_header --text ../data/splitted/train.txt --arpa ./init.arpa -S 10%

# Compression
~/VC/Experiments/wav2letter/kenlm/build/bin/build_binary -s ./init.arpa init.bin
