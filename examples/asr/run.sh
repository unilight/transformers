#!/bin/bash

# . ./path.sh || exit 1;
. ./cmd.sh || exit 1;

pip install -e ../../ || exit 1;

# general
stage=-1
stop_stage=100
save_steps=50000
logging_steps=1000
n_jobs=5      # number of parallel jobs in feature extraction
debug=no

# data related
data_dir=data
output_dir=output
alignment_dir=data/new_alignment
#alignment_dir=data/alignment
mfcc_dir=data/mfcc
processed_mfcc_dir=data/processed_mfcc

# training related
max_seq_length=46
bs=1
accum_grad=8
lr=5e-5
freeze_mods=
use_audio=
exhaustion=
fusion_place="first"
acoustic_encoder_type="conv"
acoustic_encoder_segment="first"
acoustic_encoder_layers=1

# model related
model_type=bert
model=hfl/chinese-bert-wwm

# decoding related
ppl=no
model_dir=
beam_size=10
set_type="dev"
lm_weight=0.0
lm_model_path=rnnlm/init.bin

# exp tag
tag="default"  # tag for managing experiments.

. ./parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python run_language_modeling.py \
        --output_dir=${output_dir} \
        --model_type=${model_type} \
        --model_name_or_path=${model_name_or_path} \
        --do_train --do_eval \
        --overwrite_cache \
        --data_dir=${data_dir}/cleaned
        #--train_data_file=${data_dir}/cleaned/train_clean.txt \
        #--do_eval --eval_data_file=${data_dir}/cleaned/dev_clean.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python perplexity.py \
        --output_dir=${output_dir} \
        --model_dir=${model_dir} \
        --dev_data_file="${data_dir}/cleaned/test.txt"

    python perplexity.py \
        --output_dir=${output_dir} \
        --model_dir=${model_dir} \
        --dev_data_file="${data_dir}/cleaned/dev.txt"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Data preparation"

    echo "Further segment word level alignment into char level alignment"
    for set_type in train dev test; do
        echo "Set: ${set_type}"
        local/segment-alignment.py \
            --alignment_file ${alignment_dir}/${set_type}_word \
            --out ${alignment_dir}/${set_type}
    done

    echo "Process MFCCs"
    for set_type in train dev test; do
        echo "Set: ${set_type}"
        mkdir -p ${processed_mfcc_dir}/${set_type}
        local/process_mfccs.py \
            --alignment_file ${alignment_dir}/${set_type} \
            --mfcc_scp ${mfcc_dir}/${set_type}_original \
            --output_dir ${processed_mfcc_dir}/${set_type}
    done
fi


if [ -z ${use_audio} ]; then
    echo "Please specify --use_audio ."
    exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Train ASR model"

    expdir=exp/${tag}

    python asr.py \
        --acoustic_encoder_layers=${acoustic_encoder_layers} \
        --acoustic_encoder_segment=${acoustic_encoder_segment} \
        --acoustic_encoder_type=${acoustic_encoder_type} \
        --fusion_place=${fusion_place} \
        --exhaustion=${exhaustion} \
        --use_audio=${use_audio} \
        --save_steps=${save_steps} \
        --logging_steps=${logging_steps} \
        --max_seq_length=${max_seq_length} \
        --per_device_train_batch_size=${bs} \
        --learning_rate=${lr} \
        --gradient_accumulation_steps=${accum_grad} \
        --freeze_mods=${freeze_mods} \
        --debugging=${debug} \
        --output_dir=${expdir} --overwrite_output_dir \
        --model_type=${model_type} \
        --model_name_or_path=${model} \
        --do_train --overwrite_cache \
        --data_dir=${data_dir}/original \
        --mfcc_dir=${processed_mfcc_dir}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: ASR decoding"

    if [ -z ${model_dir} ]; then
        expdir=exp/${tag}
    else
        expdir=${model_dir}
    fi

    cp exp/text-only-exhaustive/vocab.txt ${expdir}

    python decode.py \
        --acoustic_encoder_layers=${acoustic_encoder_layers} \
        --acoustic_encoder_type=${acoustic_encoder_type} \
        --fusion_place=${fusion_place} \
        --do_perplexity=${ppl} \
        --use_audio=${use_audio} \
        --set_type=${set_type} \
        --max_seq_length=${max_seq_length} \
        --output_dir=${expdir} \
        --model_type=${model_type} \
        --model_name_or_path=${expdir} \
        --overwrite_cache \
        --data_dir=${data_dir}/original \
        --mfcc_dir=${processed_mfcc_dir}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Multiprocess ASR decoding"

    if [ -z ${model_dir} ]; then
        expdir=exp/${tag}
    else
        expdir=${model_dir}
    fi

    cp exp/text-only-exhaustive/vocab.txt ${expdir}

    # Decide log file
    if [ ${ppl} = "yes" ]; then
        log_name=ppl
    else
        log_name=decoding
    fi

    # Actual decoding
    [ ! -e "${expdir}/${set_type}/log" ] && mkdir -p "${expdir}/${set_type}/log"
    local/make_subset_data.sh "${data_dir}/original/${set_type}.txt" "${n_jobs}" "${expdir}/${set_type}"
    echo $(date) "Decoding..."
    ${train_cmd} JOB=1:${n_jobs} "${expdir}/${set_type}/log/${log_name}.JOB.log" \
        python decode.py \
            --acoustic_encoder_segment=${acoustic_encoder_segment} \
            --acoustic_encoder_type=${acoustic_encoder_type} \
            --fusion_place=${fusion_place} \
            --do_perplexity=${ppl} \
            --use_audio=${use_audio} \
            --set_type=${set_type} \
            --max_seq_length=${max_seq_length} \
            --output_dir=${expdir} \
            --model_type=${model_type} \
            --model_name_or_path=${expdir} \
            --overwrite_cache \
            --scp "${expdir}/${set_type}/JOB.scp" \
            --mfcc_dir=${processed_mfcc_dir} \
            --result_path="${expdir}/${set_type}/asr_results.JOB.txt" \
            --lm_weight=${lm_weight} \
            --lm_model_path=${lm_model_path}

    if [ ${ppl} = "yes" ]; then
        for i in $(seq 1 ${n_jobs}); do
            grep 'ppl' ${expdir}/${set_type}/log/${log_name}.${i}.log | awk '{print $10}'
        done
    fi

    echo "Decoding finished."
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Error rate calculation"

    if [ -z ${model_dir} ]; then
        expdir=exp/${tag}
    else
        expdir=${model_dir}
    fi

    cat ${expdir}/${set_type}/asr_results.*.txt > ${expdir}/${set_type}/asr_results.txt

    python local/result2trn.py \
        --result_file ${expdir}/${set_type}/asr_results.txt \
        --refs ${expdir}/${set_type}/ref.trn \
        --hyps ${expdir}/${set_type}/hyp.trn

    /home/huang18/VC/Experiments/espnet/tools/kaldi/tools/sctk-2.4.10/bin/sclite \
        -r ${expdir}/${set_type}/ref.trn trn \
        -h ${expdir}/${set_type}/hyp.trn \
        -i rm -o all stdout > ${expdir}/${set_type}/result.txt

    grep -e Avg -e SPKR -m 2 ${expdir}/${set_type}/result.txt

fi
