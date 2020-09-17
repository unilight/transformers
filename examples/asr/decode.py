# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Beam search decoding for ASR.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm, trange
import torch

from transformers import ASRDataTrainingArguments as DataTrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    ASRDataset,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    BertForASR,
    DataCollatorForASR,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    acoustic_encoder_type: Optional[str] = field(
        default="conv", metadata={"help": "Acoustic encoder type."}
    )

@dataclass
class DecodeArguments:
    """
    Arguments related to decoding.
    """
    set_type: str = field(default="dev", metadata={"help": "Whether to run perplexity."})
    do_perplexity: str = field(default="no", metadata={"help": "Whether to run perplexity."})
    beam_size: int = field(default=10, metadata={"help": "Beam size."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DecodeArguments)
    )
    model_args, data_args, training_args, decode_args = parser.parse_args_into_dataclasses()
    
    if model_args.model_name_or_path is None:
        raise ValueError(
            "Please specify --model_dir."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = config.vocab_size
    config.use_audio = True if data_args.use_audio == "yes" else False
    config.fusion_place = data_args.fusion_place
    config.acoustic_encoder_type = model_args.acoustic_encoder_type

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForASR.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    model.eval()

    # Get datasets
    logger.info("*** Building Evaluation dataset ***")
    eval_dataset = (
        ASRDataset(data_args, use_audio=config.use_audio, tokenizer=tokenizer, cache_dir=model_args.cache_dir,
            mode=decode_args.set_type) # TODO: change this to an arg
    )
    
    if decode_args.do_perplexity != "no":
        # Calculate perplexity
        logger.info("*** Calculate perplexity ***")
        
        all_ppl=[]
        for example in tqdm(eval_dataset):
            lls = []
            decode_id=[]
            text_length = np.count_nonzero(example.input_ids)-1
            if config.use_audio:
                o_mfccs = example.mfccs
                #o_mfccs = np.concatenate([example.mfccs[1:text_length], example.mfccs[0][np.newaxis,...]], axis=0)
                max_frame, dimension = o_mfccs.shape[1:]
                if o_mfccs.shape[0] != text_length:
                    #print(o_mfccs.shape[0], text_length)
                    #print("original:  ", tokenizer.decode(example.input_ids[1:text_length]+[example.label]).replace(" ", ""))
                    #print(example.input_ids)
                    continue
            
            for i in range(text_length):
                attention_mask = torch.ones(i+2, dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = torch.zeros(i+2, dtype=torch.long).unsqueeze(0).to(device)

                input_ids = torch.tensor(
                    example.input_ids[:i+1] + [example.input_ids[text_length]]
                ).unsqueeze(0).to(device)
                if config.use_audio:
                    if i ==0:
                        mfccs = torch.tensor(
                            np.concatenate([o_mfccs[i][np.newaxis,...], np.zeros((1, max_frame, dimension))]),
                            dtype=torch.float
                        ).unsqueeze(0).to(device)
                    else:
                        mfccs = torch.tensor(
                            np.concatenate([o_mfccs[i][np.newaxis,...], o_mfccs[:i], np.zeros((1, max_frame, dimension))]),
                            dtype=torch.float
                        ).unsqueeze(0).to(device)
                else:
                    mfccs = None

                if i == text_length-1:
                    label = example.label
                else:
                    label = example.input_ids[i+1]
                label = torch.tensor([label]).to(device)

                with torch.no_grad():
                    outputs = model(input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                        inputs_mfccs = mfccs,
                        labels = label
                    )
                    log_likelihood = outputs[0]
                    max_id = torch.argmax(outputs[1]).detach()
                    decode_id.append(max_id)

                lls.append(log_likelihood)
            #print("original:  ", tokenizer.decode(example.input_ids[1:text_length]+[example.label]).replace(" ", ""))
            #print("predicted: ", tokenizer.decode(decode_id).replace(" ", ""))
            ppl = torch.exp(torch.stack(lls).sum() / text_length-1)
            all_ppl.append(ppl)
        
        logger.info("  %s = %s", "ppl", float(sum(all_ppl) / len(all_ppl)) )
        return
    
    # Decode
    logger.info("*** Decoding start ***")
    
    output_file = os.path.join(training_args.output_dir, "dev_results.txt")
    with open(output_file, "w") as writer:
        for example in tqdm(eval_dataset):
            n_best = []

            text_length = example.mfccs.shape[0]-1
            o_mfccs = np.concatenate([example.mfccs[1:text_length], example.mfccs[0][np.newaxis,...]], axis=0)
            max_frame, dimension = o_mfccs.shape[1:]

            for i in range(0, text_length):
                attention_mask = torch.ones(i+2, dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = torch.zeros(i+2, dtype=torch.long).unsqueeze(0).to(device)
                
                if i == 0:
                    mfccs = torch.tensor(
                        np.concatenate([o_mfccs[i][np.newaxis,...], np.zeros((1, max_frame, dimension))]),
                        dtype=torch.float
                    ).unsqueeze(0).to(device)
                    input_ids = torch.tensor(
                        [example.input_ids[0]] + [example.input_ids[text_length]],
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            inputs_mfccs = mfccs
                        )
                        log_likelihood = np.squeeze(outputs[0].cpu().numpy())
                        n_best_ids = log_likelihood.argsort()[::-1][:decode_args.beam_size]
                        for best_id in n_best_ids:
                            n_best.append(([best_id], log_likelihood[best_id]))

                else:
                    mfccs = torch.tensor(
                        np.concatenate([o_mfccs[i][np.newaxis,...], o_mfccs[:i], np.zeros((1, max_frame, dimension))]),
                        dtype=torch.float
                    ).unsqueeze(0).to(device)
                    new_n_best = []
                    for b in range(decode_args.beam_size):
                        input_ids = torch.tensor(
                            [example.input_ids[0]] + n_best[b][0] + [example.input_ids[text_length]]
                        ).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                inputs_mfccs = mfccs
                            )
                            log_likelihood = np.squeeze(outputs[0].cpu().numpy())
                            n_best_ids = log_likelihood.argsort()[::-1][:decode_args.beam_size]
                            #print(n_best_ids)
                            #print(log_likelihood[n_best_ids])
                            for best_id in n_best_ids:
                                new_n_best.append((n_best[b][0] + [best_id], n_best[b][1]+log_likelihood[best_id]))

                    n_best = sorted(new_n_best, key=lambda x: x[1])[:decode_args.beam_size]
                #print("Time step", i, ", best hypothesis:", tokenizer.decode(n_best[0][0]).replace(" ", ""))


                
            result_tuple=[
                tokenizer.decode(example.input_ids[1:text_length]+[example.label]).replace(" ", ""),
                tokenizer.decode(n_best[0][0]).replace(" ", "")
            ]
            #print("Original:   ", result_tuple[0])
            #print("Prediction: ", result_tuple[1])
            writer.write("Original:   {}\n".format(result_tuple[0]))
            writer.write("Predition:  {}\n".format(result_tuple[1]))
            writer.flush()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
