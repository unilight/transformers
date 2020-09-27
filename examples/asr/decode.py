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

import h5py
import kenlm
import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm, trange
from typing import List, NamedTuple
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
    result_path: str = field(metadata={"help": "Path to the result."})
    set_type: str = field(default="dev", metadata={"help": "Whether to run perplexity."})
    do_perplexity: str = field(default="no", metadata={"help": "Whether to run perplexity."})
    beam_size: int = field(default=10, metadata={"help": "Beam size."})
    lm_weight: float = field(default=0.0, metadata={"help": "Language model weight."})
    lm_model_path: str = field(default=None, metadata={"help": "Language model file path."})

class Hypothesis(NamedTuple):
    """ Hypothesis data type.
        Args:
            hyp (str): An hypothesis string, separated by space.
            hyp_ids (list[int]): An hypothesis, which is a list of id.
    """

    score: float = 0.0
    state: str = " "
    state_ids: List[int] = list()

def sort_hyps(hyps):
    #return sorted(hyps, key= lambda k : k.score, reverse=True)
    return hyps.sort(key= lambda k : k[0], reverse=True)

class Scorer:

    def __init__(self, tokenizer, lm, lm_weight, beam_size):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.lm = lm
        self.lm_weight = lm_weight
        self.vocab_list = list(self.vocab.keys())
        self.beam_size = beam_size

    def score(self, bert_log_likelihood, hyp):
        """
            Given current step bert_log_likelihood and an hypothesis,
            compute the lm_log_likelihood and add them up
            to returna a list of SORTED hypotheses.
        """

        if self.lm_weight > 0:
            # Add lm ll with bert ll to form current ll
            current_step_hyps = [None] * len(self.vocab)
            for char, idx in self.vocab.items():
                new_score = hyp[0] + (1-self.lm_weight) * bert_log_likelihood[idx] - self.lm_weight * self.lm.score(new_state, bos = True, eos = False)
                new_state = hyp[1] + char + " "
                new_state_ids = hyp[2] + [idx]
                current_step_hyps[idx] = Hypothesis(
                    # NOTE: Minus the lm score because it is negative
                    score = new_score,
                    state = new_state,
                    state_ids = new_state_ids
                )
            sort_hyps(current_step_hyps)
        else:
            n_best_ids = np.array(bert_log_likelihood).argsort()[::-1][:self.beam_size]
            current_step_hyps = [
                Hypothesis(
                    score = bert_log_likelihood[idx] + hyp[0],
                    state = hyp[1] + self.vocab_list[idx] + " ",
                    state_ids = hyp[2] + [idx]
                ) for idx in n_best_ids
            ]

        # Sort current ll
        # sorted_current_step_hyps = current_step_hyps.copy()
        return current_step_hyps



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
    vocab = tokenizer.get_vocab()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForASR.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    lm = kenlm.Model(decode_args.lm_model_path)

    scorer = Scorer(tokenizer, lm, decode_args.lm_weight, decode_args.beam_size)

    # Get datasets
    logger.info("*** Building Evaluation dataset ***")
    eval_dataset = (
        ASRDataset(data_args, use_audio=config.use_audio, tokenizer=tokenizer, cache_dir=model_args.cache_dir,
            mode=decode_args.set_type) # TODO: change this to an arg
    )

    # Calculate perplexity
    if decode_args.do_perplexity != "no":
        logger.info("*** Calculate perplexity ***")

        all_ppl=[]
        for j, example in tqdm(enumerate(eval_dataset)):
            lls = []
            decode_id=[]
            text_length = example.label_pos
            o_input_ids = torch.tensor(example.input_ids).to(device)
            if config.use_audio:
                with h5py.File(example.mfcc_loader, 'r') as loader:
                    o_mfccs = torch.tensor(loader["mfccs"][()]).to(device)
                max_frame, dimension = o_mfccs.shape[1:]
                if o_mfccs.shape[0] != text_length:
                    #print(o_mfccs.shape[0], text_length)
                    #print("original:  ", tokenizer.decode(example.input_ids[1:text_length]+[example.label]).replace(" ", ""))
                    #print(example.input_ids)
                    continue

            for i in range(text_length):
                attention_mask = torch.ones(i+1, dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = torch.zeros(i+1, dtype=torch.long).unsqueeze(0).to(device)

                input_ids = o_input_ids[:i+1].unsqueeze(0)
                if config.use_audio:
                    if i == 0:
                        mfccs = o_mfccs[i].unsqueeze(0).unsqueeze(0)
                    else:
                        mfccs = torch.cat([o_mfccs[i].unsqueeze(0), o_mfccs[:i]]).unsqueeze(0)
                else:
                    mfccs = None

                label = o_input_ids[i+1]

                #print(" input_ids shape", input_ids.shape)
                #print(" mfccs shape", mfccs.shape)
                #print(" label shape", label.shape)

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
            logger.info("{}/{}".format(j, len(eval_dataset)))
            logger.info("original:  {}".format(tokenizer.decode(example.input_ids[1:text_length+1]).replace(" ", "")))
            logger.info("predicted: {}".format(tokenizer.decode(decode_id).replace(" ", "")))
            ppl = torch.exp(torch.stack(lls).sum() / text_length-1)
            all_ppl.append(ppl)

        logger.info("  %s = %s", "ppl", float(sum(all_ppl) / len(all_ppl)) )
        return

    ################################################
    # Decode
    logger.info("*** Decoding start ***")

    output_file = os.path.join(decode_args.result_path)
    with open(output_file, "w") as writer:
        for j, example in tqdm(enumerate(eval_dataset)):
            n_best = []
            gt_rank=[]

            text_length = example.label_pos
            o_input_ids = example.input_ids
            with h5py.File(example.mfcc_loader, 'r') as loader:
                o_mfccs = torch.tensor(loader["mfccs"][()]).to(device)
            max_frame, dimension = o_mfccs.shape[1:]
            if o_mfccs.shape[0] != text_length:
                #print(o_mfccs.shape[0], text_length)
                #print("original:  ", tokenizer.decode(example.input_ids[1:text_length]+[example.label]).replace(" ", ""))
                #print(example.input_ids)
                continue

            # NOTE: input_ids needs `to(device)` at every time step
            for i in range(0, text_length):
                attention_mask = torch.ones(i+1, dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = torch.zeros(i+1, dtype=torch.long).unsqueeze(0).to(device)

                if i == 0:
                    mfccs = o_mfccs[i].unsqueeze(0).unsqueeze(0)
                    input_ids = torch.tensor(o_input_ids[:i+1]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            inputs_mfccs = mfccs
                        )
                        bert_log_likelihood = np.squeeze(outputs[0].cpu().numpy()).tolist()
                        current_step_hyps = scorer.score(bert_log_likelihood, Hypothesis())
                        #gt_rank.append(best_ids.tolist().index(example.input_ids[i+1]))
                        #sort_hyps(current_step_hyps)
                        #n_best = current_step_hyps[:decode_args.beam_size]
                        n_best = current_step_hyps

                else:
                    mfccs = torch.cat([o_mfccs[i].unsqueeze(0), o_mfccs[:i]]).unsqueeze(0)
                    new_n_best = []
                    for b in range(decode_args.beam_size):
                        input_ids = torch.tensor(
                            [o_input_ids[0]] + n_best[b][2]
                        ).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                inputs_mfccs = mfccs
                            )
                            bert_log_likelihood = np.squeeze(outputs[0].cpu().numpy()).tolist()
                            current_step_hyps = scorer.score(bert_log_likelihood, n_best[b])
                            #print(n_best_ids)
                            #print(log_likelihood[n_best_ids])

                            #new_n_best.extend(current_step_hyps[:decode_args.beam_size])
                            new_n_best.extend(current_step_hyps)

                    sort_hyps(new_n_best)
                    n_best = new_n_best[:decode_args.beam_size]

                    # Find if in new_n_best_sorted
                    #label_list = [item for item in new_n_best_sorted if item[0][-1] == example.input_ids[i+1]]
                    #if len(label_list) > 0:
                    #    gt_rank.append(
                    #        new_n_best_sorted.index(label_list[0])
                    #    )
                    #else:
                    #    gt_rank.append(-1)
                #print("Time step", i, ", best hypothesis:", tokenizer.decode(n_best[0][0]).replace(" ", ""))



            result_tuple=[
                tokenizer.decode(example.input_ids[1:text_length+1]).replace(" ", ""),
                n_best[0][1].replace(" ", "")
                #tokenizer.decode(n_best[0][0]).replace(" ", "")
            ]
            logger.info("{}/{}".format(j, len(eval_dataset)))
            logging.info("Original:   {}".format(result_tuple[0]))
            logging.info("Prediction: {}".format(result_tuple[1]))
            #logging.info("GT ranking: {}".format(gt_rank))
            writer.write("Original:   {}\n".format(result_tuple[0]))
            writer.write("Predition:  {}\n".format(result_tuple[1]))
            writer.flush()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
