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
Fine-tune a BERT model for ASR.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional


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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    acoustic_encoder_type: Optional[str] = field(
        default="conv", metadata={"help": "Acoustic encoder type."}
    )
    acoustic_encoder_segment: Optional[str] = field(
        default="first", metadata={"help": "Acoustic encoder segment place."}
    )
    acoustic_encoder_layers: Optional[int] = field(
        default=1, metadata={"help": "Acoustic encoder number of layers"}
    )


@dataclass
class CustomArguments:
    """
    Custom arguments.
    """
    freeze_mods: Optional[str] = field(
        default=None, metadata={"help": "List of modules to freeze (not to train), separated by a comma."}
    )
    debugging: Optional[str] = field(
        default=False, metadata={"help": "Debug flag."}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if custom_args.debugging == "yes":
        custom_args.debug = True
    else:
        custom_args.debug = False

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
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
    logger.info("custom parameters %s", custom_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        # logger.warning("You are instantiating a new config instance from scratch.")
        raise ValueError

    config.num_labels = config.vocab_size
    config.use_audio = True if data_args.use_audio == "yes" else False
    config.fusion_place = data_args.fusion_place
    config.acoustic_encoder_type = model_args.acoustic_encoder_type
    config.acoustic_encoder_segment = model_args.acoustic_encoder_segment
    config.acoustic_encoder_layers = model_args.acoustic_encoder_layers

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = BertForASR.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError

    model.resize_token_embeddings(len(tokenizer))

    # Optionally freeze parameters
    freeze_mods = [str(mod) for mod in custom_args.freeze_mods.split(",") if mod != ""]
    for mod, param in model.named_parameters():
        if any(mod.startswith(key) for key in freeze_mods):
            logging.info("freezing %s" % mod)
            param.requires_grad = False

    if data_args.max_seq_length <= 0:
        data_args.max_seq_length = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.max_seq_length = min(data_args.max_seq_length, tokenizer.max_len)

    # Get datasets

    logger.info("*** Building training dataset ***")
    train_dataset = (
        ASRDataset(data_args, use_audio=config.use_audio, tokenizer=tokenizer, cache_dir=model_args.cache_dir, debug=custom_args.debug)
        if training_args.do_train else None
    )
    data_collator = DataCollatorForASR(max_text_length = data_args.max_seq_length)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        logger.info("** Training start ***")
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
