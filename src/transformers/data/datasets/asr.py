import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from ...tokenization_bart import BartTokenizer, BartTokenizerFast
#from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
#from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ...utils import logging
from ..processors.asr import asr_convert_examples_to_features, ASRTextProcessor, ASRProcessor, InputASRFeatures
from ..processors.utils import InputFeatures


logger = logging.get_logger(__name__)


@dataclass
class ASRDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir containing the text."}
    )
    mfcc_dir: str = field(
        metadata={"help": "The data dir of the MFCC hdf5 files."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    exhaustion: str = field(
        default="no", metadata={"help": "Exhaustively use all input combinations"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    use_audio: Optional[str] = field(
        default="no", metadata={"help": "Where to use audio or not"}
    )


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class ASRDataset(Dataset):

    args: ASRDataTrainingArguments
    output_mode: str
    features: List[InputASRFeatures]

    def __init__(
        self,
        args: ASRDataTrainingArguments,
        use_audio: bool,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = ASRProcessor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
            ),
        )
        self.use_audio = use_audio

        """
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir, args.mfcc_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir, args.mfcc_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir, args.mfcc_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = asr_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
        """
        exhaustion = True if args.exhaustion == "yes" else False
        if mode == Split.dev:
            examples = self.processor.get_dev_examples(args.data_dir, args.mfcc_dir, self.use_audio, exhaustion)
        elif mode == Split.test:
            examples = self.processor.get_test_examples(args.data_dir, args.mfcc_dir, self.use_audio, exhaustion)
        else:
            examples = self.processor.get_train_examples(args.data_dir, args.mfcc_dir, self.use_audio, exhaustion)
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = asr_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


@dataclass
class ASRTextDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class ASRTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: ASRTextDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: ASRTextDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = ASRTextProcessor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                #args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = asr_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    #label_list=label_list,
                    #output_mode=self.output_mode,
                )
                #exit()
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
