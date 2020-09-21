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
""" ASR processors and helpers """

import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from dataclasses import dataclass
import h5py
from itertools import zip_longest
from kaldiio import ReadHelper
import numpy as np
import time
from tqdm.auto import tqdm, trange

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from .utils import DataProcessor, InputExample, InputFeatures


logger = logging.get_logger(__name__)

@dataclass
class InputASRExample:
    """
    A single training/test example for ASR.

    Args:
        guid: Unique id for the example.
        text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text: str
    label_pos: int
    mfcc_loader:str = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputASRFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    #label: Optional[Union[int, float]] = None
    label_pos: Optional[int] = None
    mfcc_loader: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def asr_convert_examples_to_features(
    examples: Union[List[InputExample], List[InputASRExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    #task=None,
    #label_list=None,
    #output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _asr_convert_examples_to_features(
        examples, tokenizer, max_length=max_length,
        #task=task, label_list=label_list, output_mode=output_mode
    )



def _asr_convert_examples_to_features(
    examples: Union[List[InputExample], List[InputASRExample]],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    print("Tokenization start")
    start = time.time()
    batch_encoding = tokenizer(
        [example.text for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    print("Tokenization took {:.1f} s".format(time.time() - start))

    features = []
    #for i in range(len(examples)):
    for i in tqdm(range(len(examples))):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputASRFeatures(**inputs, label_pos=examples[i].label_pos, mfcc_loader=examples[i].mfcc_loader)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

class ASRProcessor(DataProcessor):
    """Processor for an ASR data set."""

    def _read_scp(self, file_path, debug, debug_lines=100):
        with open(file_path, encoding="utf-8") as f:
            if debug:
                return [line for line in f.read().splitlines()[:debug_lines] if (len(line) > 0 and not line.isspace())]
            else:
                return [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    def _read_mfcc_loader_from_scp(self, scp, mfcc_dir, debug, debug_lines=100):
        """Return a dict: {fileid: processed_mfcc}"""
        if debug:
            fileids = [line.split(" ")[0] for line in scp][:debug_lines]
        else:
            fileids = [line.split(" ")[0] for line in scp]
        # To avoid disk access, create loader only for the first time
        #return {f: h5py.File(os.path.join(mfcc_dir, f+".hdf5"), 'r')["mfccs"][()]
        #return {f: h5py.File(os.path.join(mfcc_dir, f+".hdf5"), 'r')
        return {f: os.path.join(mfcc_dir, f+".hdf5")
            for f in fileids
        }

    def get_examples(self, mode, data_source, mfcc_dir, use_audio, exhaustion, use_scp=False, debug=False):
        if use_scp:
            scp = self._read_scp(data_source, debug)
        else:
            scp = self._read_scp(os.path.join(data_source, mode.value + ".txt"), debug)
        mfcc_loaders = self._read_mfcc_loader_from_scp(scp, os.path.join(mfcc_dir, mode.value), debug) if use_audio else None
        return self._create_examples(scp, mfcc_loaders, mode, exhaustion=exhaustion)

    def _create_examples(self, lines, mfcc_loaders, set_type, exhaustion=False):
        """Creates examples for the training, dev and test sets."""
        examples = []
        #for (i, line) in enumerate(lines):
        for (i, line) in tqdm(enumerate(lines)):
            fileid, words = line.split(" ")[:2]
            mfcc_loader = mfcc_loaders[fileid] if mfcc_loaders is not None else None
            if exhaustion:
                for j in range(len(words)):
                    guid = "%s-%s-%s" % (set_type, i, j)
                    text = words[:j+1]
                    examples.append(InputASRExample(guid=guid, text=text, label_pos=j+1, mfcc_loader=mfcc_loader))
            else:
                guid = "%s-%s" % (set_type, i)
                text = words
                examples.append(InputASRExample(guid=guid, text=text, label_pos=len(words), mfcc_loader=mfcc_loader))
        return examples
