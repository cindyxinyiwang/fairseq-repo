# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

import numpy as np

from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    SimpleConcatDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
)
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset


logger = logging.getLogger(__name__)


@register_task('sde_pretrain')
class SDEpretrainTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--charngram-dict-path', type=str, default=None,
                            help='path to the char ngram vocab')
        parser.add_argument('--subword-dict-path', type=str, default=None,
                            help='path to the bert subword vocab')
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')

    def __init__(self, args, charngram_dictionary, subword_dictionary):
        super().__init__(args)
        self.charngram_dictionary = charngram_dictionary
        self.subword_dictionary = subword_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, char_ngram=False):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename, char_ngram=char_ngram)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load data dictionary
        charngram_data_dict = cls.load_dictionary(
            args,
            args.charngram_dict_path,
            char_ngram=True,
        )
        logger.info('[char ngram] dictionary: {} types'.format(len(charngram_data_dict)))

        subword_data_dict = cls.load_dictionary(
            args,
            args.subword_dict_path,
            char_ngram=args.charngram_dict_path,
        )
        logger.info('[subword] dictionary: {} types'.format(len(subword_data_dict)))

        return SDEpretrainTask(args, charngram_data_dict, subword_data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary, label=False):
            split_path = get_path(type, split)
            #if target:
            #    impl = 'raw'
            #    split_path += '.None-None'
            #else:
            #    impl = self.args.dataset_impl
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                args=self.args,
                combine=combine,
                label=label
            )
            return dataset

        input0 = make_dataset('input0', self.charngram_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split))
        input1 = make_dataset('input1', self.charngram_dictionary)

        src_tokens = SimpleConcatDataset(input0, input1, self.charngram_dictionary, self.args)
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        input0 = make_dataset('input0', self.subword_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split))
        input1 = make_dataset('input1', self.subword_dictionary)

        trg_tokens = SimpleConcatDataset(input0, input1, self.subword_dictionary, self.args)
        dataset = {
            'id': IdDataset(),
            'net_input': {
                'charngram_tokens': src_tokens,
                'subword_tokens': trg_tokens,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
