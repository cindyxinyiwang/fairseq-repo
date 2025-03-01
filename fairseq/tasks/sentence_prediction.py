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


@register_task('sentence_prediction')
class SentencePredictionTask(LegacyFairseqTask):
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

        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='add prev_output_tokens to sample, used for encoder-decoder arch')
        parser.add_argument('--subword-data', type=str, default=None,
                            help='path to subword data')

    def __init__(self, args, data_dictionary, label_dictionary, subword_data_dictionary=None):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        self.subword_data_dictionary = subword_data_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        char_ngram = args.use_sde_embed
        dictionary = Dictionary.load(filename, char_ngram=char_ngram)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'input0', 'dict.txt'),
            source=True,
        )
        logger.info('[input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False,
            )
            logger.info('[label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict

        # load subword data dict
        if args.subword_data is not None:
            subword_data_dict = cls.load_dictionary(
                    args,
                    os.path.join(args.subword_data, 'input0', 'dict.txt'),
                    source=True,
            )
            logger.info('[input subword] dictionary: {} types'.format(len(subword_data_dict)))
        else:
            subword_data_dict = None
        return SentencePredictionTask(args, data_dict, label_dict, subword_data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        if hasattr(self.args, 'use_sde_embed') and self.args.use_sde_embed:
            return self.load_dataset_sde(split, combine)
        else:
            return self.load_dataset_normal(split, combine)

    def load_dataset_sde(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split, subword=False):
            if subword:
                return os.path.join(self.args.subword_data, type, split)
            else:
                return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary, char_ngram=True, subword=False):
            split_path = get_path(type, split, subword=subword)
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
                char_ngram=char_ngram
            )
            return dataset

        input0 = make_dataset('input0', self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split))
        input1 = make_dataset('input1', self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token, self.source_dictionary.pad())

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token, self.source_dictionary.pad())

            src_tokens = SimpleConcatDataset(input0, input1, self.dictionary, self.args)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        #src_tokens = maybe_shorten_dataset(
        #    src_tokens,
        #    split,
        #    self.args.shorten_data_split_list,
        #    self.args.shorten_method,
        #    self.args.max_positions,
        #    self.args.seed,
        #)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        if self.subword_data_dictionary is not None:
            subword_input0 = make_dataset('input0', self.subword_data_dictionary, char_ngram=False, subword=True)
            assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split, subword=True))
            subword_input1 = make_dataset('input1', self.subword_data_dictionary, char_ngram=False, subword=True)
            if self.args.init_token is not None:
                subword_input0 = PrependTokenDataset(subword_input0, self.args.init_token)
            if subword_input1 is None:
                subword_src_tokens = subword_input0
            else:
                if self.args.separator_token is not None:
                    subword_input1 = PrependTokenDataset(subword_input1, self.args.separator_token)
                subword_src_tokens = ConcatSentencesDataset(subword_input0, subword_input1)

            subword_src_tokens = RightPadDataset(
                    subword_src_tokens,
                    pad_idx=self.subword_data_dictionary.pad(),
            )
            dataset['net_input'].update(subword_src_tokens=subword_src_tokens)
            dataset['net_input']['src_lengths'] = NumelDataset(subword_src_tokens, reduce=False)

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset['net_input'].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.args.regression_target:
            label_dataset = make_dataset('label', self.label_dictionary, char_ngram=False)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path('label', split))
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert len(values) == self.args.num_classes, \
                        f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    dataset.update(
                        target=RawLabelDataset([
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ])
                    )

        if self.subword_data_dictionary is not None:
            sizes=[subword_src_tokens.sizes]
        else:
            sizes=[src_tokens.sizes]

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=sizes,
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


    def load_dataset_normal(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                args=self.args,
                combine=combine,
            )
            return dataset

        input0 = make_dataset('input0', self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path('input0', split))
        input1 = make_dataset('input1', self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions,
            self.args.seed,
        )

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset['net_input'].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.args.regression_target:
            label_dataset = make_dataset('label', self.label_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path('label', split))
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert len(values) == self.args.num_classes, \
                        f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    dataset.update(
                        target=RawLabelDataset([
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ])
                    )

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

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, 'classification_head_name', 'sentence_classification_head'),
            num_classes=self.args.num_classes,
        )

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
            loss, sample_size, logging_output = criterion(model, sample, valid=True)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)


