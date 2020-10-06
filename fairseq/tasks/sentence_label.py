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
    BeginOfWord,
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
    BeginOfWord,
    TransformEosDataset,
)
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset


logger = logging.getLogger(__name__)


@register_task('sentence_label')
class SentenceLabelTask(LegacyFairseqTask):
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
        parser.add_argument('--src-lang', type=str, default=None,
                            help='source language')

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
            os.path.join(args.data, 'dict.{}.txt'.format(args.src_lang)),
            source=True,
        )
        logger.info('[input] dictionary: {} types'.format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'dict.label.txt'),
            source=False,
        )
        logger.info('[label] dictionary: {} types'.format(len(label_dict)))

        # load subword data dict
        if args.subword_data is not None:
            subword_data_dict = cls.load_dictionary(
                    args,
                    os.path.join(args.subword_data, 'dict.tr.txt'),
                    source=True,
            )
            logger.info('[input subword] dictionary: {} types'.format(len(subword_data_dict)))
        else:
            subword_data_dict = None
        return SentenceLabelTask(args, data_dict, label_dict, subword_data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        if hasattr(self.args, 'use_sde_embed') and self.args.use_sde_embed:
            return self.load_dataset_sde(split, combine)
        else:
            return self.load_dataset_normal(split, combine)

    def load_dataset_sde(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def make_dataset(type, dictionary, char_ngram=True, subword=False):
            if subword:
                data_path = self.args.subword_data
            else:
                data_path = self.args.data
            split_path = os.path.join(data_path, split+".{}-{}.{}".format(self.args.src_lang, 'label', type))
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                args=self.args,
                combine=combine,
                char_ngram=char_ngram,
            )
            assert dataset is not None, 'could not find dataset: {}'.format(split_path)
            return dataset

        src_tokens = make_dataset(self.args.src_lang, self.source_dictionary)
        src_tokens = SimpleConcatDataset(src_tokens, None, self.dictionary, self.args)

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
            subword_src_tokens = make_dataset(self.args.src_lang, self.subword_data_dictionary, char_ngram=False, subword=True)
            subword_src_tokens = RightPadDataset(
                    subword_src_tokens,
                    pad_idx=self.subword_data_dictionary.pad(),
            )
            dataset['net_input'].update(subword_src_tokens=subword_src_tokens)
            dataset['net_input'].update(bow_mask=BeginOfWord(subword_src_tokens, self.subword_data_dictionary))

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset['net_input'].update(
                prev_output_tokens=prev_tokens_dataset,
            )

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


    def load_dataset_normal(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def make_dataset(type, dictionary):
            split_path = os.path.join(self.args.data, split+".{}-{}.{}".format(self.args.src_lang, 'label', type))
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                args=self.args,
                combine=combine,
            )
            assert dataset is not None, 'could not find dataset: {}'.format(split_path)
            return dataset

        src_tokens = make_dataset(self.args.src_lang, self.source_dictionary)

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions,
            self.args.seed,
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
                'bow_mask': BeginOfWord(src_tokens, self.source_dictionary),
                #'bow_mask': RightPadDataset(BeginOfWord(src_tokens, self.source_dictionary), pad_idx=0),
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
            try:
                loss, sample_size, logging_output = criterion(model, sample)
                if ignore_grad:
                    loss *= 0
                with torch.autograd.profiler.record_function("backward"):
                    optimizer.backward(loss)
                return loss, sample_size, logging_output
            except:
                print(sample['target'])
                print(sample['net_input']['subword_src_tokens'])
                print(sample['net_input']['bow_mask'])
                for w in sample['net_input']['subword_src_tokens'].view(-1):
                    print(self.dictionary[w])
                return 0, 0, {} 

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, valid=True)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)


