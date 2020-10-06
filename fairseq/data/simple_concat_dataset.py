# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class SimpleConcatDataset(FairseqDataset):

    def __init__(self, dataset_1, dataset_2, vocab, args):
        super().__init__()
        self.vocab = vocab
        self.args = args
        if dataset_2 is None:
            self.datasets = [dataset_1]
        else:
            self.datasets = [dataset_1, dataset_2]
        assert all(len(ds) == len(self.datasets[0]) for ds in self.datasets), \
            'datasets must have the same length'

    def __getitem__(self, index):
        return [d[index] for d in self.datasets]

    def __item_size__(self, index):
        return sum([len(d[index]) for d in self.datasets])

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        if len(self.datasets) == 1:
            max_len = max(len(tok_0[0]) for tok_0 in samples)
            concat_tokens = []
            for tok_0 in samples:
                tok_0 = tok_0[0]
                pad_len = max_len - tok_0.size(0)
                if pad_len > 0:
                    pad_tokens = torch.LongTensor(pad_len, tok_0.size(1)).fill_(self.vocab.pad()).to(tok_0.device)
                    concat = torch.cat([tok_0, pad_tokens], dim=0)
                else:
                    concat = tok_0
                concat_tokens.append(concat)
            concat_tokens = torch.stack(concat_tokens, dim=0)
            return concat_tokens 
        else:
            max_len = max(len(tok_0)+len(tok_1) for tok_0, tok_1 in samples)
            concat_tokens = []
            for tok_0, tok_1 in samples:
                pad_len = max_len - tok_0.size(0) - tok_1.size(0)
                if pad_len > 0:
                    pad_tokens = torch.LongTensor(pad_len, tok_0.size(1)).fill_(self.vocab.pad()).to(tok_0.device)
                    concat = torch.cat([tok_0, tok_1, pad_tokens], dim=0)
                else:
                    concat = torch.cat([tok_0, tok_1], dim=0)
                concat_tokens.append(concat)
            concat_tokens = torch.stack(concat_tokens, dim=0)
            return concat_tokens 

    @property
    def sizes(self):
        return sum(ds.sizes for ds in self.datasets)

    def num_tokens(self, index):
        return sum(ds.num_tokens(index) for ds in self.datasets)

    def size(self, index):
        return sum(ds.size(index) for ds in self.datasets)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(
            getattr(ds, 'supports_prefetch', False) for ds in self.datasets
        )

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
