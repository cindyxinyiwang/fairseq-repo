# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class BeginOfWord(BaseWrapperDataset):

    def __init__(self, dataset, dic, reduce=False):
        super().__init__(dataset)
        self.reduce = reduce
        self.dic = dic
        self.bow_lists = []
        for data in dataset:
            mask = []
            ws = []
            for i, w in enumerate(data):
                word = dic[w]
                #ws.append(word)
                if word.startswith('\u2581'):
                    #mask.append(1)
                    mask.append(i)
                else:
                    # hack for corner case
                    #if i > 0 and (word == 'ten' or word == 'tan') and mask[-1] == 1:
                    #    mask[-1] = 0
                    continue
                    #mask.append(0)
            #print(ws)
            #print(mask)
            self.bow_lists.append(mask)

    def __getitem__(self, index):
        return torch.LongTensor(self.bow_lists[index])

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.reduce:
            return sum(samples)
        else:
            return samples
