# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch
from fairseq.file_io import PathManager
from fairseq.data import data_utils

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
        char_ngram=False,
        max_char_size=50,
        bpe_ngram=False,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    if char_ngram:
                        ids = []
                        for w in line.split():
                            # max_char_len
                            if bpe_ngram:
                                char_ngram = data_utils.word_to_bpe_ngram(w, dict, max_char_size)
                            else:
                                char_ngram = data_utils.word_to_ngram(w, dict, max_char_size)
                            ids.append(char_ngram)
                        if append_eos:
                            char_ngram = [dict.eos()] + [dict.pad()]*(max_char_size-1)
                            ids.append(char_ngram)
                        ids = torch.LongTensor(ids)
                    else:
                        ids = dict.encode_line(
                            line=line,
                            line_tokenizer=tokenize,
                            add_if_not_exist=False,
                            consumer=replaced_consumer,
                            append_eos=append_eos,
                            reverse_order=reverse_order,
                        )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(PathManager.get_local_path(filename), "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
