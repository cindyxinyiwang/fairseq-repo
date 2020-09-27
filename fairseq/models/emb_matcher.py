# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    SDE,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


logger = logging.getLogger(__name__)


@register_model('emb_matcher')
class EmbMatcher(BaseFairseqModel):


    def __init__(self, args, emb_train, emb_match, dict_train, dict_match):
        super().__init__()
        self.args = args
        self.emb_train = emb_train
        self.emb_match = emb_match
        self.dict_train = dict_train
        self.dict_match = dict_match

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        emb_train = SDE(task.charngram_dictionary, dim=args.encoder_embed_dim)
        emb_match = nn.Embedding(len(task.subword_dictionary), args.encoder_embed_dim, task.subword_dictionary.pad())
        emb_match.weight.requires_grad = False
        return cls(args, emb_train, emb_match, task.charngram_dictionary, task.subword_dictionary)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, charngram_tokens, subword_tokens, **kwargs):
        emb_charngram = self.emb_train(charngram_tokens)
        emb_subword = self.emb_match(subword_tokens).sum(dim=2)

        subword_counts = torch.clamp((1-subword_tokens.eq(self.dict_match.pad()).int()).sum(dim=2), min=1)
        emb_subword = emb_subword / subword_counts.unsqueeze(dim=-1)

        charngram_padding_mask = charngram_tokens[:, :, 0].eq(self.dict_train.pad())
        subword_padding_mask = subword_tokens[:, :, 0].eq(self.dict_match.pad())
        if not charngram_padding_mask.any():
            charngram_padding_mask = None
        if not subword_padding_mask.any():
            subword_padding_mask = None
        if charngram_padding_mask is not None:
            assert subword_padding_mask is not None
            assert charngram_padding_mask.int().sum() == subword_padding_mask.int().sum()
            emb_charngram = emb_charngram * (1 - charngram_padding_mask.unsqueeze(-1).type_as(emb_charngram))
            emb_subword = emb_subword * (1 - subword_padding_mask.unsqueeze(-1).type_as(emb_subword))
        return emb_charngram, emb_subword 

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Only keep the embedding
        keys_to_delete = []
        for k in list(state_dict.keys()):
            if k == 'encoder.sentence_encoder.embed_tokens.weight':
                state_dict['emb_match.weight'] = state_dict[k]
            keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

@register_model_architecture('emb_matcher', 'emb_matcher')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.dropout = getattr(args, 'dropout', 0.1)

