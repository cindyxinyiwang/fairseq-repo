# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True, valid=False, l_subword=0, l_kl=0, l_rep=0):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        logits, extra = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        #if len(logits) == 2:
        if False:
            logits_sde, logits_subword = logits
            targets = model.get_targets(sample, [logits_sde]).view(-1)
            sample_size = targets.numel()
            lprobs_sde = F.log_softmax(logits_sde, dim=-1, dtype=torch.float32)
            loss_sde = F.nll_loss(lprobs_sde, targets, reduction='sum')

            lprobs_subword = F.log_softmax(logits_subword, dim=-1, dtype=torch.float32)
            loss_subword = F.nll_loss(lprobs_subword, targets, reduction='sum')

            kl = F.kl_div(lprobs_sde, lprobs_subword, reduction='sum', log_target=True)

            sent_rep_sde, sent_rep_subword = extra[0]['sent_rep'], extra[1]['sent_rep']
            rep_loss = -F.cosine_similarity(sent_rep_sde, sent_rep_subword)
            rep_loss = rep_loss.sum()

            loss = loss_sde + loss_subword + 0.1*kl + 0.1*rep_loss
            logits = logits_sde+logits_subword
        else:
            targets = model.get_targets(sample, [logits]).view(-1)
            sample_size = targets.numel()

            if not self.regression_target:
                logits = logits.view(-1, logits.size(-1))
                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                loss = F.nll_loss(lprobs, targets, reduction='sum')
            else:
                logits = logits.view(-1).float()
                targets = targets.float()
                loss = F.mse_loss(logits, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
