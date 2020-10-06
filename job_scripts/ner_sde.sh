#!/bin/bash

LR=5e-06              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=1      # Batch size.
UPDATE_FREQ=32     # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=pretrained_models/xlmr/xlmr.base/model.pt
DATA_DIR=data-bin/ner_bpe_ngram/
MODEL_DIR=checkpoints/ner_sde/tr/

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:
#FAIRSEQ_PATH=/path/to/fairseq
#FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa

mkdir -p $MODEL_DIR
#CUDA_VISIBLE_DEVICES=0 fairseq-train \
fairseq-train \
    $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --use-sde-embed --subword-data data-bin/ner/ \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_label  \
    --src-lang tr \
    --arch roberta_base --max-positions 512 --sent-label \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_prediction --num-classes 7 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --max-epoch 30 \
    --log-format simple --log-interval 5 \
  	--save-dir $MODEL_DIR \
    --cpu  --distributed-world-size 1 \
	  --seed $SEED
	  #--seed $SEED >> $MODEL_DIR/train.log 2>&1
