#!/bin/bash

LR=5e-06              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8      # Batch size.
UPDATE_FREQ=16     # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=pretrained_models/xlmr/xlmr.base/model.pt
DATA_DIR=data-bin/xnli/ar/
MODEL_DIR=checkpoints/xnli/ar/

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:
#FAIRSEQ_PATH=/path/to/fairseq
#FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa

mkdir -p $MODEL_DIR
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_prediction --init-token 0 --separator-token 2 \
    --arch roberta_base --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_prediction --num-classes 3 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --max-epoch 30 \
    --log-format simple --log-interval 25 \
  	--save-dir $MODEL_DIR \
	  --seed $SEED >> $MODEL_DIR/train.log 2>&1
