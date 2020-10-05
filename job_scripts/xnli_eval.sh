
MODEL_DIR=checkpoints/xnli/ar_lr1e-5/
DATA_DIR=data-bin/xnli/vi/
#DATA_DIR=data-bin/xnli_bpe_ngram/ur/
#MODEL_DIR=checkpoints/xnli/ar_sde_cat_init/
#    --subword-data data-bin/xnli/ur/ \
#    --use-sde-embed \
CUDA_VISIBLE_DEVICES=$1 python eval_classification.py \
    $DATA_DIR \
    --gen-subset test \
    --max-positions 512 \
    --arch roberta_base \
    --task sentence_prediction --init-token 0 --separator-token 2 \
    --num-classes 3 \
    --max-sentences 4 \
    --path $MODEL_DIR/checkpoint_best.pt 

 
