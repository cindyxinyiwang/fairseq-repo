#!/bin/bash

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

VOCAB_SIZE=4000
RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/var_rus_sepspm"$VOCAB_SIZE"/
BINARIZED_DDIR=data-bin/var_rus_sepspm"$VOCAB_SIZE"/
FAIR_SCRIPTS=scripts/
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
SPM_DECODE=$FAIR_SCRIPTS/spm_decode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl
FILTER=mosesdecoder/scripts/training/clean-corpus-n.perl

LANS=(
  rus
  bel
  ukr)
MAIN=rus

#for LAN in ${LANS[@]}; do
#  mkdir -p "$PROC_DDIR"/"$LAN"_eng
#
#  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.*-eng  ; do
#    src=`echo $f | sed 's/-eng$//g'`
#    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
#    if [ ! -f "$src" ]; then
#      echo "src=$src, trg=$trg"
#      python preprocess_scripts/cut-corpus.py 0 < $f > $src
#      python preprocess_scripts/cut-corpus.py 1 < $f > $trg
#    fi
#  done
#  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.{eng,$LAN} ; do
#    f1=${f/orig/mtok}
#    if [ ! -f "$f1" ]; then
#      echo "tokenize $f1..."
#      cat $f | perl $TOKENIZER > $f1
#    fi
#  done
#done

## learn BPE with sentencepiece for English
#TRAIN_FILES="$RAW_DDIR"/"$MAIN"_eng/ted-train.mtok.eng
#echo "learning BPE for eng over ${TRAIN_FILES}..."
#python "$SPM_TRAIN" \
#      --input=$TRAIN_FILES \
#      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE".eng \
#      --vocab_size=$VOCAB_SIZE \
#      --character_coverage=1.0 \
#      --model_type=bpe
#
#TRAIN_FILES="$RAW_DDIR"/"$MAIN"_eng/ted-train.mtok."$MAIN"
#echo "learning BPE for eng over ${TRAIN_FILES}..."
#python "$SPM_TRAIN" \
#      --input=$TRAIN_FILES \
#      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE"."$MAIN" \
#      --vocab_size=$VOCAB_SIZE \
#      --character_coverage=1.0 \
#      --model_type=bpe
#
## train a separate BPE model for main language, then encode the data with this BPE model
#for LAN in ${LANS[@]}; do
#  python "$SPM_ENCODE" \
#        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
#        --output_format=piece \
#        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng  \
#        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter.eng \
# 
#  python "$SPM_ENCODE" \
#        --model="$PROC_DDIR"/spm"$VOCAB_SIZE"."$MAIN".model \
#        --output_format=piece \
#        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN"  \
#        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter."$LAN" \
#
#  # filter out training data longer than 200 words
#  $FILTER "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter $LAN eng "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" 1 200
# 
#  echo "encoding valid/test data with learned BPE..."
#  for split in dev test;
#  do
#  python "$SPM_ENCODE" \
#        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
#        --output_format=piece \
#        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok.eng  \
#        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  
#  done
#  for split in dev test;
#  do
#  python "$SPM_ENCODE" \
#        --model="$PROC_DDIR"/spm"$VOCAB_SIZE"."$MAIN".model \
#        --output_format=piece \
#        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok."$LAN"  \
#        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LAN" 
#  done
#done
#
#
## Concatenate all the training data from all languages to get combined vocabulary
#mkdir -p $BINARIZED_DDIR
#mkdir -p $BINARIZED_DDIR/M2O/
#
#fairseq-preprocess -s $MAIN -t eng \
#  --trainpref "$PROC_DDIR"/"$MAIN"_eng/ted-train.spm"$VOCAB_SIZE" \
#  --validpref "$PROC_DDIR"/"$MAIN"_eng/ted-dev.spm"$VOCAB_SIZE" \
#  --testpref "$PROC_DDIR"/"$MAIN"_eng/ted-test.spm"$VOCAB_SIZE" \
#  --workers 8 \
#  --thresholdsrc 0 \
#  --thresholdtgt 0 \
#  --destdir $BINARIZED_DDIR
#
#echo "Binarize the data..."
#for LAN in ${LANS[@]}; do
#  # Binarize the data for many-to-one translation
#  fairseq-preprocess --source-lang $LAN --target-lang eng \
#        --trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" \
#        --validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.spm"$VOCAB_SIZE" \
#        --testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.spm"$VOCAB_SIZE" \
#      	--srcdict $BINARIZED_DDIR/dict."$MAIN".txt \
#      	--tgtdict $BINARIZED_DDIR/dict.eng.txt \
#        --destdir $BINARIZED_DDIR/M2O/
#done

## Preprocess for SDE
#for LAN in ${LANS[@]}; do
#  for split in train dev test; do
#    python $SPM_DECODE \
#      --model="$PROC_DDIR"/spm"$VOCAB_SIZE"."$MAIN".model \
#      --input="$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LAN" > "$PROC_DDIR"/"$LAN"_eng/ted-"$split".sde."$LAN"
#
#    cp "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  "$PROC_DDIR"/"$LAN"_eng/ted-"$split".sde.eng
#  done
#done
#

SDE_BINARIZED_DDIR=data-bin/var_rus_sde_sepspm"$VOCAB_SIZE"/

fairseq-preprocess -s $MAIN -t eng \
  --src-char-ngram \
  --max-char-size 50 \
  --nwordssrc 32000 \
  --only-source \
  --trainpref "$PROC_DDIR"/"$MAIN"_eng/ted-train.sde \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $SDE_BINARIZED_DDIR

for LAN in ${LANS[@]}; do
  fairseq-preprocess -s $LAN -t eng \
    --src-char-ngram \
    --max-char-size 50 \
    --trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.sde \
    --validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.sde \
    --testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.sde \
    --workers 8 \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --srcdict $SDE_BINARIZED_DDIR/dict."$MAIN".txt \
    --tgtdict $BINARIZED_DDIR/dict.eng.txt \
    --destdir $SDE_BINARIZED_DDIR
done
