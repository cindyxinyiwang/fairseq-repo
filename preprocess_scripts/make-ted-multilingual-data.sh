#!/bin/bash

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

VOCAB_SIZE=16000
RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/multilingual_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=fairseq/data-bin/ted_azetur_spm"$VOCAB_SIZE"/
FAIR_SCRIPTS=fairseq/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl

LANS=(
  aze
  tur)

for LAN in ${LANS[@]}; do
  mkdir -p "$PROC_DDIR"/"$LAN"_eng

  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.*-eng  ; do
    src=`echo $f | sed 's/-eng$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
    if [ ! -f "$src" ]; then
      echo "src=$src, trg=$trg"
      python preprocess_scripts/cut-corpus.py 0 < $f > $src
      python preprocess_scripts/cut-corpus.py 1 < $f > $trg
    fi
  done
  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.{eng,$LAN} ; do
    f1=${f/orig/mtok}
    if [ ! -f "$f1" ]; then
      echo "tokenize $f1..."
      cat $f | perl $TOKENIZER > $f1
    fi
  done
done

# learn BPE with sentencepiece
TRAIN_FILES=$(for LAN in "${LANS[@]}"; do echo "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN"; echo "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
      --input=$TRAIN_FILES \
      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE" \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=1.0 \
      --model_type=bpe

for i in ${!LANS[*]}; do
  LAN=${LANS[$i]}
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN" "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE"."$LAN" "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".eng \
        --min-len 1 --max-len 200 
  
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok."$LAN" "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok.eng  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LAN" "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  
  done
done

# Concatenate all the training data from all languages to get combined vocabulary
mkdir -p $BINARIZED_DDIR
mkdir -p $BINARIZED_DDIR/M2O/
mkdir -p $BINARIZED_DDIR/O2M/

for LAN in ${LANS[@]}; do
  cat $PROC_DDIR/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE"."$LAN" >> $BINARIZED_DDIR/combined-train.src
  cat $PROC_DDIR/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".eng >> $BINARIZED_DDIR/combined-train.eng
done
fairseq-preprocess -s src -t eng \
  --trainpref $BINARIZED_DDIR/combined-train \
  --joined-dictionary \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $BINARIZED_DDIR

echo "Binarize the data..."
for LAN in ${LANS[@]}; do
  # Binarize the data for many-to-one translation
  fairseq-preprocess --source-lang $LAN --target-lang eng \
        --trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.src.txt \
      	--tgtdict $BINARIZED_DDIR/dict.eng.txt \
        --destdir $BINARIZED_DDIR/M2O/
  
  # Binarize the data for one-to-many translation
  fairseq-preprocess --source-lang eng --target-lang $LAN \
        --trainpref "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LAN"_eng/ted-dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LAN"_eng/ted-test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.eng.txt \
      	--tgtdict $BINARIZED_DDIR/dict.src.txt \
        --destdir $BINARIZED_DDIR/O2M/
done
