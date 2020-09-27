
DICT=pretrained_models/xlmr/xlmr.base/dict.txt
MODEL=pretrained_models/xlmr/xlmr.base/sentencepiece.bpe.model

#for file in xnli/data/*input0*; do
#  f=${file/input0/input0.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done
#
#for file in xnli/data/*input1*; do
#  f=${file/input1/input1.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done
#
#for file in xnli/data/*label*; do
#  f=${file/label/label.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done

for LAN in ar; do
mkdir -p data-bin/xnli_bpe_ngram/"$LAN"/
#cat xnli/data/xnli.train.input0."$LAN" xnli/data/xnli.train.input1."$LAN" > data-bin/xnli_charngram/"$LAN"/combined.input
#fairseq-preprocess \
#--only-source \
#--char-ngram \
#--max-char-size 50 \
#--nwordssrc 32000 \
#--trainpref data-bin/xnli_charngram/"$LAN"/combined.input \
#--destdir data-bin/xnli_charngram/"$LAN"/ \
#--workers 32 
for type in input0 input1; do
  fairseq-preprocess \
  --only-source \
  --char-ngram \
  --max-char-size 50 \
  --trainpref xnli/data/xnli.train."$type"."$LAN" \
  --validpref xnli/data/xnli.dev."$type"."$LAN" \
  --testpref xnli/data/xnli.test."$type"."$LAN" \
  --destdir data-bin/xnli_bpe_ngram/"$LAN"/"$type" \
  --srcdict $DICT \
  --workers 32 
done
done

fairseq-preprocess \
  --only-source \
  --trainpref xnli/data/xnli.train.label.ar \
  --validpref xnli/data/xnli.dev.label.ar \
  --testpref xnli/data/xnli.test.label.ar \
  --destdir data-bin/xnli_bpe_ngram/ar/label \
  --workers 32

#for LAN in bg de el en es fr hi ru sw th tr ur vi zh; do
#  fairseq-preprocess \
#  --only-source \
#  --trainpref xnli/data/xnli.train.label."$LAN" \
#  --validpref xnli/data/xnli.dev.label."$LAN" \
#  --testpref xnli/data/xnli.test.label."$LAN" \
#  --destdir data-bin/xnli/"$LAN"/label \
#  --srcdict data-bin/xnli/ar/label/dict.txt \
#  --workers 32
#done
