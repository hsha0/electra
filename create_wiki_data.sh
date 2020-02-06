for num in $(seq 27 99)
do
    python3 create_pretraining_data.py --input_file=gs://electra/wiki/processed_wiki/AB/wiki_$num --output_file=gs://electra/data/AB_wiki_$num.tfrecord --vocab_file=vocab.txt
done