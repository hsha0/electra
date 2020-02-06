for num in $(seq 0 9)
do
    python3 create_pretraining_data.py --input_file=gs://electra/wiki/processed_wiki/AB/wiki_0$num --output_file=gs://electra/data/AB_wiki_0$num.tfrecord --vocab_file=vocab.txt
done