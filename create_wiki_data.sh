for num in $(seq 0 9)
do
    python3 create_pretraining_data.py --input_file=gs://electra/wiki/processed_wiki/AA/wiki_0$num --output_file=gs://electra/data_128/AA_wiki_0$num.tfrecord --vocab_file=vocab.txt
done