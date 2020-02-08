for num in $(seq 1 22)
do
    python3 create_pretraining_data.py --input_file=gs://electra/bookcorpus/processed_book/bookcorpus_$num.txt --output_file=gs://electra/data_128/bookcorpus_$num.tfrecord --vocab_file=vocab.txt
done