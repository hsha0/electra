for num in $(seq 1 22)
do
    python3 create_pretraining_data.py --input_file=../bucket/bookcorpus/processed_book/bookcorpus_$num.txt --output_file=../bucket/data/bookcorpus_$num.tfrecord --vocab_file=vocab.txt
done