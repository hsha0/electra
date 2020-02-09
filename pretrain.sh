TPU_NAME='grpc://10.77.129.26:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data_128/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_lamb_128 \
--vocab_file=vocab.txt \
--do_train=True \
--train_batch_size=128 \
--max_seq_length=128 \
--num_train_steps=1000000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
