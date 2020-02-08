TPU_NAME='grpc://10.9.165.250:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_lamb \
--vocab_file=vocab.txt \
--do_train=True \
--train_batch_size=128 \
--max_seq_length=256 \
--num_train_steps=125000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
