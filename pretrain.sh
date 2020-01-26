TPU_NAME='grpc://10.0.124.242:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=test_256.tfrecord \
--output_dir=$ELECTRA_GC/results \
--vocab_file=vocab.txt \
--do_train=True \
--train_batch_size=128 \
--max_seq_length=256 \
--num_train_steps=1000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
