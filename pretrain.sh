TPU_NAME='grpc://10.122.63.170:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data_128_sent_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/electra_h256_bz2048_lr0.001 \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=0.001 \
--train_batch_size=2048 \
--max_seq_length=128 \
--num_train_steps=250000 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=6250 \
--iterations_per_loop=6250 \
--use_tpu=true \
--tpu_name=$TPU_NAME
