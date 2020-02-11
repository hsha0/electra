TPU_NAME='grpc://10.70.112.18:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data_128_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/electra_5e-4_CLS_seg_clip \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=5e-4 \
--train_batch_size=128 \
--max_seq_length=128 \
--num_train_steps=1000000 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=6250 \
--iterations_per_loop=6250 \
--use_tpu=true \
--tpu_name=$TPU_NAME
