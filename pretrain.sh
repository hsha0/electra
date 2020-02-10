TPU_NAME='grpc://10.37.248.170:8470'
ELECTRA_GC='gs://electra'

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data_128/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/electra_6e-4 \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=6e-4 \
--train_batch_size=128 \
--max_seq_length=128 \
--num_train_steps=62500 \
--max_predictions_per_seq=20 \
--use_tpu=true \
--tpu_name=$TPU_NAME
