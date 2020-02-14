TPU_NAME='grpc://10.43.245.90:8470'
ELECTRA_GC='gs://electra'
MODEL=ale
CONFIG='config/ale_small.json'

python3 run_pretrain.py \
--electra_config_file=$CONFIG \
--input_file=$ELECTRA_GC/data_128_sent_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/${MODEL}_h256_l128 \
--vocab_file=vocab.txt \
--model=$MODEL \
--do_train=True \
--learning_rate=5e-5 \
--train_batch_size=1024 \
--max_seq_length=128 \
--num_train_steps=1 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
