TPU_NAME='grpc://10.68.243.130:8470'
ELECTRA_GC='gs://electra'
MODEL=bert

python3 run_pretrain.py \
--input_file=$ELECTRA_GC/data_128_sent_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/$MODEL_electra_h768_s128 \
--vocab_file=vocab.txt \
--model=$MODEL \
--do_train=True \
--learning_rate=2e-4 \
--train_batch_size=512 \
--max_seq_length=128 \
--num_train_steps=25000 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
