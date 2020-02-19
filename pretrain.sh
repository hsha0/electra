TPU_NAME='grpc://10.125.243.18:8470'
MODEL=electra
SIZE=small
MAX_SEQ_L=1024
LR=5e-4
DISC_W=1
TRAIN_STEP=62500

ELECTRA_GC='gs://electra'
CONFIG=config/${MODEL}_${SIZE}.json

python3 run_pretrain.py \
--electra_config_file=$CONFIG \
--input_file=$ELECTRA_GC/data_128_sent_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/${MODEL}_${SIZE}_seq${MAX_SEQ_L}_lr${LR}_w${DISC_W} \
--vocab_file=vocab.txt \
--disc_loss_weight=${DISC_W} \
--model=$MODEL \
--do_train=True \
--learning_rate=${LR} \
--train_batch_size=128 \
--max_seq_length=${MAX_SEQ_L} \
--num_train_steps=${TRAIN_STEP} \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
