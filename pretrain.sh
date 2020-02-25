TPU_NAME='grpc://10.123.241.34:8470'
MODEL=electra
SIZE=small
LR=4e-3
DISC_W=50
TRAIN_STEP=125000
TOTAL=125000
WARM_UP=1250
BZ=1024

ELECTRA_GC='gs://electra'
CONFIG=config/${MODEL}_${SIZE}.json
MAX_SEQ_L=128

python3 run_pretrain.py \
--electra_config_file=$CONFIG \
--input_file=$ELECTRA_GC/data_128_sent_CLS/*.tfrecord \
--output_dir=$ELECTRA_GC/electra_pretrain/${MODEL}_${SIZE}_seq${MAX_SEQ_L}_lr${LR}_w${DISC_W}_bz${BZ} \
--vocab_file=vocab.txt \
--disc_loss_weight=${DISC_W} \
--model=$MODEL \
--num_warmup_steps=${WARM_UP} \
--do_train=True \
--learning_rate=${LR} \
--train_batch_size=${BZ} \
--max_seq_length=${MAX_SEQ_L} \
--num_train_steps=${TRAIN_STEP} \
--total_num_train_steps=${TOTAL} \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=True \
--tpu_name=$TPU_NAME
