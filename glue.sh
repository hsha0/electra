ELECTRA_GC='gs://electra'
TPU_NAME='grpc://10.77.135.10:8470'
MODEL=electra
SIZE=small
SEED=$$
CKPT=electra_small_seq128_lr2e-3_w50_bz1024
CKPT_NUM=125000
TASK_INDEX=$1

TASKS=(MRPC CoLA MNLI SST-2 QQP QNLI WNLI RTE STS-B)
#LRS=(3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4)
#BZS=(32 32 32 32 32 32 32 32 32)

#LAMB
LRS=(2e-4 3e-4 4e-4 3e-4 1e-5 2e-4 3e-4 3e-4 2e-5)
BZS=(64 256 256 256 1024 256 32 32 16)

#ADAM
#LRS=(4e-5 3e-4 4e-4 3e-4 1e-5 2e-4 3e-4 3e-4 2e-5)
#BZS=(8 256 256 256 1024 256 32 32 16)

EPOCHS=(3 3 3 3 3 3 3 10 10)


TASK=${TASKS[${TASK_INDEX}]}
LR=${LRS[${TASK_INDEX}]}
BZ=${BZS[${TASK_INDEX}]}
EPOCH=${EPOCHS[${TASK_INDEX}]}
#LR=$2
#BZ=$3
#EPOCH=$4
INIT_CKPT=$ELECTRA_GC/electra_pretrain/${CKPT}/model.ckpt-${CKPT_NUM}

CONFIG=config/${MODEL}_${SIZE}.json

echo ${SEED}_${TASK}

python3 run_classifier.py \
--electra_config_file=$CONFIG \
--task_name=${TASK} \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/${MODEL}_${SIZE}/${TASK}_${SEED} \
--init_checkpoint=$INIT_CKPT \
--vocab_file=vocab.txt \
--model=$MODEL \
--do_train=True \
--do_eval=True \
--train_batch_size=${BZ} \
--learning_rate=${LR} \
--max_seq_length=128 \
--num_train_epochs=${EPOCH} \
--use_tpu=True \
--tpu_name=$TPU_NAME

echo ${SEED}_${TASK}
