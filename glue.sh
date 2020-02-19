ELECTRA_GC='gs://electra'
TPU_NAME='grpc://10.121.225.26:8470'
MODEL=ale
SIZE=small
INIT_CKPT=$ELECTRA_GC/electra_pretrain/${MODEL}_${SIZE}_seq128_lr5e-4/model.ckpt-342000
SEED=12345
TASK_INDEX=5

TASKS=(MRPC CoLA MNLI SST-2 QQP QNLI WNLI RTE STS-B)
LRS=(2e-5 1e-5 3e-5 1e-5 5e-5 1e-5 2e-5 3e-5 2e-5)
BZS=(32 16 128 32 128 32 16 32 16)
EPOCHS=(3 3 3 3 3 3 3 10 10)

TASK=${TASKS[${TASK_INDEX}]}
LR=${LRS[${TASK_INDEX}]}
BZ=${BZS[${TASK_INDEX}]}
EPOCH=${EPOCHS[${TASK_INDEX}]}

CONFIG=config/${MODEL}_${SIZE}.json


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
--seed=${SEED} \
--use_tpu=True \
--tpu_name=$TPU_NAME
