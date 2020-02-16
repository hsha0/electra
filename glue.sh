TPU_NAME='grpc://10.23.181.186:8470'
INIT_CKPT=$ELECTRA_GC/electra_pretrain/electra_h1024_w50/model.ckpt-15000
MODEL=ale
SIZE=small
SEED=12345
TASK_INDEX=0

TASKS=(MRPC CoLA MNLI SST-2 QQP QNLI WNLI RTE STS-B)
LRS=(2e-5 1e-5 3e-5 1e-5 5e-5 1e-5 2e-5 3e-5 2e-5)
BZS=(32 16 128 32 128 32 16 32 16)

TASK=TASK[${TASK_INDEX}]
LR=LRS[${TASK_INDEX}]
BZ=BZS[${TASK_INDEX}]

CONFIG=config/${MODEL}_${SIZE}.json
ELECTRA_GC='gs://electra'

python3 run_classifier.py \
--electra_config_file=$CONFIG \
--task_name=${TASK} \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/h1024_0/$TASK \
--init_checkpoint=$INIT_CKPT \
--vocab_file=vocab.txt \
--model=$MODEL \
--do_train=True \
--do_eval=True \
--train_batch_size=${BZ} \
--learning_rate=${LR} \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=${SEED} \
--use_tpu=True \
--tpu_name=$TPU_NAME
