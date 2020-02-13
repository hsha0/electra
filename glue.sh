TPU_NAME='grpc://10.8.246.2:8470'
ELECTRA_GC='gs://electra'
TASK=MNLI
# INIT_CKPT=$ELECTRA_GC/electra_pretrain/electra_5e-4_bz1024/model.ckpt-125000
INIT_CKPT=gs://bert_sh/bert_pretrain/bert_5e-4/model.ckpt-87500

python3 run_classifier.py \
--task_name=$TASK \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/bert_small/$TASK \
--init_checkpoint= $INIT_CKPT \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=32 \
--learning_rate=3e-4 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=$1 \
--use_tpu=True \
--tpu_name=$TPU_NAME
