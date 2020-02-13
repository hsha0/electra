TPU_NAME='grpc://10.34.148.162:8470'
ELECTRA_GC='gs://electra'
#INIT_CKPT=$ELECTRA_GC/electra_pretrain/electra_5e-4_bz1024/model.ckpt-125000
TASK=MRPC

python3 run_classifier.py \
--task_name=$TASK \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/h768_s0/$TASK \
--init_checkpoint=$INIT_CKPT \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=32 \
--learning_rate=0.001 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=$1 \
--use_tpu=True \
--tpu_name=$TPU_NAME
