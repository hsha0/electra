TPU_NAME='grpc://10.84.229.210:8470'
ELECTRA_GC='gs://electra'
INIT_CKPT=$ELECTRA_GC/electra_pretrain/electra_h1024_w50/model.ckpt-12000
TASK=MNLI

python3 run_classifier.py \
--task_name=$TASK \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/h1024_w50/$TASK \
--init_checkpoint=$INIT_CKPT \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=128 \
--learning_rate=3e-5 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=$1 \
--use_tpu=True \
--tpu_name=$TPU_NAME
