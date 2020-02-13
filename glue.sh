TPU_NAME='grpc://10.8.246.2:8470'
ELECTRA_GC='gs://electra'
TASK=CoLA

python3 run_classifier.py \
--task_name=$TASK \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/bz1024/$TASK \
--init_checkpoint=$ELECTRA_GC/electra_pretrain/electra_5e-4_bz1024/model.ckpt-125000 \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=256 \
--learning_rate=1e-4 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=12345 \
--use_tpu=True \
--tpu_name=$TPU_NAME