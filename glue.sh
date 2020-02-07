TPU_NAME='grpc://10.97.148.162:8470'
ELECTRA_GC='gs://electra'
TASK=MNLI

python3 run_classifier.py \
--task_name=$TASK \
--data_dir=$ELECTRA_GC/glue/glue_data/$TASK \
--output_dir=$ELECTRA_GC/glue/glue_results/glue_$TASK \
--init_checkpoint=$ELECTRA_GC/electra_ckpt/model.ckpt-125000 \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=32 \
--learning_rate=3e-4 \
--max_seq_length=256 \
--num_train_epochs=3.0 \
--use_tpu=true \
--tpu_name=$TPU_NAME